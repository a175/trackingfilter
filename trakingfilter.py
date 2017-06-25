import numpy
import cv2

class VideoViewer:
    """
    This impliments some fundumental functions to show video.
    """
    def __init__(self,modify_frame=None):
        self.mouse_event_note={}
        self.mouse_event_result=[]
        self.direction=0
        self.frame_variation_to_show=-1
        self.frames=[]
        self.modify_frame=modify_frame

        self.COLOR_MOUSE_EVENT_SELECTED =(  0,255,255,1)
        self.COLOR_MOUSE_EVENT_SELECTING=(  0,  0,255,1)
        self.COLOR_MOUSE_EVENT_SELECT   =(255,  0,255,1)
        
    def set_original(self,video):
        """
        Set the original video capture 
        video should have been  opened by cv2.VideoCapture(filename).
        """
        self.original=video
        w=self.original.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        h=self.original.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.mouse_event_layer=numpy.zeros((h,w,4))

    def get_current_frame_position(self):
        """
        Returns position of self.frame.
        """
        return self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1

    def set_modify_frame(self,modify_frame):
        """
        Set a function to modify each frame
        """
        self.modify_frame=modify_frame

    def overlap_layer(self,base,layer):
        """
        Returns overlaped image.
        4-6th channels of layer is a value between 0 and 1
        """
        r=numpy.zeros(base.shape)
        m=layer[:,:,3]
        r[:,:,0]=numpy.floor(base[:,:,0]*(-m+1)+layer[:,:,0]*m)
        r[:,:,1]=numpy.floor(base[:,:,1]*(-m+1)+layer[:,:,1]*m)
        r[:,:,2]=numpy.floor(base[:,:,2]*(-m+1)+layer[:,:,2]*m)
        return r.astype(numpy.uint8)
        
    def update_frame(self,pos=None,verbose=True):
        """
        Update frame.
        """
        self.mouse_event_layer[:,:,:]=0
        if not pos is None:
            if pos>=0:
                self.original.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
            else:
                self.direction=0
                self.original.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
                if verbose:
                    print "*This is the first frame.*"
        (ret, frame) = self.original.read()
        if ret:
            frames=[numpy.copy(frame)]
            if not self.modify_frame is None:
                frames = frames+self.modify_frame(frame)
            self.frames=frames
        else:
            self.direction=0
            if verbose:
                print "*This is the final frame.*"

    def show_command_list(self,xkeyeventtitle):
        """
        Prints command list for showvideo()
        """
        print "+","q|c: quit;"
        print "+","f: forward;"
        print "+","b: reverse;"
        print "+","n: pause at next frame;"
        print "+","p: pause at previous frame;"
        print "+","e: 10 n;"
        print "+","a: 10 p;"
        print "+",">: pause at final frame;"
        print "+","<: pause at first frame;"
        print "+","u: rotate stored position and pause at last position;"
        print "+","w: store this frame position (and pause);"
        print "+","o: change variation of modified frame (0 is original);"
        print "+","[: previous step"
        print "+","]: next step"
        print "+","?: show this message"
        if xkeyeventtitle!="":
            print "+","x:", xkeyeventtitle
        print ""

    def showvideo(self,initialpos,windowname,xkeyevent=None, xkeyeventtitle=""):
        """
        Open window of video.
        xkeyevent: function if x is typed.
        If xkeyevnt is not None and x is typed, 
        then xkeyevent() is called.
        """
        self.show_command_list(xkeyeventtitle)
        self.key_w_ring=[initialpos]
        print "frameposition:", initialpos
        self.update_frame(initialpos)
        while(self.original.isOpened()):
            if self.direction>0:
                self.update_frame()
                if self.direction<50:
                    self.direction=self.direction-1
                    pos=self.get_current_frame_position()
                    print "frameposition:", pos
            elif self.direction<0:
                pos=self.get_current_frame_position()
                pos=pos-1
                self.update_frame(pos)
                if self.direction>-50:
                    self.direction=self.direction+1
                    pos=self.get_current_frame_position()
                    print "frameposition:", pos
            frame = self.overlap_layer(self.frames[self.frame_variation_to_show],self.mouse_event_layer)
            cv2.imshow(windowname,frame)
            keyinput=cv2.waitKey(1) & 0xFF
            if keyinput == ord('q'):
                print "Quit"
                return 0
            elif keyinput == ord('c'):
                print "Close"
                return 0
            elif keyinput == ord(']'):
                self.direction=0
                return 1
            elif keyinput == ord('['):
                self.direction=0
                return -1
            elif keyinput == ord('p'):
                self.direction=-1
            elif keyinput == ord('n'):
                self.direction=1
            elif keyinput == ord('a'):
                self.direction=-10
            elif keyinput == ord('e'):
                self.direction=10
            elif keyinput == ord('>'):
                pos=self.original.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)-1
                self.update_frame(pos)
                self.direction=0
                pos=self.get_current_frame_position()
                print "frameposition:", pos
            elif keyinput == ord('<'):
                pos=0
                self.update_frame(pos)
                self.direction=0
                pos=self.get_current_frame_position()
                print "frameposition:", pos
            elif keyinput == ord('u'):
                if self.key_w_ring != []:
                    pos=self.key_w_ring.pop()
                    self.key_w_ring.insert(0,pos)
                    self.update_frame(pos)
                    self.direction=0
                    pos=self.get_current_frame_position()
                    print "frameposition:", pos
                    print "w_ring:",self.key_w_ring
            elif keyinput == ord('f'):
                self.direction=100
                pos=self.get_current_frame_position()
                print "frameposition:", pos, "->"
            elif keyinput == ord('b'):
                self.direction=-100
                pos=self.get_current_frame_position()
                print "frameposition:", pos, "<-"
            elif keyinput == ord('o'):
                self.frame_variation_to_show=self.frame_variation_to_show-1
                if not self.frames is None:
                    self.frame_variation_to_show=self.frame_variation_to_show%len(self.frames)
                print "frame variation:", self.frame_variation_to_show 
            elif keyinput == ord('w'):
                pos=self.get_current_frame_position()
                self.key_w_ring.append(pos)
                self.direction=0
                print "w:", pos
                print "w_ring:",self.key_w_ring
            elif keyinput == ord('?'):
                self.show_command_list(xkeyeventtitle)
            elif keyinput == ord('x'):
                if not xkeyevent is None:
                    print "x:", xkeyeventtitle
                    print "+ x"
                    xkeyevent()
                    print "leave x:", xkeyeventtitle

    def select_circle_by_mouse(self,event, x, y, flags, param):
        """
        Select circle and append to self.inputted_data.
        This is used as mouse event callback.
        """
        if event == cv2.EVENT_LBUTTONUP:
            (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
            self.direction=self.mouse_event_note["direction"]
            self.mouse_event_layer=numpy.copy(self.mouse_event_note["frame"])
            self.mouse_event_note={}
            pos=self.get_current_frame_position()
            self.inputted_data.append((pos,xp,x,yp,y))
            radius=numpy.sqrt((xp-x)**2 + (yp-y)**2)
            cv2.circle(self.mouse_event_layer, (xp, yp), 1,self.COLOR_MOUSE_EVENT_SELECTED, -1, 8, 10)
            cv2.circle(self.mouse_event_layer, (xp,yp), int(radius),self.COLOR_MOUSE_EVENT_SELECTED, 1)
            print "circle: ", ((xp, yp), (x,y))
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=self.mouse_event_layer

            self.direction=0
        elif  event == cv2.EVENT_MOUSEMOVE:
            if "LBUTTONDOWN" in self.mouse_event_note:
                (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
                self.mouse_event_layer=numpy.copy(self.mouse_event_note["frame"])
                radius=numpy.sqrt((xp-x)**2 + (yp-y)**2)
                cv2.circle(self.mouse_event_layer, (xp,yp), 1,self.COLOR_MOUSE_EVENT_SELECTING, -1, 8, 10)
                cv2.circle(self.mouse_event_layer, (xp,yp), int(radius),self.COLOR_MOUSE_EVENT_SELECTING, 1)
                
    def select_rectangle_by_mouse(self,event, x, y, flags, param):
        """
        Select rectangle and append to self.inputted_data.
        This is used as mouse event callback.
        """
        if event == cv2.EVENT_LBUTTONUP:
            (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
            self.direction=self.mouse_event_note["direction"]
            self.mouse_event_layer=numpy.copy(self.mouse_event_note["frame"])
            self.mouse_event_note={}
            pos=self.get_current_frame_position()
            self.inputted_data.append((pos,xp,x,yp,y))
            cv2.rectangle(self.mouse_event_layer, (xp, yp), (x,y),self.COLOR_MOUSE_EVENT_SELECTED, 1)
            print "rectangle: ", ((xp, yp), (x,y))            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=self.mouse_event_layer
            self.direction=0
        elif  event == cv2.EVENT_MOUSEMOVE:
            if "LBUTTONDOWN" in self.mouse_event_note:
                (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
                self.mouse_event_layer=numpy.copy(self.mouse_event_note["frame"])
                cv2.rectangle(self.mouse_event_layer, (xp, yp), (x,y),self.COLOR_MOUSE_EVENT_SELECTING, 1)


    def select_point_by_mouse(self,event, x, y, flags, param):
        """
        Select point and append to self.inputted_data.
        This is used as mouse event callback.
        """
        if event == cv2.EVENT_LBUTTONUP:
            self.mouse_event_layer=numpy.copy(self.mouse_event_note["frame"])
            (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
            self.mouse_event_note={}
            cv2.rectangle(self.mouse_event_layer, (xp-1, yp-1), (xp+1,yp+1),self.COLOR_MOUSE_EVENT_SELECTED, 2)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=numpy.copy(self.mouse_event_layer)
            self.mouse_event_layer=numpy.copy(self.mouse_event_layer)
            cv2.rectangle(self.mouse_event_layer, (x-1, y-1), (x+1,y+1), self.COLOR_MOUSE_EVENT_SELECT, 1)
            self.direction=0
            print "point: ", (x,y)
            pos=self.get_current_frame_position()
            self.inputted_data.append((pos,x,y))


class PartcleFilter(VideoViewer):
    def __init__(self):
        VideoViewer.__init__(self)
        self.filter_list={}
    
    def reset_filter_position(self):
        """
        Reset positions to apply filter.
        """
        self.filter_position={}
        
    def append_filter_position(self,pos,x,y,r):
        """
        Append positions and radius of filter to apply filter.
        pos: frame number to apply
        x, y: coordinate (of center of filter).
        r: key of the dictionary of filters.
        """
        if not pos in self.filter_position:
            self.filter_position[pos]=[]
        self.filter_position[pos].append((x,y,r))

    def save_modified_video(self,outfile,verbose=-1):
        """
        Save the modified video as outfile.
        verbose: Comment each 100/verbose %  if verbose>0
                 Comment each frame  if verbose=0
                 Ommit comment if verbose<0
        """        
        fps=self.original.get(cv2.cv.CV_CAP_PROP_FPS)
        n_frames=self.original.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        if verbose>0:
            step=n_frames/verbose
        else:
            step=0
        flag=step
        w=self.original.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        h=self.original.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        fourcc = cv2.cv.CV_FOURCC(*'XVID')
        out = cv2.VideoWriter(outfile,fourcc,fps, (int(w),int(h)))
        
        self.original.set(cv2.cv.CV_CAP_PROP_POS_FRAMES,0)
        pos=0
        while(self.original.isOpened()):
            (ret, frame) = self.original.read()
            if ret:
                if pos in self.filter_position:
                    frame=self.apply_filters(frame,self.filter_position[pos])
                out.write(frame)
            else:
                break
            pos=pos+1
            if verbose>=0:
                if pos > flag:
                    flag=flag+step
                    print "+ Done ", pos,"/",n_frames
        if verbose>=0:
            print "+ Done."
        out.release()
        self.original.release()

    def apply_filters(self,frame,filters):
        """
        Apply the filters to frame.
        """        
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sh=frame[:,:,0].shape
        fl=numpy.zeros(sh)
        ma=numpy.zeros(sh)
        for (y,x,r) in filters:
            f=self.filter_list[r].get_image(x,y,sh[0],sh[1])
            m=self.filter_list[r].get_mask(x,y,sh[0],sh[1])
            (x0,x1,y0,y1)=self.filter_list[r].get_box(x,y,sh[0],sh[1])
            fl[x0:x1,y0:y1]=(fl[x0:x1,y0:y1]+f)
            ma[x0:x1,y0:y1]=(ma[x0:x1,y0:y1]+m)
        ma=numpy.abs(ma-0.5)+0.5
        fl=fl/ma
        fl=cv2.GaussianBlur(fl,(5,5),0)
        frame[:,:,0]=(frame[:,:,0]+fl[:,:])%180
        frame=cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        return frame

    def append_filter_to_dict(self,radius):
        """
        Create a stamp with the radius.
        Append it to the dictionary with key `radius`.
        """        
        r=int(radius)
        self.filter_list[r]=CircleStamp(r,60)
    
class CircleStamp:
    def __init__(self,radius,color):
        self.radius=radius
        self.image=numpy.array([[ color*self.filter_weight(i-radius,j-radius,radius)  for j in range(2*radius)] for i in range(2*radius)],numpy.float32)
        (ret,m)=cv2.threshold(self.image,1,255,cv2.THRESH_TRUNC)
        self.mask=m
        
    def filter_weight(self,x,y,r):
        d=numpy.sqrt(x**2+y**2)
        if d>r:
            w=0
        else:
            w=1
        return w

    def get_crop_box(self,x,y,width,height):
        x0=max(self.radius-x,0)
        y0=max(self.radius-y,0)
        x1=min(2*self.radius,self.radius+width-x)
        y1=min(2*self.radius,self.radius+height-y)
        return (x0,x1,y0,y1)
                     
    def get_image(self,x,y,width,height):
        (x0,x1,y0,y1)=self.get_crop_box(x,y,width,height)
        return self.image[x0:x1,y0:y1]

    def get_mask(self,x,y,width,height):
        (x0,x1,y0,y1)=self.get_crop_box(x,y,width,height)
        return self.mask[x0:x1,y0:y1]

    def get_box(self,x,y,width,height):
        x0=max(x-self.radius,0)
        y0=max(y-self.radius,0)
        x1=min(x+self.radius,width)
        y1=min(y+self.radius,height)
        return (x0,x1,y0,y1)


class TrackingFilter(PartcleFilter):
    """
    Change the color of some particles in the video.
    """
    def __init__(self):
        PartcleFilter.__init__(self)
        self.bg=[]
        self.template=[]
        self.set_masktype("1")
        self.set_templatematchingtype("apply")
        self.features=None
        self.CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        
    def reset_traced_particle(self):
        self.traced_particle=traced_particle()
    
    def get_gray_image_for_template_match(self,frame):
        """
        Returns the gray image for template match.
        Input image should BGR.
        The format of output is defined by self.gray_type:
        """
        
        if self.gray_type=="GRAY":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif self.gray_type=="BGR:b":
            return numpy.copy(frame)[:,:,0]
        elif self.gray_type=="BGR:g":
            return numpy.copy(frame)[:,:,1]
        elif self.gray_type=="BGR:r":
            return numpy.copy(frame)[:,:,2]
        elif self.gray_type=="HSV:h":
            hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return hsv[:,:,0]
        elif self.gray_type=="HSV:s":
            hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return hsv[:,:,1]
        elif self.gray_type=="HSV:v":
            hsv=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            return hsv[:,:,2]
    
    def get_gray_image_for_optical_flow(self,frame):
        """"
        Returns gray image to calculate optical flow.
        * grayize by get_gray_image_for_template_match.
        * template match 
        * mask
        """
        g=self.get_gray_image_for_template_match(frame)
        mask=self.get_mask_for_opticalflow(frame)
        g=self.apply_template_match(g)
        g=self.apply_mask(g,mask)
        return g

    def set_outfile(self,outfilename):
        """
        Set the filename to output the modified video.
        """
        self.outfile = outfilename        

    def append_bg(self,pos):
        """
        Append the image of the position to backgounds.
        """
        self.original.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
        (ret, frame) = self.original.read()
        self.bg.append(frame)

    def set_gray_type(self,t):
        """
        Set the gray image type for template matching.
        """
        self.gray_type=t
        
    def set_masktype(self,t):
        """
        Set mask or not after template matching.
        """
        self.masktype=t
                    
    def set_templatematchingtype(self,t):
        """
        Set apply or not template matching.
        """
        self.templatematchingtype=t        
    
    def reset_template(self):
        """
        Reset the images of templates for template matching.
        """
        self.template=[]
        
    def append_template(self,pos,xp,x,yp,y):
        """
        Append an image as templates for template matching.
        """
        self.original.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
        (ret, frame) = self.original.read()
        gray = self.get_gray_image_for_template_match(frame)
        self.template.append(gray[yp:y,xp:x])

    def set_radius_of_filter(self,radius):
        """
        Append a filter with radius to dictionary.
        And set radius of filter to use.
        """
        self.radius_of_filter=int(radius)
        self.append_filter_to_dict(self.radius_of_filter)

    def add_or_remove_from_filterposition(self, pos, x, y, r):
        """
        If 
        the distance of between a point in self.filter and (x,y)
        is less that r,
        then remove it;
        otherwise append it.
        """
        if pos in self.filter_position:
            for (i,(xi,yi,ri)) in enumerate(self.filter_position[pos]):
                if (x-xi)**2+(y-yi)**2<r**2:
                    self.filter_position[pos].pop(i)
                    return ((xi,yi),ri,-1)
        self.append_filter_position(pos,x,y,self.radius_of_filter)
        return ((x,y),self.radius_of_filter,1)
        
    def get_mask_for_opticalflow(self,frame):
        """
        Returns mask for gray image.
        """
        if self.bg==[]:
            return None
        operator = numpy.ones((3, 3), numpy.uint8)
        gray = self.get_gray_image_for_template_match(frame)
        bg = self.get_gray_image_for_template_match(self.bg[-1])
        temp =cv2.absdiff(gray,bg)
        temp = cv2.threshold(temp, 15, 255, cv2.THRESH_BINARY)[1]
        temp = cv2.dilate(temp, operator, iterations=4)
        temp = cv2.erode(temp, operator, iterations=4)
        temp = cv2.dilate(temp, operator, iterations=4)
        temp = cv2.erode(temp, operator, iterations=4)
        mask = temp
        return mask
        
    def apply_mask(self,frame,mask):
        """
        Apply the mask
        """
        if not mask is None:
            if self.masktype=="1":
                return cv2.bitwise_or(frame,cv2.bitwise_not(mask))
            elif self.masktype=="0":
                return cv2.bitwise_and(frame,mask)
        return numpy.copy(frame)

    def apply_template_match(self, gray):
        """
        Apply template matching.
        """
        r = numpy.copy(gray)
        if self.templatematchingtype=="apply":
            ss = r.shape
            n = len(self.template)
            for template in self.template:
                res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
                s=res.shape
                d0=(ss[0]-s[0])//2
                d1=(ss[1]-s[1])//2
                r[d0:s[0]+d0,d1:s[1]+d1]=r[d0:s[0]+d0,d1:s[1]+d1]+255*res/n
        return r
            
    def update_traced_particle_if_needed(self,frame_p,frame_c,pos_pre,pos_cur):
        """
        If version is old then calculate optical flow.
        """
        pos=self.get_current_frame_position()
        idnum=[]
        pt_prev=[]
        for (i,pti) in self.traced_particle.x_particles_to_update(pos_pre,pos_cur):
            pt_prev.append(pti)
            idnum.append(i)
        if pt_prev != []:
            pt_cur=self.calc_optical_flow_between_frames(frame_p,frame_c,pt_prev)
            for (idi,pti) in zip(idnum,pt_cur):
                self.traced_particle.update_particle(idi,pos_pre,pos_cur,pti)
        self.traced_particle.cleanup_particle(pos_pre,pos_cur)

    def calc_optical_flow_between_frames(self,frame_p,frame_c,pts):
        """
        Calculate optical flows.
        """
        gray_c = self.get_gray_image_for_optical_flow(frame_c)
        gray_p = self.get_gray_image_for_optical_flow(frame_p)
        (features,status,err)=cv2.calcOpticalFlowPyrLK(gray_p,gray_c,numpy.array(pts, numpy.float32),None,winSize = (10, 10), maxLevel = 5,criteria = self.CRITERIA,flags = 0)
        if features is None:
            return []
        return [ None if status[i] == 0 else (pti[0],pti[1]) for (i,pti) in enumerate(features)]

class TrackingFilterUI(TrackingFilter):
    """
    UI of TrackingFilter
    """
    def __init__(self):
        TrackingFilter.__init__(self)
        self.runlevel=[
            self.input_filename,
            self.select_grayimage,
            self.select_background,
            self.select_masktype,
            self.select_template,
            self.select_templatematchingtype,
            self.select_filter_radius,
            self.select_and_trace_particle,
            self.check_filter
        ]
        self.frame_prev=None
        
        self.COLOR_PARTICLE_BOUNDARY     =(15, 100, 200)
        self.COLOR_PARTICLE_BOUNDARY_EMPH=(15, 250, 100)

    def get_gradation_color_a(self,num,tot):
        return (20+(200//tot)*num,220-(200//tot)*num,70)
    
    def get_gradation_color_b(self,num,tot):
        return (20+(200//tot)*num,70,220-(200//tot)*num)

    def input_filename(self):
        print "Input filenames to use."
        print ""
        print "Filename of original video:"
        video=raw_input()
        print "Filename to save the modified video:"
        outfile=raw_input()
        original=cv2.VideoCapture(video)
        self.set_original(original)
        self.set_outfile(outfile)
        print ""
        print "``` python"
        print "import numpy"
        print "import cv2"
        print "import trackingfilter"
        print ""
        print "tf=trackingfilter.TrackingFilterUI()"
        print "original=cv2.VideoCapture(\"%s\")" % video
        print "tf.set_original(original)"
        print "tf.set_outfile(\"%s\")" % outfile
        print "```"
        print ""
        return 1

    def modify_frame_to_select_grayimage(self,oframe):
        modifiedframes=[]
        frame=numpy.copy(oframe)
        g=self.get_gray_image_for_template_match(frame)
        frame[:,:,0]=g
        frame[:,:,1]=g
        frame[:,:,2]=g
        modifiedframes.append(numpy.copy(frame))
        return modifiedframes

    def select_grayimage(self):
        print "Choose single channel image."
        print ""
        self.set_gray_type("GRAY")
        self.frame_variation_to_show=-1
        self.set_modify_frame(self.modify_frame_to_select_grayimage)
        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
        flag=self.showvideo(0, "frame",
                            xkeyevent=self.select_grayimage_by_key,
                            xkeyeventtitle="choose gray type")
        cv2.destroyWindow("frame")
        if flag <= 0:
            return flag
        print ""
        print "``` python"
        print "tf.set_gray_type(\"%s\")" % self.gray_type
        print "```"
        print ""
        return flag


    def select_background(self):
        print "Choose backgroud frames."
        print "The last frame of w_ring is used as background."
        print ""
        self.frame_variation_to_show=-1
        self.set_modify_frame(self.modify_frame_to_select_grayimage)
        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
        flag=self.showvideo(0,"frame")
        cv2.destroyWindow("frame")
        if flag <= 0:
            return flag
        pos=self.key_w_ring[-1]
        print ""
        print "``` python"
        print "tf.append_bg(",pos,")"
        print "```"
        print ""
        self.append_bg(pos)
        return flag

    def select_grayimage_by_key(self):
        print "++ g :gray scale"
        print "++ r :one of channels of BGR"
        print "++ h :one of channels of HSV"
        keyinput=cv2.waitKey(0) & 0xFF
        if keyinput == ord('g'):
            self.set_gray_type("GRAY")
            print "graytype: gray"
        elif keyinput == ord('r'):
            keyinput=cv2.waitKey(0) & 0xFF
            print "+++ r :R channel of BGR"
            print "+++ g :G channel of BGR"
            print "+++ b :G channel of BGR"
            if keyinput == ord('r'):
                self.set_gray_type("BGR:r")
                print "graytype: r of BGR"
            elif keyinput == ord('g'):
                self.set_gray_type("BGR:g")
                print "graytype: g of BGR"
            elif keyinput == ord('b'):
                self.set_gray_type("BGR:b")
                print "graytype: b of BGR"
        elif keyinput == ord('h'):
            keyinput=cv2.waitKey(0) & 0xFF
            print "+++ h :H channel of HSV"
            print "+++ s :S channel of HSV"
            print "+++ v :V channel of HSV"
            if keyinput == ord('h'):
                self.set_gray_type("HSV:h")
                print "graytype: h of HSV"
            elif keyinput == ord('s'):
                self.set_gray_type("HSV:s")
                print "graytype: s of HSV"
            elif keyinput == ord('v'):
                self.set_gray_type("HSV:v")
                print "graytype: v of HSV"
        pos=self.get_current_frame_position()
        self.update_frame(pos)

    def modify_frame_to_select_masktype(self,oframe):
        modifiedframes=[]
        frame=numpy.copy(oframe)
        mask=self.get_mask_for_opticalflow(frame)
        g=self.get_gray_image_for_template_match(frame)
        frame[:,:,0]=g
        frame[:,:,1]=g
        frame[:,:,2]=g
        modifiedframes.append(numpy.copy(frame))
        g=self.apply_mask(g,mask)
        frame[:,:,0]=g
        frame[:,:,1]=g
        frame[:,:,2]=g
        modifiedframes.append(numpy.copy(frame))
        return modifiedframes
                
    def select_masktype(self):
        print "Choose mask type for background."
        print ""
        self.frame_variation_to_show=-1
        self.set_modify_frame(self.modify_frame_to_select_masktype)
        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
        flag=self.showvideo(0,"frame",
                            xkeyevent=self.select_masktype_by_key,
                            xkeyeventtitle="choose mask type")
        cv2.destroyWindow("frame")
        if flag <= 0:
            return flag
        print ""
        print "``` python"
        print "tf.set_masktype(\"%s\")" % self.masktype
        print "```"
        print ""
        return flag

    def select_masktype_by_key(self):
        print "++ n :No mask"
        print "++ 0 :Mask by black"
        print "++ 1 :Mask by white"
        keyinput=cv2.waitKey(0) & 0xFF
        if keyinput == ord('n'):
            self.set_masktype("None")
            print "masktype: None"
        elif keyinput == ord('0'):
            self.set_masktype("0")
            print "masktype: white",0
        elif keyinput == ord('1'):
            self.set_masktype("1")
            print "masktype: black",1
        pos=self.get_current_frame_position()
        self.update_frame(pos)

    def select_template(self):
        print "Choose template image for template matching."
        print ""
        print "+","mouse drag: append the rectangle;"
        self.event_x=None
        self.inputted_data=[]
        self.frame_variation_to_show=-1
        self.set_modify_frame(self.modify_frame_to_select_grayimage)

        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
        cv2.setMouseCallback("frame", self.select_rectangle_by_mouse)
        flag=self.showvideo(0,"frame")
        cv2.destroyWindow("frame")
        if flag <= 0:
            return flag
        
        templates=[]
        for (pos,xp,x,yp,y) in self.inputted_data:
            if yp>y:
                t=yp
                yp=y
                y=t
            if xp>x:
                t=xp
                xp=x
                x=t
            templates.append((pos,xp,x,yp,y))
        print ""
        print "``` python"
        self.reset_template()
        print "tf.reset_template()"
        for (pos,xp,x,yp,y) in templates:
            self.append_template(pos,xp,x,yp,y)
            print "tf.append_template(",pos,",",xp,",",x,",",yp,",",y,")"
        print "```"
        print ""
        return flag

    def modify_frame_to_select_templatematchingtype(self,oframe):
        modifiedframes=[]
        frame=numpy.copy(oframe)
        g=self.get_gray_image_for_template_match(frame)
        mask=self.get_mask_for_opticalflow(frame)
        g=self.apply_mask(g,mask)
        frame[:,:,0]=g
        frame[:,:,1]=g
        frame[:,:,2]=g
        modifiedframes.append(numpy.copy(frame))

        frame=numpy.copy(oframe)
        g=self.get_gray_image_for_optical_flow(frame)
        frame[:,:,0]=g
        frame[:,:,1]=g
        frame[:,:,2]=g
        modifiedframes.append(numpy.copy(frame))

        return modifiedframes

    def select_templatematchingtype(self):
        print "Choose apply template matching or not."
        print ""
        self.frame_variation_to_show=-1
        self.set_modify_frame(self.modify_frame_to_select_templatematchingtype)


        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
        flag=self.showvideo(0,"frame",
                            xkeyevent=self.select_templatematchingtype_by_key,
                            xkeyeventtitle="choose apply template matching or not")
        cv2.destroyWindow("frame")
        if flag <= 0:
            return flag
        print ""
        print "``` python"
        print "tf.set_templatematchingtype(\"%s\")" % self.templatematchingtype
        print "```"
        print ""
        return flag

    def select_templatematchingtype_by_key(self):
        print "++ n :Not apply"
        print "++ a :Apply"
        keyinput=cv2.waitKey(0) & 0xFF
        if keyinput == ord('n'):
            self.set_templatematchingtype("None")
            print "Template matching: not apply"
        elif keyinput == ord('a'):
            self.set_templatematchingtype("apply")
            print "Template matching: apply"
        pos=self.get_current_frame_position()
        self.update_frame(pos)

    def select_filter_radius(self):
        print "Choose radius of filter."
        print ""
        print "+","mouse drag: append the circle;"
        self.event_x=None
        self.inputted_data=[]
        self.frame_variation_to_show=-1
        self.set_modify_frame(self.modify_frame_to_select_grayimage)

        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
        cv2.setMouseCallback("frame", self.select_circle_by_mouse)
        flag=self.showvideo(0,"frame")
        cv2.destroyWindow("frame")
        if flag <= 0:
            return flag
        if self.inputted_data == []:
            rr=[10]
        else:
            rr=[numpy.sqrt((xp-x)**2 + (yp-y)**2) for (pos,xp,x,yp,y) in self.inputted_data]
        print ""
        print "``` python"
        for radius in rr:
            self.set_radius_of_filter(radius)
            print "tf.set_radius_of_filter(",radius,")"
        print "```"
        print ""
        return flag

    def modify_frame_to_select_and_trace_particle(self,oframe):
        modifiedframes=[]
        pos=self.get_current_frame_position()
        if not self.frame_prev is None:
            if self.direction != 0:
                self.update_traced_particle_if_needed(self.frame_prev,oframe,self.pos_pre,pos)
        self.frame_prev=numpy.copy(oframe)
        self.pos_pre=pos
        
        frame=numpy.copy(oframe)
        g=self.get_gray_image_for_optical_flow(frame)
        frame[:,:,0]=g
        frame[:,:,1]=g
        frame[:,:,2]=g
        modifiedframes.append(numpy.copy(frame))
        frame=self.draw_tracing_lines(frame,pos)
        frame=self.draw_traced_particle(frame,pos)
        modifiedframes.append(numpy.copy(frame))
        frame=numpy.copy(oframe)
        modifiedframes.append(numpy.copy(frame))
        frame=self.draw_tracing_lines(frame,pos)
        frame=self.draw_traced_particle(frame,pos)
        modifiedframes.append(numpy.copy(frame))
        return modifiedframes

        
    def draw_tracing_lines(self,frame,pos):
        """
        Draw lines between pair of racked points in pos to frame.
        """
        for (i,pos_0) in enumerate(self.traced_particle.initial_pos):
            pt0=self.traced_particle.get_point_int(i,pos)
            if pt0 is None:
                continue
            if pos>pos_0:
                for tt in range(5):
                    pt1=self.traced_particle.get_point_int(i,pos-(tt+1))
                    if pt1 is None:
                        continue
                    cv2.line(frame,pt0,pt1,self.get_gradation_color_a(tt,5),1)
                    pt0=pt1
            pt0=self.traced_particle.get_point_int(i,pos)
            if pos<pos_0:
                for tt in range(5):
                    pt1=self.traced_particle.get_point_int(i,pos+(tt+1))
                    if pt1 is None:
                        continue
                    cv2.line(frame,pt0,pt1,self.get_gradation_color_b(tt,5),1)
                    pt0=pt1
        return frame


    def draw_traced_particle(self,frame,pos):
        """
        Draw circle at points in pos to frame.
        """
        for (i,pt,sh) in self.traced_particle.x_particles(pos):
            if sh is None:
                cv2.line(frame,(pt[0]-5, pt[1]-5),(pt[0]+5, pt[1]+5),self.COLOR_PARTICLE_BOUNDARY,1)
                cv2.line(frame,(pt[0]-5, pt[1]+5),(pt[0]+5, pt[1]-5),self.COLOR_PARTICLE_BOUNDARY,1)
            else:
                cv2.circle(frame,pt, 1,self.COLOR_PARTICLE_BOUNDARY, -1, 8, 10)
                cv2.circle(frame,pt,sh,self.COLOR_PARTICLE_BOUNDARY, 1)
                if not self.traced_particle.get_close_particle(pos,pt[0],pt[1],5,exceptid=[i]) is None:
                    cv2.circle(frame,pt,self.radius_of_filter,self.COLOR_PARTICLE_BOUNDARY_EMPH,1)
        return frame

    def select_and_trace_particle(self):
        print "Select particles and trace them."
        print ""
        flag=-1
        self.frame_variation_to_show=-1
        self.set_modify_frame(self.modify_frame_to_select_and_trace_particle)
        self.reset_traced_particle()
        print "+","mouse drag: move/remove particle (If particle is moved to the left top corner of window, then it will be removed.);"
        print "+","mouse click: add new particle;"
        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
        cv2.setMouseCallback("frame", self.edit_traced_particle_by_mouse)
        flag=self.showvideo(0,"frame")
        cv2.destroyWindow("frame")
        if flag <= 0:
            return flag
        
        ff=[ (pos,pt,sh) for (i,pos, pt,sh) in self.traced_particle.x_all_alive_particles()]
        self.reset_filter_position()
        for (pos,(x,y),sh) in ff:
            self.append_filter_position(pos,x,y,sh)
        print ""
        print "``` python"
        print "tf.reset_filter_position()"
        print "ff=",ff
        print "for (pos,(x,y),sh) in ff:"
        print "    tf.append_filter_position(pos,x,y,sh)"
        print ""
        print "```"
        print ""
        return flag
        pass

    def edit_traced_particle_by_mouse(self,event, x, y, flags, param):
        """
        Select particle to add or remove.
        This is used as mouse event callback.
        """
        
        if event == cv2.EVENT_LBUTTONUP:
            self.mouse_event_layer=numpy.copy(self.mouse_event_note["frame"])
            (dx,dy)=self.mouse_event_note["difference"]
            (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
            (idnum,pos)=self.mouse_event_note["original"]
            (xi,yi)=self.traced_particle.get_point_int(idnum,pos)
            self.mouse_event_note={}
            if x+y<self.radius_of_filter:
                pt=(xi,yi)
                print "remove",idnum,pt,"."
                self.traced_particle.set_particle(idnum,pos,pt,None)
            else:
                pt=(x+dx,y+dy)
                if idnum<0:
                    print "add at", pt, "."
                else:
                    print "move",idnum,"to", pt, "."
                self.traced_particle.set_particle(idnum,pos,pt,self.radius_of_filter)
            self.update_frame(pos)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.direction=0
            pos=self.get_current_frame_position()
            idnum=self.traced_particle.get_close_particle(pos,x,y,self.radius_of_filter)
            if idnum is None:
                self.traced_particle.append_new_sequence_of_particle(pos,(x,y),self.radius_of_filter)
                idnum=-1
                xi=x
                yi=y
            else:
                (xi,yi)=self.traced_particle.get_point_int(idnum,pos)
            dx=xi-x
            dy=yi-y
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["original"]=(idnum,pos)
            self.mouse_event_note["difference"]=(dx,dy)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=numpy.copy(self.mouse_event_layer)
            cv2.rectangle(self.mouse_event_layer, (xi-1, yi-1), (xi+1,yi+1),self.COLOR_MOUSE_EVENT_SELECT, 1)
            cv2.circle(self.mouse_event_layer, (x+dx,y+dy), int(self.radius_of_filter),self.COLOR_MOUSE_EVENT_SELECT, 1)
            cv2.line(self.mouse_event_layer,(int(self.radius_of_filter),0),(0,int(self.radius_of_filter)),self.COLOR_MOUSE_EVENT_SELECT,1)
        elif  event == cv2.EVENT_MOUSEMOVE:
            if not "LBUTTONDOWN" in self.mouse_event_note:
                return
            self.mouse_event_layer=numpy.copy(self.mouse_event_note["frame"])
            if "original" in self.mouse_event_note:
                (idnum,pos)=self.mouse_event_note["original"]
                (xi,yi)=self.traced_particle.get_point_int(idnum,pos)
                cv2.rectangle(self.mouse_event_layer, (xi-1, yi-1), (xi+1,yi+1),self.COLOR_MOUSE_EVENT_SELECT, 1)
                cv2.circle(self.mouse_event_layer, (xi,yi), int(self.radius_of_filter),self.COLOR_MOUSE_EVENT_SELECT, 1)
            if "difference" in self.mouse_event_note:
                (dx,dy)=self.mouse_event_note["difference"]
                cv2.circle(self.mouse_event_layer, (x+dx,y+dy), 1,self.COLOR_MOUSE_EVENT_SELECTING, -1, 8, 10)
                cv2.circle(self.mouse_event_layer, (x+dx,y+dy), int(self.radius_of_filter),self.COLOR_MOUSE_EVENT_SELECTING, 1)
            cv2.line(self.mouse_event_layer,(int(self.radius_of_filter),0),(0,int(self.radius_of_filter)),self.COLOR_MOUSE_EVENT_SELECT,1)

    def modify_frame_to_check_filter(self,oframe):
        modifiedframes=[]
        frame=numpy.copy(oframe)
        pos=self.get_current_frame_position()
        if pos in self.filter_position:
            frame=self.apply_filters(oframe,self.filter_position[pos])
        modifiedframes.append(numpy.copy(frame))

        frame=self.draw_filter_position(frame,pos)
        modifiedframes.append(numpy.copy(frame))
        return modifiedframes

    def check_filter(self):
        self.frame_variation_to_show=-1
        self.set_modify_frame(self.modify_frame_to_check_filter)

        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
        cv2.setMouseCallback("frame", self.edit_filterposition_by_mouse)
        flag=self.showvideo(0,"frame")
        cv2.destroyWindow("frame")
        if flag <= 0:
            return flag
        print ""
        print "``` python"
        print "tf.reset_filter_position()"
        for pos in self.filter_position:
            print "ff=",self.filter_position[pos]
            print "for (x,y,r) in ff:"
            print "    tf.append_filter_position(",pos,",x,y,r,)"
            print ""
        print "```"
        print ""

        print "Save modified video as", self.outfile
        self.save_modified_video(self.outfile,10)
        print ""
        print ""
        print "``` python"
        print "save_modified_video()"
        print "```"
        print ""
        print "If you have the ffmpeg, then you may run ffmpeg to convert."
        print "``` shell"
        print "# ffmpeg -i",self.outfile, "-f mp4 -vcodec h264 -qscale 20 -acodec aac -ab 128", self.outfile+".mp4"
        print "```"
        print ""
        return flag

    def run(self,initstep=0):
        """
        This function runs functions in the list self.runlevel,
        which is local variable.
        If the function returns positive,
        then runs next function.
        If the function returns negative,
        then runs previouts function.
        If the function returns zero,
        then exit.
        """
        currentstep=initstep
        while 0 <= currentstep and currentstep < len(self.runlevel):
            print "Section", currentstep
            print "==========="
            flag=self.runlevel[currentstep]()
            if flag>0:
                currentstep=currentstep+1
            elif flag<0:
                currentstep=currentstep-1
                if currentstep<0:
                    currentstep=0
            else:
                print "Exit at Step", currentstep
                print ""
                print "``` python"
                print "tf.run(",currentstep,")"
                print "```"
                print ""
                break
            print "\n***\n"

        print "end."
        
    def edit_filterposition_by_mouse(self,event, x, y, flags, param):
        """
        Select point to add or remove.
        This is used as mouse event callback.
        """
        pos=self.get_current_frame_position()
        if event == cv2.EVENT_LBUTTONUP:
            self.mouse_event_layer=numpy.copy(self.mouse_event_note["frame"])
            (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
            self.mouse_event_note={}
            (pt,r,f)=self.add_or_remove_from_filterposition(pos,xp,yp,4)
            if f>0:
                print "add", x,y, "."
                cv2.circle(self.mouse_event_layer, pt, 1,self.COLOR_MOUSE_EVENT_SELECTED, -1, 8, 10)
                cv2.circle(self.mouse_event_layer, pt, self.radius_of_filter,self.COLOR_MOUSE_EVENT_SELECTED, 1)
            else:
                print "remove", x,y, ".", pt
                cv2.circle(self.mouse_event_layer, pt, 1,self.COLOR_MOUSE_EVENT_SELECTED, -1, 8, 10)
                cv2.circle(self.mouse_event_layer, pt,  self.radius_of_filter,self.COLOR_MOUSE_EVENT_SELECTED, 1)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=numpy.copy(self.mouse_event_layer)
            cv2.rectangle(self.mouse_event_layer, (x-1, y-1), (x+1,y+1),self.COLOR_MOUSE_EVENT_SELECT, 1)
            self.direction=0
            print "point: ", (x,y)
            self.inputted_data.append((pos,x,y))

    def draw_filter_position(self,frame,pos):
        """
        Draw circle at points in pos to frame.
        """
        if pos in self.filter_position:
            for (x,y,r) in self.filter_position[pos]:
                cv2.circle(frame, (x,y), 1,self.COLOR_PARTICLE_BOUNDARY, -1, 8, 10)
                cv2.circle(frame, (x,y), r,self.COLOR_PARTICLE_BOUNDARY, 1)
        return frame

class traced_particle:
    def __init__(self):
        self.points=[]
        self.shape=[]
        self.version=[]
        self.initial_pos=[]
        self.current_version=0
        
    def append_new_sequence_of_particle(self,pos,pt,sh):
        self.points.append({})
        self.shape.append({})
        self.version.append({})
        self.initial_pos.append(pos)
        self.set_particle(-1, pos, pt, sh)
        
    def set_particle(self, idnum, pos, pt, sh):
        self.current_version=self.current_version+1
        self.points[idnum][pos]=pt
        self.shape[idnum][pos]=sh
        self.version[idnum][pos]=self.current_version

    def update_particle(self, idnum, pos_pre, pos_cur, pt):
        self.version[idnum][pos_cur]=self.version[idnum][pos_pre]
        if pt is None:
            self.points[idnum][pos_cur]=self.points[idnum][pos_pre]
            self.shape[idnum][pos_cur]=None
        else:
            self.points[idnum][pos_cur]=pt
            self.shape[idnum][pos_cur]=self.shape[idnum][pos_pre]

    def cleanup_particle(self, pos_pre, pos_cur):
        for (pti,shi,vi,pos_0_i) in zip(self.points,self.shape,self.version,self.initial_pos):
            if (pos_cur-pos_0_i)*(pos_cur-pos_pre)<=0:
                continue
            if not pos_cur in vi:
                continue
            if pos_pre in vi:
                if vi[pos_pre] <= vi[pos_cur]:
                    continue
                if not shi[pos_pre] is None:
                    continue
            vi.pop(pos_cur)
            pti.pop(pos_cur)
            shi.pop(pos_cur)


    def get_point(self, idnum, pos):
        if not pos in self.points[idnum]:
            return None
        return self.points[idnum][pos]

    
    def get_point_int(self, idnum, pos):
        pt=self.get_point(idnum, pos)
        if pt is None:
            return None
        return (int(pt[0]),int(pt[1]))

    def x_particles(self,pos):
        for (i,(pt,sh)) in enumerate(zip(self.points,self.shape)):
            if pos in pt:
                if sh[pos] is None:
                    yield (i,(int(pt[pos][0]),int(pt[pos][1])),None)
                else:
                    yield (i,(int(pt[pos][0]),int(pt[pos][1])),int(sh[pos]))

    def x_all_alive_particles(self):
        for (i,(pt,sh)) in enumerate(zip(self.points,self.shape)):
            for pos in pt.keys():
                if not sh[pos] is None:
                    yield (i,pos,(int(pt[pos][0]),int(pt[pos][1])),int(sh[pos]))

    def get_close_particle(self, pos, x, y, r,exceptid=[]):
        for (i,pt) in enumerate(self.points):
            if not pos in pt:
                continue
            if (pt[pos][0]-x)**2+(pt[pos][1]-y)**2<r**2:
                if not i in exceptid:
                    return i
        return None

    def x_particles_to_update(self, pos_pre, pos_cur):
        for (i,(pti,shi,vi,pos_0_i)) in enumerate(zip(self.points,self.shape,self.version,self.initial_pos)):
            if (pos_cur-pos_0_i)*(pos_cur-pos_pre)<=0:
                continue
            if not pos_pre in vi:
                continue
            if pos_cur in vi:
                if vi[pos_cur] >= vi[pos_pre]:
                    continue
            if shi[pos_pre] is None:
                continue
            if pti[pos_pre] is None:
                continue
            yield (i,pti[pos_pre])

if __name__=="__main__":
    import sys, os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    tf=TrackingFilterUI()
    tf.run()
    cv2.destroyAllWindows()


