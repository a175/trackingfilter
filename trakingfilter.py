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
            cv2.circle(self.mouse_event_layer, (xp, yp), 1, (0, 255, 255,1), -1, 8, 10)
            cv2.circle(self.mouse_event_layer, (xp,yp), int(radius), (0, 255, 255,1), 1)
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
                cv2.circle(self.mouse_event_layer, (xp,yp), 1, (0, 0, 255,1), -1, 8, 10)
                cv2.circle(self.mouse_event_layer, (xp,yp), int(radius), (0, 0, 255,1), 1)

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
            cv2.rectangle(self.mouse_event_layer, (xp, yp), (x,y), (0, 255, 255,1), 1)
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
                cv2.rectangle(self.mouse_event_layer, (xp, yp), (x,y), (0, 0, 255,1), 1)


    def select_point_by_mouse(self,event, x, y, flags, param):
        """
        Select point and append to self.inputted_data.
        This is used as mouse event callback.
        """
        if event == cv2.EVENT_LBUTTONUP:
#            self.direction=self.mouse_event_note["direction"]
            self.mouse_event_layer=numpy.copy(self.mouse_event_note["frame"])
            (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
            self.mouse_event_note={}
            cv2.rectangle(self.mouse_event_layer, (xp-1, yp-1), (xp+1,yp+1),(15, 100, 255,1), 2)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=numpy.copy(self.mouse_event_layer)
            self.mouse_event_layer=numpy.copy(self.mouse_event_layer)
            cv2.rectangle(self.mouse_event_layer, (x-1, y-1), (x+1,y+1), (255,0, 255,1), 1)
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
        fl=numpy.zeros(frame[:,:,0].shape)
        ma=numpy.zeros(frame[:,:,0].shape)
        for (y,x,r) in filters:
            (f,m,x0,y0,x1,y1)=self.filter_list[r]
            if fl[x+x0:x+x1,y+y0:y+y1].shape == f.shape:
                fl[x+x0:x+x1,y+y0:y+y1]=(fl[x+x0:x+x1,y+y0:y+y1]+f)
                ma[x+x0:x+x1,y+y0:y+y1]=(ma[x+x0:x+x1,y+y0:y+y1]+m)
        ma=numpy.abs(ma-0.5)+0.5
        fl=fl/ma
        fl=cv2.GaussianBlur(fl,(5,5),0)
        frame[:,:,0]=(frame[:,:,0]+fl[:,:])%180
        frame=cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        return frame

    def filter_weight(self,x,y,r):
        d=numpy.sqrt(x**2+y**2)
        if d>r:
            w=0
        else:
            w=1
        return w

    def append_filter_to_dict(self,radius):
        """
        Create a filter with the radius.
        Append it and its infomation (mask, margin, size)
        to the dictionary with key `radius`.
        """
        
        r=int(radius)
        f=numpy.array([[ 60*self.filter_weight(i-r,j-r,radius)  for j in range(2*r)] for i in range(2*r)],numpy.float32)
        (ret,m)=cv2.threshold(f,1,255,cv2.THRESH_TRUNC)
        self.filter_list[r]=(f,m,-r,-r,r,r)


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
        
        self.default_direction=1

    def reset_tracking_data_version(self, initpos):
        """
        Reset the version of tracking data.
        """
        self.tracking_data_current_version=0
        self.tracking_data_version={}
        self.tracking_data_version[initpos]=self.tracking_data_current_version

    def reset_tracked_point(self,initpos):
        self.tracked_point={}
        self.tracked_point[initpos]=self.original_features[initpos]
        self.tracked_previous_point={}
        self.tracked_previous_point[initpos]=[None for i in self.tracked_point[initpos]]
        self.waited_previous_point={}
        self.waited_previous_point[initpos]=[]

    def set_default_direction(self,direction):
        self.default_direction=direction

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

    def reset_original_features(self):
        """
        Reset the initial positions to calculate optical flows.
        """
        self.original_features={}
        
    def append_original_features(self,pos,x,y):
        """
        Append the initial positions to calculate optical flows.
        pos: frame num
        x,y: coordinate
        """
        if pos in self.original_features:
            self.original_features[pos] = numpy.append(self.original_features[pos], [[x, y]], axis = 0).astype(numpy.float32)
        else:
            self.original_features[pos] = numpy.array([[x, y]], numpy.float32)
            
    
    def track_optical_flow_between_frames(self,frame_p,frame,features_p):
        """
        Calculate optical flows.
        """
        gray = self.get_gray_image_for_optical_flow(frame)
        gray_p = self.get_gray_image_for_optical_flow(frame_p)
        #flow = cv2.calcOpticalFlowFarneback(gray_p,gray,0.5,3,15, 3, 5, 1.2, 0)
        (features,status,err)=cv2.calcOpticalFlowPyrLK(
            gray_p,
            gray,
            features_p,
            None,
            #winSize = (5, 5),
            winSize = (10, 10),
            #winSize = (20, 20),
            #maxLevel = 3,
            maxLevel = 5,
            criteria = self.CRITERIA,
            flags = 0)
        i=0
        j=0
        previd=[]
        notfoundid=[]
        if not features is None:
            while i < len(features):
                if status[i] == 0:
                    features = numpy.delete(features, i, 0)
                    status = numpy.delete(status, i, 0)
                    notfoundid.append(j)
                    i -= 1
                else:
                    previd.append(j)
                i = i+1
                j = j+1
        return (features,previd,notfoundid)
    
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

    def add_or_remove_from_tracked_point(self, x, y, r):
        """
        If 
        the distance of between a point in self.tracked_point and (x,y)
        is less that r,
        then remove it;
        otherwise append it.
        """

        pos=self.get_current_frame_position()
        self.tracking_data_current_version=self.tracking_data_current_version+1
        self.tracking_data_version[pos]=self.tracking_data_current_version
        if pos in self.tracked_point:
            i=0
            for feature in self.tracked_point[pos]:
                if (x-feature[0])*(x-feature[0])+(y-feature[1])*(y-feature[1])<r*r:
                    self.tracked_point[pos] = numpy.delete(self.tracked_point[pos], i, 0)
                    previd=self.tracked_previous_point[pos].pop(i)
                    self.waited_previous_point[pos].append(previd)
                    return ((feature[0],feature[1]),previd,-1)
                else:
                    i=i+1
            self.tracked_point[pos] = numpy.append(self.tracked_point[pos], [[x, y]], axis = 0).astype(numpy.float32)
            if len(self.waited_previous_point[pos])>0:
                previd=self.waited_previous_point[pos].pop()
            else:
                previd=None
            self.tracked_previous_point[pos].append(previd)
            return ((x,y),previd,1)

        else:
            self.tracked_point[pos]=numpy.array([[x, y]], numpy.float32)
            self.tracked_previous_point[pos]=[None]
            self.waited_previous_point[pos]=[]
            return ((x,y),None,1)
        
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
    
    def update_trackingpoint_if_needed(self,oframe):
        """
        If version is old then calculate optical flow.
        """
        
        if self.direction*self.default_direction <= 0:
            return
        pos=self.get_current_frame_position()
        if not pos-self.default_direction in self.tracking_data_version:
            return
        if pos in self.tracking_data_version:
            if self.tracking_data_version[pos] >= self.tracking_data_version[pos-self.default_direction]:
                return
        if not pos-self.default_direction in self.tracked_point:
            return
        (features,previd,notfoundid)=self.track_optical_flow_between_frames(self.original_frame,oframe,self.tracked_point[pos-self.default_direction])
        if features is None:
            return
        self.tracked_point[pos]=features
        self.tracked_previous_point[pos]=previd
        self.waited_previous_point[pos]=notfoundid
        self.tracking_data_version[pos]=self.tracking_data_current_version



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
            self.select_feature,
            self.track_optical_flow_all_and_check,
            self.check_filter
        ]

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
        print ""
        print "tf=TrackingFilterUI()"
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
        self.default_direction=0
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

    def modify_frame_to_select_feature(self,oframe):
        modifiedframes=[]
        frame=numpy.copy(oframe)
        g=self.get_gray_image_for_optical_flow(frame)
        frame[:,:,0]=g
        frame[:,:,1]=g
        frame[:,:,2]=g
        modifiedframes.append(numpy.copy(frame))

        return modifiedframes
    
    def select_feature(self):
        print "Choose target unit."
        print ""
        print "+","mouse click: select the particle;"
        self.event_x=None
        self.inputted_data=[]
        self.frame_variation_to_show=-1
        self.set_modify_frame(self.modify_frame_to_select_feature)

        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)        
        cv2.setMouseCallback("frame", self.select_point_by_mouse)        
        flag=self.showvideo(0,"frame")
        cv2.destroyWindow("frame")
        if flag <= 0:
            return flag
        print ""
        print "``` python"
        self.reset_original_features()
        print "tf.reset_original_features()"
        for (pos,x,y) in self.inputted_data:
            self.append_original_features(pos,x,y)
            print "tf.append_original_features(",pos,",",x,",",y,")"
        print "```"
        print ""
        return flag

    def modify_frame_to_track_optical_flow_all_and_check(self,oframe):
        modifiedframes=[]
        self.update_trackingpoint_if_needed(oframe)
        self.original_frame=numpy.copy(oframe)
        pos=self.get_current_frame_position()
        
        frame=numpy.copy(oframe)
        g=self.get_gray_image_for_optical_flow(frame)
        frame[:,:,0]=g
        frame[:,:,1]=g
        frame[:,:,2]=g        
        frame=self.draw_tracked_pairs_n(frame,pos)
        frame=self.draw_tracked_points(frame,pos)
        modifiedframes.append(numpy.copy(frame))
        frame=self.draw_tracked_pairs_p(frame,pos)
        frame=self.draw_tracked_points(frame,pos)
        modifiedframes.append(numpy.copy(frame))

        frame=numpy.copy(oframe)
        frame=self.draw_tracked_pairs_n(frame,pos)
        frame=self.draw_tracked_points(frame,pos)
        modifiedframes.append(numpy.copy(frame))
        frame=self.draw_tracked_pairs_p(frame,pos)
        frame=self.draw_tracked_points(frame,pos)
        modifiedframes.append(numpy.copy(frame))
        return modifiedframes

    def track_optical_flow_all_and_check(self):
        print "Track points."
        print ""
        flag=-1
        self.frame_variation_to_show=-1
        self.set_modify_frame(self.modify_frame_to_track_optical_flow_all_and_check)

        trackedpoints={}
        for initpos in self.original_features:
            print "Track points from the frame",initpos ,"to the final frame."
            print "If [f] or [n] is used, then optical flow will be computed."
            print ""
            print "+","mouse click: remove it if there is a particle near; otherwise append new one;"

            self.reset_tracking_data_version(initpos)
            self.reset_tracked_point(initpos)
            self.set_default_direction(1)
            cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
            cv2.setMouseCallback("frame", self.edit_tracked_points_by_mouse)
            flag=self.showvideo(initpos,"frame")
            cv2.destroyWindow("frame")
            if flag <= 0:
                return flag
 
            print "Track points from the frame",initpos ,"to the first frame."
            print "If [b] or [p] is used, then optical flow will be computed."
            print ""
            print "+","mouse click: remove it if there is a particle near; otherwise append new one;"
            self.reset_tracking_data_version(initpos)
            self.set_default_direction(-1)
            cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
            cv2.setMouseCallback("frame", self.edit_tracked_points_by_mouse)
            flag=self.showvideo(initpos,"frame")
            cv2.destroyWindow("frame")
            if flag <= 0:
                return flag
            trackedpoints[initpos]=self.tracked_point

        self.reset_filter_position()
        print ""
        print "``` python"
        print "tf.reset_filter_position()"
        for initpos in trackedpoints:
            for pos in trackedpoints[initpos]:
                ff=[]
                for feature in trackedpoints[initpos][pos]:
                    ff.append((feature[0],feature[1]))
                for (x,y) in ff:
                    self.append_filter_position(pos,x,y,self.radius_of_filter)
                print "ff=",ff
                print "for (x,y) in ff:"
                print "    tf.append_filter_position(",pos,",x,y,",self.radius_of_filter,")"
                print ""
        print "```"
        print ""
        return flag

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
#            self.direction=self.mouse_event_note["direction"]
            self.mouse_event_layer=numpy.copy(self.mouse_event_note["frame"])
            (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
            self.mouse_event_note={}
            (pt,r,f)=self.add_or_remove_from_filterposition(pos,xp,yp,4)
            if f>0:
                print "add", x,y, "." 
                cv2.circle(self.mouse_event_layer, pt, 1, (15, 100, 255,1), -1, 8, 10)
                cv2.circle(self.mouse_event_layer, pt, self.radius_of_filter, (15, 100, 255,1), 1)
            else:
                print "remove", x,y, ".", pt
                cv2.circle(self.mouse_event_layer, pt, 1, (200, 100, 0,1), -1, 8, 10)
                cv2.circle(self.mouse_event_layer, pt,  self.radius_of_filter, (200, 100, 0,1), 1)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=numpy.copy(self.mouse_event_layer)
            cv2.rectangle(self.mouse_event_layer, (x-1, y-1), (x+1,y+1), (255,0, 255,1), 1)
            self.direction=0
            print "point: ", (x,y)
            self.inputted_data.append((pos,x,y))

    def edit_tracked_points_by_mouse(self,event, x, y, flags, param):
        """
        Select point to add or remove.
        This is used as mouse event callback.
        """
        
        if event == cv2.EVENT_LBUTTONUP:
#            self.direction=self.mouse_event_note["direction"]
            self.mouse_event_layer=numpy.copy(self.mouse_event_note["frame"])
            (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
            self.mouse_event_note={}
            (pt,previd,f)=self.add_or_remove_from_tracked_point(xp,yp,4)
            if f>0:
                print "add", x,y, "." 
                cv2.circle(self.mouse_event_layer, pt, 1, (15, 100, 255,1), -1, 8, 10)
                cv2.circle(self.mouse_event_layer, pt, self.radius_of_filter, (15, 100, 255,1), 1)
            else:
                print "remove", x,y, "."
                cv2.circle(self.mouse_event_layer, pt, 1, (200, 100, 0,1), -1, 8, 10)
                cv2.circle(self.mouse_event_layer, pt,  self.radius_of_filter, (200, 100, 0,1), 1)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=numpy.copy(self.mouse_event_layer)
            cv2.rectangle(self.mouse_event_layer, (x-1, y-1), (x+1,y+1), (255,0, 255,1), 1)
            self.direction=0
            print "point: ", (x,y)
            pos=self.get_current_frame_position()
            self.inputted_data.append((pos,x,y))

    def x_tracked_pair(self,pos,prevpos):
        """
        Yields pairs of tracked pair between pos and prevpos
        """

        if not pos in self.tracked_point:
            return
        if not prevpos in self.tracked_point:
            return
        if not pos in self.tracked_previous_point:
            return

        ppt=self.tracked_point[prevpos]
        plen=len(ppt)
        for (f1,pid) in zip(self.tracked_point[pos],self.tracked_previous_point[pos]):
            if pid is None:
                continue
            if pid < plen:
                yield ((f1[0],f1[1]),(ppt[pid][0],ppt[pid][1]))

    def x_pair_of_close_point(self,points,r):
        """
        yields pair of points closer than r
        """
        for (i,pti) in enumerate(points):
            for ptj in points[:i]:
                d=numpy.sqrt((pti[0]-ptj[0])**2+(pti[1]-ptj[1])**2)
                if d < r:
                    yield ((pti[0],pti[1]),(ptj[0],ptj[1]))
                    
    def draw_tracked_pairs_p(self,frame,pos):
        """
        Draw lines between pair of racked points in pos to frame.
        """
        if pos in self.tracked_point:
            for tt in range(5):
                pos1=pos+tt*self.default_direction
                pos0=pos+(tt+1)*self.default_direction
                for (a,b) in self.x_tracked_pair(pos0,pos1):
                    cv2.line(frame,a,b,(200-40*tt,200-40*tt,200),1)
        return frame

    def draw_tracked_pairs_n(self,frame,pos):
        """
        Draw lines between pair of racked points in pos to frame.
        """
        if pos in self.tracked_point:
            for tt in range(5):
                pos0=pos-tt*self.default_direction
                pos1=pos-(tt+1)*self.default_direction
                for (a,b) in self.x_tracked_pair(pos0,pos1):
                    cv2.line(frame,a,b,(100,255,200-40*tt),1)
        return frame

    def draw_tracked_points(self,frame,pos):
        """
        Draw circle at points in pos to frame.
        """
        if pos in self.tracked_point:
            for pt in self.tracked_point[pos]:
                cv2.circle(frame, (pt[0], pt[1]), 1, (15, 100, 255), -1, 8, 10)
                cv2.circle(frame, (pt[0], pt[1]),  self.radius_of_filter, (15, 100, 255), 1)
            for (a,b) in self.x_pair_of_close_point(self.tracked_point[pos],10):
                cv2.circle(frame,a, 1, (100, 250, 15), 2)
                cv2.circle(frame,b, 1, (100, 250, 15), 2)
            for (a,b) in self.x_pair_of_close_point(self.tracked_point[pos],2*self.radius_of_filter):
                cv2.circle(frame,a,self.radius_of_filter,(15, 250, 100),1)
                cv2.circle(frame,b,self.radius_of_filter,(15, 250, 100),1)
        return frame

    def draw_filter_position(self,frame,pos):
        """
        Draw circle at points in pos to frame.
        """
        if pos in self.filter_position:
            for (x,y,r) in self.filter_position[pos]:
                cv2.circle(frame, (x,y), 1, (15, 100, 255), -1, 8, 10)
                cv2.circle(frame, (x,y), r, (15, 100, 255), 1)
        return frame


if __name__=="__main__":
    import sys, os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    tf=TrackingFilterUI()
    tf.run()
    cv2.destroyAllWindows()
