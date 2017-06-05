import numpy
import cv2

class TrackingFilter:
    def __init__(self):
        self.mouse_event_note={}
        self.mouse_event_result=[]
        self.direction=0
        self.bg=[]
        self.template=[]
        self.set_masktype("1")
        self.set_templatematchingtype("apply")
        self.features=None
        self.CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        self.flag_mask=False
        self.flag_templatematch=False
        self.flag_gray=False
        self.flag_trackedpoint=False
        self.flag_original=False
        self.flag_filter=False
        self.flag_tracking=False
        self.default_direction=0

    def get_gray_image_for_template_match(self,frame):
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
        g=self.get_gray_image_for_template_match(frame)
        mask=self.get_mask_for_opticalflow(frame)
        g=self.apply_template_match(g)
        g=self.apply_mask(g,mask)
        return g

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
        print "tf=TrackingFilter()"
        print "original=cv2.VideoCapture(\"%s\")" % video
        print "tf.set_original(original)"
        print "tf.set_outfile(\"%s\")" % outfile
        print "```"
        print ""
        return 1
    
    def set_original(self,video):
        self.original=video
        
    def set_outfile(self,outfilename):
        self.outfile = outfilename
        
    def select_background(self):
        print "Choose backgroud frames."
        print "The last frame of w_ring is used as background."
        print ""
        self.flag_gray=True
        self.flag_mask=False
        self.flag_templatematch=False
        self.flag_trackedpoint=False
        self.flag_original=False
        self.flag_filter=False
        self.flag_tracking=False
        self.default_direction=0

        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)        
        flag=self.showvideo(0)
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

    def append_bg(self,pos):
        self.original.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
        (ret, frame) = self.original.read()
        self.bg.append(frame)

    def select_grayimage(self):
        print "Choose single channel image."
        print ""
        self.set_gray_type("GRAY")
        self.flag_mask=False
        self.flag_templatematch=False
        self.flag_gray=True
        self.flag_trackedpoint=False
        self.flag_original=False
        self.flag_filter=False
        self.flag_tracking=False
        self.default_direction=0

        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)        
        flag=self.showvideo(0,
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

    def set_gray_type(self,t):
        self.gray_type=t
        
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
        pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
        self.update_frame(pos)

                
    def select_masktype(self):
        print "Choose mask type for background."
        print ""
        self.flag_mask=True
        self.flag_templatematch=False
        self.flag_gray=True
        self.flag_trackedpoint=False
        self.flag_original=False
        self.flag_filter=False
        self.flag_tracking=False
        self.default_direction=0

        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)        
        flag=self.showvideo(0,
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
        pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
        self.update_frame(pos)
        
    def set_masktype(self,t):
        self.masktype=t
            
    def select_templatematchingtype(self):
        print "Choose apply template matching or not."
        print ""
        self.flag_mask=True
        self.flag_templatematch=True
        self.flag_gray=True
        self.flag_trackedpoint=False
        self.flag_original=False
        self.flag_filter=False
        self.flag_tracking=False
        self.default_direction=0

        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)        
        flag=self.showvideo(0,
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
        pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
        self.update_frame(pos)
        
    def set_templatematchingtype(self,t):
        self.templatematchingtype=t
        
    def select_template(self):
        print "Choose template image for template matching."
        print ""
        print "+","mouse drag: append the rectangle;"
        self.event_x=None
        self.inputted_data=[]
        self.flag_mask=False
        self.flag_templatematch=False
        self.flag_gray=True
        self.flag_trackedpoint=False
        self.flag_original=False
        self.flag_filter=False
        self.flag_tracking=False
        self.default_direction=0

        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
        cv2.setMouseCallback("frame", self.select_rectangle_by_mouse)
        flag=self.showvideo(0)
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
    
    def reset_template(self):
        self.template=[]
        
    def append_template(self,pos,xp,x,yp,y):
        self.original.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
        (ret, frame) = self.original.read()
        gray = self.get_gray_image_for_template_match(frame)
        self.template.append(gray[yp:y,xp:x])

    def select_feature(self):
        print "Choose target unit."
        print ""
        print "+","mouse click: select the particle;"
        self.event_x=None
        self.inputted_data=[]
        self.flag_mask=True
        self.flag_templatematch=True
        self.flag_gray=True
        self.flag_trackedpoint=False
        self.flag_original=False
        self.flag_filter=False
        self.flag_tracking=False
        self.default_direction=0

        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)        
        cv2.setMouseCallback("frame", self.select_point_by_mouse)        
        flag=self.showvideo(0)
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
#        self.adjust_original_features()
#        print "tf.adjust_original_features()"
        print "```"
        print ""
        return flag
    
    def reset_original_features(self):
        self.original_features={}
        
    def append_original_features(self,pos,x,y):
        if pos in self.original_features:
            self.original_features[pos] = numpy.append(self.original_features[pos], [[[x, y]]], axis = 0).astype(numpy.float32)
        else:
            self.original_features[pos] = numpy.array([[[x, y]]], numpy.float32)
    def adjust_original_features(self):
        for pos in self.original_features:
            self.original.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
            (ret, frame) = self.original.read()
            mask=self.get_mask(frame)
            frame=self.apply_mask(frame,mask)
            frame=self.apply_template_match(frame)
            frame=self.apply_mask(frame,mask)
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.cornerSubPix(frame, self.original_features[pos], (10, 10), (-1, -1), self.CRITERIA)
            
    def track_optical_flow_all_and_check(self):
        print "Track points."
        print ""
        
        self.flag_mask=False
        self.flag_templatematch=False
        self.flag_gray=False
        self.flag_trackedpoint=True
        self.flag_original=False
        self.flag_filter=False
        self.flag_tracking=True
        self.tracked_point={}
        trackedpoints={}
        for initpos in self.original_features:
            print "Track points from the frame",initpos ,"to the final frame."
            print "If [f] or [n] is used, then optical flow will be computed."
            print ""
            print "+","mouse click: remove it if there is a particle near; otherwise append new one;"

            self.default_direction=1
            self.tracked_point={}
            self.tracked_point[initpos]=self.original_features[initpos]
            cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
            cv2.setMouseCallback("frame", self.edit_tracked_points_by_mouse)
            flag=self.showvideo(initpos)
            cv2.destroyWindow("frame")
            if flag <= 0:
                return flag
 
            print "Track points from the frame",initpos ,"to the first frame."
            print "If [b] or [p] is used, then optical flow will be computed."
            print ""
            print "+","mouse click: remove it if there is a particle near; otherwise append new one;"
            self.default_direction=-1
            cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
            cv2.setMouseCallback("frame", self.edit_tracked_points_by_mouse)
            flag=self.showvideo(initpos)
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
                    ff.append((feature[0][0],feature[0][1]))
                for (x,y) in ff:
                    self.append_filter_position(pos,x,y)
                print "ff=",ff
                print "for (x,y) in ff:"
                print "    tf.append_filter_position(",pos,",x,y)"
                print ""
        print "```"
        print ""
        return flag
    
    def reset_filter_position(self):
        self.filter_position={}
        
    def append_filter_position(self,pos,x,y):
        if not pos in self.filter_position:
            self.filter_position[pos]=[]
        self.filter_position[pos].append((x,y))
    
    def track_optical_flow_between_frames(self,frame_p,frame,features_p):
        gray = self.get_gray_image_for_optical_flow(frame)
        gray_p = self.get_gray_image_for_optical_flow(frame_p)
        #calcOpticalFlowFarneback
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
        i = 0
        if not features is None:
            while i < len(features):
                if status[i] == 0:
                    features = numpy.delete(features, i, 0)
                    status = numpy.delete(status, i, 0)
                    i -= 1
                i += 1
        return features

    def check_filter(self):
        self.filter=numpy.array([[-0.3*min(100,max(0,2*(100-((i-10)*(i-10)+(j-10)*(j-10))))) for j in range(20)] for i in range(20)],numpy.float32)
        self.flag_mask=False
        self.flag_templatematch=False
        self.flag_gray=False
        self.flag_trackedpoint=False
        self.flag_original=False
        self.flag_filter=True
        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
        flag=self.showvideo(0)
        cv2.destroyWindow("frame")
        if flag <= 0:
            return flag
        print ""
        print ""
        print "``` python"
        print "```"
        print ""
        return flag
        
    def apply_filter(self,frame,y,x):
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if frame[x-10:x+10,y-10:y+10,0].shape == self.filter.shape:
            frame[x-10:x+10,y-10:y+10,0]=(frame[x-10:x+10,y-10:y+10,0]+self.filter)%180
        frame=cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
        return frame
    def run(self,initstep=0):
        runlevel=[
            self.input_filename,
            self.select_grayimage,
            self.select_background,
            self.select_masktype,
            self.select_template,
            self.select_templatematchingtype,
            self.select_feature,
            self.track_optical_flow_all_and_check,
            self.check_filter
        ]
        currentstep=initstep
        while 0 <= currentstep and currentstep < len(runlevel):
            print "Section", currentstep
            print "==========="
            flag=runlevel[currentstep]()
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
                                                                        
    def select_rectangle_by_mouse(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
            self.direction=self.mouse_event_note["direction"]
            self.frame=numpy.copy(self.mouse_event_note["frame"])
            self.mouse_event_note={}
            pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
            self.inputted_data.append((pos,xp,x,yp,y))
            cv2.rectangle(self.frame, (xp, yp), (x,y), (0, 255, 255), 1)
            print "rectangle: ", ((xp, yp), (x,y))            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=self.frame
            self.direction=0
        elif  event == cv2.EVENT_MOUSEMOVE:
            if "LBUTTONDOWN" in self.mouse_event_note:
                (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
                self.frame=numpy.copy(self.mouse_event_note["frame"])
                cv2.rectangle(self.frame, (xp, yp), (x,y), (0, 0, 255), 1)

    def select_point_by_mouse(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
#            self.direction=self.mouse_event_note["direction"]
            self.frame=numpy.copy(self.mouse_event_note["frame"])
            (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
            self.mouse_event_note={}
            cv2.rectangle(self.frame, (xp-1, yp-1), (xp+1,yp+1),(15, 100, 255), 2)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=numpy.copy(self.frame)
            self.frame=numpy.copy(self.frame)
            cv2.rectangle(self.frame, (x-1, y-1), (x+1,y+1), (255,0, 255), 1)
            self.direction=0
            print "point: ", (x,y)
            pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
            self.inputted_data.append((pos,x,y))
            
    def add_or_remove_from_tracked_point(self, x, y, r):
        pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
        self.tracking_data_current_version=self.tracking_data_current_version+1
        self.tracking_data_version[pos]=self.tracking_data_current_version
        if pos in self.tracked_point:
            i=0
            for feature in self.tracked_point[pos]:
                if (x-feature[0][0])*(x-feature[0][0])+(y-feature[0][1])*(y-feature[0][1])<r*r:
                    print "remove", i, feature
                    self.tracked_point[pos] = numpy.delete(self.tracked_point[pos], i, 0)
                    return ((feature[0][0],feature[0][1]),-1)
                else:
                    i=i+1
            print "add", x,y
            self.tracked_point[pos] = numpy.append(self.tracked_point[pos], [[[x, y]]], axis = 0).astype(numpy.float32)
            return ((x,y),1)

        else:
            print "add", x,y, "."
            self.tracked_point[pos]=numpy.array([[[x, y]]], numpy.float32)
            return ((x,y),1)
        
    def edit_tracked_points_by_mouse(self,event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
#            self.direction=self.mouse_event_note["direction"]
            self.frame=numpy.copy(self.mouse_event_note["frame"])
            (xp,yp)=self.mouse_event_note["LBUTTONDOWN"]
            self.mouse_event_note={}
            (pt,f)=self.add_or_remove_from_tracked_point(xp,yp,10)
            if f>0:
                cv2.circle(self.frame, pt, 1, (15, 100, 255), -1, 8, 10)
                cv2.circle(self.frame, pt, 10, (15, 100, 255), 1)
            else:
                cv2.circle(self.frame, pt, 1, (200, 100, 0), -1, 8, 10)
                cv2.circle(self.frame, pt, 10, (200, 100, 0), 1)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=numpy.copy(self.frame)
            cv2.rectangle(self.frame, (x-1, y-1), (x+1,y+1), (255,0, 255), 1)
            self.direction=0
            print "point: ", (x,y)
            pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
            self.inputted_data.append((pos,x,y))

    def get_mask_for_opticalflow(self,frame):
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
        
    def get_mask(self,frame):
        if self.bg==[]:
            return None
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        bg = self.bg[-1]
        bg = cv2.cvtColor(bg, cv2.COLOR_BGR2HSV)
        temp =cv2.absdiff(img,bg)
        temp = cv2.threshold(temp, 15, 255, cv2.THRESH_BINARY)[1]
        operator = numpy.ones((3, 3), numpy.uint8)
        temp = cv2.dilate(temp, operator, iterations=4)
        temp = cv2.erode(temp, operator, iterations=4)
        temp = cv2.dilate(temp, operator, iterations=4)
        temp = cv2.erode(temp, operator, iterations=4)
        temp[:,:,1]=temp[:,:,0]
        temp[:,:,2]=temp[:,:,0]
        mask = temp
        return mask

    def apply_mask(self,frame,mask):
        if not mask is None:
            if self.masktype=="1":
                return cv2.bitwise_or(frame,cv2.bitwise_not(mask))
            elif self.masktype=="0":
                return cv2.bitwise_and(frame,mask)
        return numpy.copy(frame)

    def apply_template_match(self, gray):
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
        if self.direction*self.default_direction <= 0:
            return
        pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
        if not pos-self.default_direction in self.tracking_data_version:
            return
        if pos in self.tracking_data_version:
            if self.tracking_data_version[pos] >= self.tracking_data_version[pos-self.default_direction]:
                return
        if not pos-self.default_direction in self.tracked_point:
            return
        features=self.track_optical_flow_between_frames(self.original_frame,oframe,self.tracked_point[pos-self.default_direction])
        if features is None:
            return
        self.tracked_point[pos]=features
        self.tracking_data_version[pos]=self.tracking_data_current_version

    def update_frame(self,pos=None,verbose=True):
        if not pos is None:
            if pos>=0:
                self.original.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
            else:
                self.direction=0
                self.original.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
                if verbose:
                    print "frameposition:", pos
                    print "*This is the first frame.*"
        (ret, oframe) = self.original.read()
        if ret:
            if self.flag_tracking:
                self.update_trackingpoint_if_needed(oframe)
            self.original_frame=numpy.copy(oframe)
            frame=oframe
            if self.flag_gray:
                g=self.get_gray_image_for_template_match(frame)
                if self.flag_templatematch:
                    g=self.apply_template_match(g)
                if self.flag_mask:
                    mask=self.get_mask_for_opticalflow(frame)
                    g=self.apply_mask(g,mask)
                    
                frame[:,:,0]=g
                frame[:,:,1]=g
                frame[:,:,2]=g
            if self.flag_trackedpoint:
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                if pos in self.tracked_point:
                    for (i,feature) in enumerate(self.tracked_point[pos]):
                        cv2.circle(frame, (feature[0][0], feature[0][1]), 1, (15, 100, 255), -1, 8, 10)
                        cv2.circle(frame, (feature[0][0], feature[0][1]), 10, (15, 100, 255), 1)
                        for feature2 in self.tracked_point[pos][:i]:
                            dsq=(feature[0][0]-feature2[0][0])**2+(feature[0][1]-feature2[0][1])**2
                            if dsq < 4*100:
                                cv2.circle(frame, (feature[0][0], feature[0][1]), 10, (15, 250, 100), 1)
                                cv2.circle(frame, (feature2[0][0], feature2[0][1]), 10, (15, 250, 100), 1)
                            if dsq < 100:
                                cv2.circle(frame, (feature[0][0], feature[0][1]), 1, (100, 250, 15), 2)
                                cv2.circle(frame, (feature2[0][0], feature2[0][1]), 1, (100, 250, 15), 2)
            if self.flag_filter:
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                if pos in self.filter_position:
                    for (x,y) in self.filter_position[pos]:
                        frame=self.apply_filter(frame,x,y)

            self.frame = frame
        else:
            self.direction=0
            pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            if verbose:
                print "frameposition:", pos
                print "*This is the final frame.*"
            
    def showvideo(self,initialpos,xkeyevent=None, xkeyeventtitle="command mode"):
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
        print "+","o: toggle original/modified;"
        print "+","[: previous step"
        print "+","]: next step"
        if not xkeyevent is None:
            print "+","x:", xkeyeventtitle
        print ""
        self.key_w_ring=[initialpos]
        print "frameposition:", initialpos
        self.tracking_data_current_version=0
        self.tracking_data_version={}
        self.tracking_data_version[initialpos]=self.tracking_data_current_version
        self.update_frame(initialpos)
        while(self.original.isOpened()):
            if self.direction>0:
                self.update_frame()
                if self.direction<50:
                    self.direction=self.direction-1
                    pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                    print "frameposition:", pos
            elif self.direction<0:
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                pos=pos-1
                self.update_frame(pos)
                if self.direction>-50:
                    self.direction=self.direction+1
                    pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                    print "frameposition:", pos
            if self.flag_original:
                cv2.imshow('frame',self.original_frame)
            else:
                cv2.imshow('frame',self.frame)
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
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                print "frameposition:", pos
            elif keyinput == ord('<'):
                pos=0
                self.update_frame(pos)
                self.direction=0
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                print "frameposition:", pos
            elif keyinput == ord('u'):
                if self.key_w_ring != []:
                    pos=self.key_w_ring.pop()
                    self.key_w_ring.insert(0,pos)
                    self.update_frame(pos)
                    self.direction=0
                    pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                    print "frameposition:", pos
                    print "w_ring:",self.key_w_ring
            elif keyinput == ord('f'):
                self.direction=100
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                print "frameposition:", pos, "->"
            elif keyinput == ord('b'):
                self.direction=-100
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                print "frameposition:", pos, "<-"
            elif keyinput == ord('o'):
                self.flag_original=not self.flag_original
                print "originalframe:", self.flag_original
            elif keyinput == ord('w'):
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                self.key_w_ring.append(pos)
                self.direction=0
                print "w:", pos
                print "w_ring:",self.key_w_ring
            elif keyinput == ord('x'):
                if not xkeyevent is None:
                    print "x:", xkeyeventtitle
                    print "+ x"
                    xkeyevent()
                    print "leave x:", xkeyeventtitle

if __name__=="__main__":
    import sys, os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
    tf=TrackingFilter()
    tf.run()
    original.release()
    cv2.destroyAllWindows()
