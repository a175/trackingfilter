import numpy
import cv2

class TrackingFilter:
    def __init__(self,original):
        self.original = original
        cv2.namedWindow("frame", cv2.CV_WINDOW_AUTOSIZE)
        self.mouse_event_note={}
        self.mouse_event_result=[]
        self.key_event_result=[]
        self.direction=0
        (ret, self.frame) = self.original.read()
        self.bg=[]
        self.template=[]
        self.features=None
        self.CRITERIA = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        self.flag_mask=False
        self.flag_templatematch=False
        self.flag_gray=False

        
        
    def select_background(self):
        print "Choose backgroud frames."
        print ""
        self.key_event_result=[]
        self.flag_mask=False
        self.flag_templatematch=False
        self.flag_gray=False
        flag=self.showvideo(0)
        if flag <= 0:
            return flag
        pos=self.key_event_result[-1]
        print ""
        print "``` python"
        print "tf.append_bg(",pos,")"
        print "```"
        print ""
        self.append_bg(pos)
        return flag
    
    def append_bg(self,pos):
        self.flag_mask=False
        self.flag_templatematch=False
        self.flag_gray=False
        self.update_frame(pos,False)
        self.bg.append(numpy.copy(self.frame))

    
    def select_template(self):
        print "Choose target unit."
        print ""
        print "+","mouse drag: append the rectangle;"
        self.event_x=None
        self.inputted_data=[]
        cv2.setMouseCallback("frame", self.select_rectangle_by_mouse)
        self.flag_mask=True
        self.flag_templatematch=False
        self.flag_gray=True
        flag=self.showvideo(0)
        if flag <= 0:
            return flag

        print ""
        print "``` python"
        self.reset_template()
        print "tf.reset_template()"
        for (pos,xp,x,yp,y) in self.inputted_data:
            if yp>y:
                t=yp
                yp=y
                y=t
            if xp>x:
                t=xp
                xp=x
                x=t
            self.append_template(pos,xp,x,yp,y)
            print "tf.reset_template(",pos,",",xp,",",x,",",yp,",",y,")"
        print "```"
        print ""
        return flag
    def reset_template(self):
        self.template=[]
    def append_template(self,pos,xp,x,yp,y):
        self.original.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, pos)
        (ret, frame) = self.original.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        self.template.append(gray[yp:y,xp:x])

    def select_feature(self):
        print "Choose target unit."
        print ""
        print "+","mouse click: select the particle;"
        self.event_x=None
        self.inputted_data=[]
        cv2.setMouseCallback("frame", self.select_point_by_mouse)
        self.flag_mask=True
        self.flag_templatematch=True
        self.flag_gray=True
        flag=self.showvideo(0)
        if flag <= 0:
            return flag
        print ""
        print "``` python"
        self.reset_original_features()
        print "tf.reset_original_features()"
        for (pos,x,y) in self.inputted_data:
            self.append_original_features(pos,x,y)
            print "tf.append_original_features(",pos,",",x,",",y,")"
        self.adjust_original_features()
        print "tf.adjust_original_features()"
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
        
    def run(self):
        currentstep=1
        while currentstep > 0:
            print "Step", currentstep
            print "==========="
            if currentstep==1:
                flag=self.select_background()
            elif currentstep==2:
                flag=self.select_template()
            elif currentstep==3:
                flag=self.select_feature()
            else:
                print "end."
                break
            if flag>0:
                currentstep=currentstep+1
            elif flag<0:
                currentstep=currentstep-1
                if currentstep==0:
                    currentstep=1
            else:
                break
            print "\n***\n"

                                                                        
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
            cv2.rectangle(self.frame, (xp-1, yp-1), (xp+1,yp+1), (255,255, 255), 2)
            
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_event_note["LBUTTONDOWN"]=(x,y)
            self.mouse_event_note["direction"]=self.direction
            self.mouse_event_note["frame"]=numpy.copy(self.frame)
            cv2.rectangle(self.frame, (x-1, y-1), (x+1,y+1), (255,0, 255), 1)
            self.direction=0
            print "point: ", (x,y)
            pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
            self.inputted_data.append((pos,x,y))

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
            #return cv2.bitwise_and(frame,mask)                
            return cv2.bitwise_or(frame,cv2.bitwise_not(mask))
    def apply_template_match(self,frame):
        template = self.template[-1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        res = cv2.matchTemplate(gray,template,cv2.TM_CCOEFF_NORMED)
        s=res.shape
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ss=hsv.shape
        d0=(ss[0]-s[0])//2
        d1=(ss[1]-s[1])//2
#                hsv[d0:s[0]+d0,d1:s[1]+d1,0]=90*res
#                hsv[d0:s[0]+d0,d1:s[1]+d1,1]=255*res
        hsv[d0:s[0]+d0,d1:s[1]+d1,2]=255*res
        
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

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
        (ret, frame) = self.original.read()
        if ret:
            if self.flag_templatematch:
                mask=self.get_mask(frame)
                frame=self.apply_mask(frame,mask)
                frame=self.apply_template_match(frame)
                frame=self.apply_mask(frame,mask)
            if self.flag_mask:
                mask=self.get_mask(frame)
                frame=self.apply_mask(frame,mask)
            if self.flag_gray:
                frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.frame = frame
        else:
            self.direction=0
            pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            if verbose:
                print "frameposition:", pos
                print "*This is the final frame.*"
            
        
    def showvideo(self,initialpos):
        print "+","q|c: quit;"
        print "+","f: forward;"
        print "+","b: reverse;"
        print "+","n: pause at next frame;"
        print "+","p: pause at previous frame;"
        print "+","e: pause at final frame;"
        print "+","a: pause at first frame;"
        print "+","w: store this frame position;"
        print "+","[: previous step"
        print "+","]: next step"
        print ""
        print "frameposition:", initialpos
        self.update_frame(initialpos)
        while(self.original.isOpened()):
            if self.direction>0:
                self.update_frame()
            elif self.direction<0:
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                pos=pos-1
                self.update_frame(pos)

            cv2.imshow('frame',self.frame)
            keyinput=cv2.waitKey(1) & 0xFF
            if keyinput == ord('q'):
                print "Quit"
                return 0
            elif keyinput == ord('c'):
                print "Close"
                return 0
            elif keyinput == ord('p'):
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                pos=pos-1
                self.update_frame(pos)
                self.direction=0
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                print "frameposition:", pos
            elif keyinput == ord('n'):
                self.update_frame()
                self.direction=0
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                print "frameposition:", pos
            elif keyinput == ord('e'):
                pos=self.original.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)-1
                self.update_frame(pos)
                self.direction=0
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                print "frameposition:", pos
            elif keyinput == ord('a'):
                pos=0
                self.update_frame(pos)
                self.direction=0
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                print "frameposition:", pos
            elif keyinput == ord('f'):
                self.direction=1
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                print "frameposition:", pos, "->"
            elif keyinput == ord('b'):
                self.direction=-1
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                print "frameposition:", pos, "<-"
            elif keyinput == ord(']'):
                self.direction=0
                return 1
            elif keyinput == ord('['):
                self.direction=0
                return -1

            elif keyinput == ord('w'):
                pos=self.original.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)-1
                print "w:", pos
                self.key_event_result.append(pos)



if __name__=="__main__":
    original=cv2.VideoCapture('a.avi')
    tf=TrackingFilter(original)
    tf.run()
    original.release()
    cv2.destroyAllWindows()
