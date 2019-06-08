import cv2
import numpy as np
import pyautogui
import time
import math
# cap = cv2.VideoCapture(0)
 
# kernel = np.ones((5,5), np.uint8)

def setCursorPos( current_p, prev_p):
	
	mouse_p = np.zeros(2)
	
	if abs(current_p[0]-prev_p[0])<5 and abs(current_p[1]-prev_p[1])<5:
		mouse_p[0] = current_p[0] + .7*(prev_p[0]-current_p[0]) 
		mouse_p[1] = current_p[1] + .7*(prev[1]-current_p[1])
	else:
		mouse_p[0] = current_p[0] + .1*(prev_p[0]-current_p[0])
		mouse_p[1] = current_p[1] + .1*(prev_p[1]-current_p[1])
	
	return mouse_p

def contours(hand_hist_mask):
    hand_hist_mask_gray = cv2.cvtColor(hand_hist_mask,cv2.COLOR_BGR2GRAY)
    ret, thresh= cv2.threshold(hand_hist_mask_gray, 150, 255, 0)
    contour, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    return contour

def max_contour(contour_list):
    max_ind=0
    max_area=0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area = cv2.contourArea(cnt)
        if area>max_area:
            max_area = area
            max_ind = i

    return contour_list[max_ind]

def centroid(max_contour_list):
    M = cv2.moments(max_contour_list)

    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return cx ,cy

    else:
        return None

def euclids(x1,y1, x2,y2):
    p = (x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)
    p =np.sqrt(p)

    return p

def draw_rect(frame):
    # ret, frame = cap.read()
    # frame= cv2.flip(frame,1)
    rows, cols, _ = frame.shape
    # print(rows,cols)

    #### to draw 9 rectangles for refrence of keeping hand to detect the skin

    global hand_one_rect_x, hand_one_rect_y, hand_two_rect_y, hand_two_rect_x

    hand_one_rect_x = np.array([6*rows/20, 6*rows/20, 6*rows/20, 8*rows/20, 8*rows/20, 8*rows/20, 10*rows/20, 10*rows/20, 10*rows/20], dtype= np.uint32)

    hand_one_rect_y =np.array([9*cols/20, 10*cols/20, 11*cols/20, 9*cols/20, 10*cols/20, 11*cols/20, 9*cols/20, 10*cols/20, 11*cols/20], dtype= np.uint32)



    hand_two_rect_x= hand_one_rect_x + 10
    hand_two_rect_y= hand_one_rect_y + 10

    for i in range(9):
        cv2.rectangle(frame, (hand_one_rect_y[i], hand_one_rect_x[i]),(hand_two_rect_y[i],hand_two_rect_x[i]), (0,0,255), 1)


    return frame
    ##### TAKING SAMPLE OF HAND COLOR 
def hand_histogram(frame):
    global hand_one_rect_x, hand_one_rect_y

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    roi = np.zeros([90, 10, 3], dtype=hsv_frame.dtype)

    for i in range(9):
        roi[i*10: i*10 + 10, 0:10] = hsv_frame[hand_one_rect_x[i]:hand_one_rect_x[i] + 10, hand_one_rect_y[i]:hand_one_rect_y[i] + 10]  



    hand_hist = cv2.calcHist([roi], [0], None, [256], [0, 256])
    return (cv2.normalize(hand_hist, hand_hist, 0, 255, cv2.NORM_MINMAX))
    
def hist_masking(frame,hand_hist):
    hsv_frame=cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    dst = cv2.calcBackProject([hsv_frame], [0, 1], hand_hist, [0, 180, 0, 256], 1)

    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    
    cv2.filter2D(dst, -1, disc, dst)
    ret, thresh = cv2.threshold(dst, 150, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thresh = cv2.medianBlur(thresh,5)
    thresh = cv2.erode(thresh, None, iterations=5)
    thresh = cv2.dilate(thresh, None, iterations=5)
    thresh = cv2.merge((thresh, thresh, thresh))
    
    return(thresh)

def farthest_point(defects, contour, centroid):
    if defects is not None and centroid is not None:
        s = defects[:, 0][:, 0]
        cx, cy = centroid

        x = np.array(contour[s][:, 0][:, 0], dtype=np.float)
        y = np.array(contour[s][:, 0][:, 1], dtype=np.float)

        xp = cv2.pow(cv2.subtract(x, cx), 2)
        yp = cv2.pow(cv2.subtract(y, cy), 2)
        dist = cv2.sqrt(cv2.add(xp, yp))

        dist_max_i = np.argmax(dist)

        if dist_max_i < len(s):
            farthest_defect = s[dist_max_i]
            farthest_point = tuple(contour[farthest_defect][0])
            return farthest_point
        else:
            return None


def img_operation(frame,hand_hist):
    global lock1,lock2,lock3
    global value1, value2, value3,prev
    frame = hist_masking(frame,hand_hist)
    contour_list = contours(frame)
    max_contour_list = max_contour(contour_list)
    centroid_val = centroid(max_contour_list)
    cv2.circle(frame, centroid_val, 5, (0,255,0), -1)
    frame = cv2.drawContours(frame, max_contour_list, -1, (0,255,0), 3)

    cnt = max_contour_list
    hull = cv2.convexHull(cnt,returnPoints = False)

    defects = cv2.convexityDefects(cnt,hull)
    x = defects.shape
    count =1
    for i in range(x[0]):
        s,e,f,d = defects[i,0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
        b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
        c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
        angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) 

        if angle <= math.pi / 2:
        # cv2.line(frame,start,end,[0,255,0],2)
            count = count+1
            cv2.circle(frame,far,5,[0,0,255],-1)

    # print(count)


    far_points = farthest_point(defects, max_contour_list, centroid_val)
    cv2.circle(frame, far_points, 10, [255,0,0], -1)

    # p = euclids(far_points[0], far_points[1], centroid_val[0], centroid_val[1])


    print(count)
    pyautogui.FAILSAFE = False
    if(count == 1 or count == 2):
        # pyautogui.moveTo((mp[0]-150)*3.45, (mp[1]-50)*3.85)
    	lock1 = 1; lock2 = 0; lock3 = 0; value2 = 1; value3=1;

    if(count == 3):
    	lock2 = 1; lock1 = 0; lock3 = 0; value1 = 1; value3 = 1;

    if(count == 5):
    	lock3=1; lock1=0; lock2=0; value2=1; value1=1;

    mp = setCursorPos(far_points, prev)
    if(lock1 == 1):
    	value1 = 0
    	pyautogui.moveTo((mp[0]-150)*3.45, (mp[1]-50)*3.85)

    if(lock2 == 1 and value2 == 1):
    	value2 = 0
    	pyautogui.click()

    if(lock3 == 1 and value3== 1):
    	value3 = 0
    	pyautogui.scroll(-10)
    prev = far_points

    return frame


def main():
    is_hand_hist_created=False
    global lock1,lock2,lock3
    lock1 = lock3 = 0 
    lock2 = 0
    global value1, value2,value3
    value1 = value2 = value3 = 1
    global hand_hist,sub,prev
    prev=[0,0]
    # cap1= cv2.VideoCapture(0)
    cap= cv2.VideoCapture(1)
    sub = cv2.createBackgroundSubtractorMOG2()

    while(cap.isOpened()):
        pressed_key = cv2.waitKey(30) & 0xFF
        ret, frame=cap.read()
        # frame = cv2.resize(frame, (1366,768))
        
        frame= cv2.flip(frame,1)


        if pressed_key & 0xFF == ord("z"):
            # t = time.time()
            is_hand_hist_created=True
            hand_hist=hand_histogram(frame)
            
            # cv2.putText(frame,"DONE",(220,340),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255))
            # while(time.time()<t+5):
            # cv2.putText(frame,"DONE",(220,340),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255))
              
            

        if is_hand_hist_created:
            frame = img_operation(frame, hand_hist)
        else:
            frame = draw_rect(frame)


        
        cv2.imshow("frame",frame)

        if pressed_key==27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()