#youtube video link: https://youtu.be/ULoM5KS8W9c 

import cv2
import numpy as np
import math


def draw_rect(frame):
    
    rows, cols, _ = frame.shape

    #### to draw 9 rectangles for refrence of keeping hand to detect the skin

    global hand_one_rect_x, hand_one_rect_y, hand_two_rect_y, hand_two_rect_x

    hand_one_rect_x = np.array([7*rows/20, 7*rows/20, 7*rows/20, 9*rows/20, 9*rows/20, 9*rows/20, 11*rows/20, 11*rows/20, 11*rows/20], dtype= np.uint32)

    hand_one_rect_y =np.array([10*cols/20, 11*cols/20, 12*cols/20, 10*cols/20, 11*cols/20, 12*cols/20, 10*cols/20, 11*cols/20, 12*cols/20], dtype= np.uint32)



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

def centroid(max_contour):
    M = cv2.moments(max_contour)

    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])

        return cx ,cy

    else:
        return None
    
def hist_masking(frame):
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


def image_operation(frame):

    frame_mask = hist_masking(frame)
    frame_gray = cv2.cvtColor(frame_mask,cv2.COLOR_BGR2GRAY)
    ret, thresh= cv2.threshold(frame_gray, 150, 255, 0)
    contour, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    cnt = max(contour, key=cv2.contourArea)
    cx , cy =centroid(cnt)

    cv2.circle(frame, (cx,cy), 5, (0,255,0), -1)
    frame = cv2.drawContours(frame, contour, -1, (0,255,0), 3)

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
            count = count+1
            cv2.circle(frame,far,5,[0,0,255],-1)

    if count == 1:
        cv2.putText(frame,'1',(10,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,255,0),5,cv2.LINE_AA)

    if count == 2:
        cv2.putText(frame,'2',(10,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,0,255),5,cv2.LINE_AA)

    if count == 3:
        cv2.putText(frame,'3',(10,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,255,255),5,cv2.LINE_AA)

    if count == 4:
        cv2.putText(frame,'4',(10,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(0,0,255),5,cv2.LINE_AA)

    if count == 5:
        cv2.putText(frame,'5',(10,100), cv2.FONT_HERSHEY_SIMPLEX, 4,(255,0,0),5,cv2.LINE_AA)


    return frame


def main():
    global hand_hist,sub
    is_hand_hist_created=False
    
    cap= cv2.VideoCapture(0)
    sub = cv2.createBackgroundSubtractorMOG2()

    while(cap.isOpened()):
        pressed_key = cv2.waitKey(1)
        ret, frame=cap.read()
        # frame = cv2.resize(frame, (1366,768))
        
        frame= cv2.flip(frame,1)


        if pressed_key & 0xFF == ord("s"):
            
            is_hand_hist_created=True
            hand_hist=hand_histogram(frame)
                          
        if is_hand_hist_created:
            frame = image_operation(frame)
        else:
            frame = draw_rect(frame)


        
        cv2.imshow("frame",frame)

        if pressed_key==27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
