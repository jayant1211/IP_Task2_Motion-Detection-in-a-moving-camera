import numpy as np
import cv2

cap = cv2.VideoCapture("Videos/stationary.mp4")

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )


# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
color = np.random.randint(0,255,(100,3))

mask = np.zeros_like(old_frame)

while(1):
    ret,frame = cap.read()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # calculating optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params )

    if p1 is None:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Selecting good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        if (a - c > 0):
            cv2.putText(frame, "Right", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        elif (a - c < 0):
            cv2.putText(frame, "Left", (150, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        #if (b-d>0):
        #    cv2.putText(frame, "Up", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        #elif (b-d<0):
        #   cv2.putText(frame, "Down", (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        temp = cv2.line(mask, (a, b), (c, d), (0,255,0), 2)
    frame = cv2.add(frame, temp)

    cv2.imshow('frame',frame)
    # updating the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    k = cv2.waitKey(30)
    if k == 32:
        break

cv2.destroyAllWindows()
cap.release()