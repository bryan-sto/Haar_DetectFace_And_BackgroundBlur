import cv2
import time
import numpy as np
from imutils.video import VideoStream
import argparse
import numpy as np


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", type=str,
	default="haarcascade_frontalface_default.xml",
	help="path to haar cascade face detector")
args = vars(ap.parse_args())

# load the haar cascade face detector from
print("[INFO] loading face detector...")
detector = cv2.CascadeClassifier(args["cascade"])

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
prev_frame_time = 0
new_frame_time = 0

 
while True:
    
    new_frame_time = time.time()
    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    fps = int(fps)
    fps = str(fps)
    font = cv2.FONT_HERSHEY_SIMPLEX = 1
    abc = "FPS :"
    hehe = "coordinate: "
    
    # Step 1: Capture the frame
    frame = vs.read()
    
    # Step 2: Convert to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Step 3: Create a mask based on medium to high Saturation and Value
    # - These values can be changed (the lower ones) to fit your environment
    mask = cv2.inRange(hsv, (0, 75, 40), (180, 255, 255))
    
    # We need a to copy the mask 3 times to fit the frames
    mask_3d = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    
    # Step 4: Create a blurred frame using Gaussian blur
    blurred_frame = cv2.GaussianBlur(frame, (25, 25), 0)
    
    # Step 5: Combine the original with the blurred frame based on mask
    frame = np.where(mask_3d == (255, 255, 255), frame, blurred_frame)
    
    # test
    rects = detector.detectMultiScale(hsv, scaleFactor=1.05,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)
	
    # loop over the bounding boxes
    print(rects)
    #shapes = np.zeros_like(blur, np.uint8)
        
    for (x, y, w, h) in rects:
        center = (x + w//2, y + h//2)
        faceROI = frame[y:y+h,x:x+w]
        cv2.putText(frame, abc, (5,20), font, 1, (139, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame , fps, (50, 20), font, 1, (139, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, hehe , (6, 50), font, 1,(139, 0, 0),1,cv2.LINE_AA)
        cv2.putText(frame, str(center), (105, 50), font, 1,(139, 0, 0),1,cv2.LINE_AA)
        
        #draw the face bounding box on the image
        face = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imshow('frame', faceROI)
        
        # Step 6: Show the frame with blurred background
        cv2.imshow("Webcam", frame)
    
    # If q is pressed terminate
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Release and destroy all windows
vs.release()
cv2.destroyAllWindows()