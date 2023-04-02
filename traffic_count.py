from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *

FILE_PATH = "Videos/10fps.mp4" 
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]
# For Mask
mask = cv2.imread('masks/10dtp_1.png')

# For Tracker
tracker = Sort(max_age=200, min_hits=3, iou_threshold=0.3)
cap  = cv2.VideoCapture(FILE_PATH)
model = YOLO("yolo_weights/yolov8n.pt")

# Initialize the coordinates for line which is specific for the video
limitsUp = [834, 260, 837, 442]
limitsDown = [335, 649, 592, 492]

totalCountUp = []
totalCountDown = []
current_frame = 0

while True:
    _, img = cap.read()
    if img is None: 
        print('Completed') # At the end of video, print completed and break code
        break
    img_reg = cv2.bitwise_and(img, mask)

    results = model(img_reg, stream=True)
    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
            w,h = x2-x1, y2-y1

            # Classname
            cls = int(box.cls[0])

            # Confodence score
            conf = math.ceil(box.conf[0]*100)/100
            
            if conf > 0.6:
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (x2,y2), scale=1, thickness=1)
                # cvzone.cornerRect(img, (x1,y1,w,h), l=9)
                currentArray = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, currentArray))

    resultTracker = tracker.update(detections)

    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)   

    for res in resultTracker:
        current_frame+=1
        x1,y1,x2,y2,id = res
        x1,y1,x2,y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
        w,h = x2-x1, y2-y1

        cvzone.putTextRect(img, f'{id}', (x1,y1), scale=1, thickness=1, colorR=(0,0,255))
        cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=1, colorR=(255,0,255))

        cx, cy = x1 + w // 2, y1 + h // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        if limitsUp[0] - 15 < cx < limitsUp[2] + 15 and limitsUp[1] < cy < limitsUp[3]:
            if totalCountUp.count(id) == 0:
                totalCountUp.append(id)
                cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
                cv2.putText(img,'Exit:' + str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
                cv2.putText(img,'Entry:' + str(len(totalCountDown)),(929,145),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)
                cv2.imwrite('C:/Users/itani/Downloads/roi/GitHub_Projects/Image_Capturer_From_Video/Traffic_Offenders/vehicle@' +str(current_frame) + '.jpg', img)

        if limitsDown[0] < cx < limitsDown[2] and limitsDown[3] < cy < limitsDown[1]:
            if totalCountDown.count(id) == 0:
                totalCountDown.append(id)
                cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
    
    cv2.putText(img,str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
    cv2.putText(img,str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)

    cv2.imshow('Image', img)
    #cv2.imshow('ImgRead', img_reg)
    if cv2.waitKey(1) == ord('q'):
        break
