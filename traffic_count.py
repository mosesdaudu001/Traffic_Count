from ultralytics import YOLO
import cv2
import cvzone
import math
from sort import *
import os

class ObjectDetection():

    def __init__(self, capture, result):
        self.capture = capture
        self.result = result
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names

    def load_model(self):
        model = YOLO("yolo_weights/yolov8n.pt") 
        model.fuse()

        return model
    
    def predict(self, img):
        results = self.model(img, stream=True)
        return results
    
    def plot_boxes(self, results, detections):

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1,y1,x2,y2 = box.xyxy[0]
                x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)
                w,h = x2-x1, y2-y1

                # Classname
                cls = int(box.cls[0])
                currentClass = self.CLASS_NAMES_DICT[cls]

                # Confodence score
                conf = math.ceil(box.conf[0]*100)/100

                if conf > 0.5:
                    currentArray = np.array([x1,y1,x2,y2,conf])
                    detections = np.vstack((detections, currentArray))
                    
        return detections
   
    def track_detect(self, img, detections, tracker, limitsUp, limitsDown, totalCountUp, totalCountDown, current_frame):
        resultTracker = tracker.update(detections)

        cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 0, 255), 5)
        cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 0, 255), 5)

        for res in resultTracker:
            x1,y1,x2,y2,id = res
            x1,y1,x2,y2, id = int(x1), int(y1), int(x2), int(y2), int(id)
            w,h = x2-x1, y2-y1

            cvzone.putTextRect(img, f'ID: {id}', (x1,y1), scale=1, thickness=1, colorR=(0,0,255))
            cvzone.cornerRect(img, (x1,y1,w,h), l=9, rt=1, colorR=(255,0,255))

            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            if limitsUp[0] - 15 < cx < limitsUp[2] + 15 and limitsUp[1] < cy < limitsUp[3]:
                if totalCountUp.count(id) == 0:
                    totalCountUp.append(id)
                    cv2.line(img, (limitsUp[0], limitsUp[1]), (limitsUp[2], limitsUp[3]), (0, 255, 0), 5)
                    cv2.putText(img,'Exit:' + str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
                    cv2.putText(img,'Entry:' + str(len(totalCountDown)),(929,145),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)

            if limitsDown[0] < cx < limitsDown[2] and limitsDown[3] < cy < limitsDown[1]:
                if totalCountDown.count(id) == 0:
                    totalCountDown.append(id)
                    cv2.line(img, (limitsDown[0], limitsDown[1]), (limitsDown[2], limitsDown[3]), (0, 255, 0), 5)
        
        cv2.putText(img,str(len(totalCountUp)),(929,345),cv2.FONT_HERSHEY_PLAIN,5,(139,195,75),7)
        cv2.putText(img,str(len(totalCountDown)),(1191,345),cv2.FONT_HERSHEY_PLAIN,5,(50,50,230),7)

        return img

    def __call__(self):

        cap = cv2.VideoCapture(self.capture)
        assert cap.isOpened()

        result_path = os.path.join(self.result, 'results.avi')

        codec = cv2.VideoWriter_fourcc(*'XVID')
        vid_fps =int(cap.get(cv2.CAP_PROP_FPS))
        vid_width,vid_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(result_path, codec, vid_fps, (vid_width, vid_height))

        tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

        mask = cv2.imread('masks/10dtp_1.png')

        limitsUp = [834, 260, 837, 442]
        limitsDown = [335, 649, 592, 492]
        totalCountUp = []
        totalCountDown = []
        current_frame = 0

        if not os.path.exists(self.result):
            os.makedirs(self.result)
            print("Result folder created successfully")
        else:
            print("Result folder already exist")

        while True:

            _, img = cap.read()
            assert _
            img_reg = cv2.bitwise_and(img, mask)
            
            detections = np.empty((0,5))
            results = self.predict(img_reg)
            detections = self.plot_boxes(results, detections)
            detect_frame = self.track_detect(img, detections, tracker, limitsUp, limitsDown, totalCountUp, totalCountDown, current_frame)

            out.write(detect_frame)
            cv2.imshow('Image', detect_frame)
            if cv2.waitKey(1) == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        
        
    
detector = ObjectDetection(capture="Videos/10fps.mp4" , result='result')
detector()