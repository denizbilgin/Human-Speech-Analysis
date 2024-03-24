import math
import cv2
from ultralytics import YOLO
import cvzone

if __name__ == '__main__':
    # Video capturing with opencv
    cam = cv2.VideoCapture(0)   # Connecting to webcam to a cam variable

    # Adding image backgorund
    image_bg = cv2.imread("recources/background.png")
    image_bg = cv2.resize(image_bg, (1280, 720))

    # Creating instance of YOLO model
    model = YOLO("yolo-weights/yolov8n.pt")

    while True:
        success, frame = cam.read()

        # Applying the model and getting the results
        results = model(frame, stream=True)

        # Looping through the results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1

                cvzone.cornerRect(frame, (x1, y1, w, h))
                confidence = math.ceil((box.conf[0]*100))/100
                cvzone.putTextRect(frame, f"{confidence}", (x1, y1-20))

        #cv2.imshow("Webcam", frame)
        image_bg[116:116+480, 442:442+640] = frame
        cv2.imshow("Background", image_bg)

        # Closing the camera
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break