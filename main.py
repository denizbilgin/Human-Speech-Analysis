import cv2

if __name__ == '__main__':
    # Video capturing with opencv
    cam = cv2.VideoCapture(0)   # Connecting to webcam to a cam variable

    # Adding image backgorund
    image_bg = cv2.imread("recources/background.png")
    image_bg = cv2.resize(image_bg, (1280, 720))

    # Creating instance of face detector model
    model = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

    num = 0

    while True:
        success, frame = cam.read()

        # Applying the cascade model and getting the results
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = model.detectMultiScale(gray, 1.3, 5)

        # Looping through the faces
        for face in faces:
            x, y, w, h = face

            # Drawing border boxes
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # Getting roi of the frame
            roi_gray = gray[y:y + h, x:x + w]

            # Original colored frame
            roi_color = frame[y:y + h, x:x + w]

        image_bg[116:116+480, 442:442+640] = frame

        if num == 100:
            cv2.imshow("100. frame budur", frame[y+2:y+h-2, x+2:x+w-2])
            cv2.waitKey(0)
        num += 1

        cv2.imshow("Background", image_bg)

        # Closing the camera
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break