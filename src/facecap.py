import os
import cv2
import pickle
import datetime

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_default.xml')
# face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
# smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
# banana_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_banana.xml')
# car_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_cars.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizers/face-trainner.yml")

labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

cap = cv2.VideoCapture(0)

while True:
    # Capture Frame by Frame
    ret, frame = cap.read()
    if type(frame) == type(None):
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # banana = banana_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    # cars = car_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)

    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
        roi_color = frame[y:y + h, x:x + w]

        if x != 0 and y != 0 and w != 0 and h != 0:
            # os.system("say Object_detected")
            text = "Occupied"
            cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        else:
            text = "Unoccupied"
            cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            # pass

        # Recognizer
        id_, conf = recognizer.predict(roi_gray)
        # print(conf)
        # if conf >= 45:
        if 2 <= conf <= 85:
            # print("Label= ", labels[id_])
            # print("ID= ", id_)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0, 0, 255)
            stroke = 1
            cv2.putText(frame, name, (x, y), font, 2, color, stroke, cv2.LINE_AA)

        img_item = "images/fabrice/my_image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 230, 40)
        stroke = 2  # Thickness of the line
        endcoord_x = x + w
        endcoord_y = y + h
        cv2.rectangle(frame, (x, y), (endcoord_x, endcoord_y), color, stroke)

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), stroke)

        # smilee = smile_cascade.detectMultiScale(roi_gray)
        # for (sx, sy, sw, sh) in smilee:
        #     cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 240), stroke)

    # for (bx, by, bw, bh) in banana:
    #     roi_gray = gray[by:by + bh, bx:bx + bw]  # (ycord_start, ycord_end)
    #     roi_color = frame[by:by + bh, bx:bx + bw]
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     color1 = (255, 20, 20)
    #     color2 = (60, 240, 255)
    #     stroke = 1
    #     cv2.putText(roi_color, 'Banana', (bx, by), font, 2, color1, stroke, cv2.LINE_AA)
    #     cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), color2, stroke)
    #
    # for (cx, cy, cw, ch) in cars:
    #     roi_gray = gray[cy:cy + ch, cx:cx + cw]  # (ycord_start, ycord_end)
    #     roi_color = frame[cy:cy + ch, cx:cx + cw]
    #     font = cv2.FONT_HERSHEY_SIMPLEX
    #     color3 = (255, 255, 255)
    #     color4 = (102, 255, 102)
    #     stroke = 1
    #     cv2.putText(roi_color, 'Cars', (cx, cy), font, 2, color3, stroke, cv2.LINE_AA)
    #     cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), color4, stroke)

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
