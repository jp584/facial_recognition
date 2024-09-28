import cv2

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
# ML model
recognizer = cv2.face.LBPHFaceRecognizer_create(radius=1, neighbors=8, grid_x=8, grid_y=8)
recognizer.read("Trainer.yml")
name_dict = {2: "Nuevo", 1: "Juan Pablo Via"}

# Set confidence threshold for unknown faces (lower value = more confident)
CONFIDENCE_THRESHOLD = 70  # Adjust this value based on your needs

while True:
    ret, frame = video.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        serial, conf = recognizer.predict(gray[y:y+h, x:x+w])
        # Confidence threshold: lower confidence = better match
        print(f"Serial: {serial}, Confidence: {conf}")
        if conf < CONFIDENCE_THRESHOLD:
            # Recognized face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for recognized faces
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 255, 0), -1)
            cv2.putText(frame, name_dict.get(serial, "Desconocido"), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            # Unknown face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for unknown faces
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (0, 0, 255), -1)
            cv2.putText(frame, "Desconocido", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k == ord("q"):
        break

video.release()
cv2.destroyAllWindows()