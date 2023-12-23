from ultralytics import YOLO
import cv2
import cvzone
import math
import time

#cap = cv2.VideoCapture('C:/Users/Dell/Downloads/mixkit-professional-chefs-work-in-restaurant-kitchen-15875-medium.mp4')
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)


model = YOLO("C:/Users/Dell/Downloads/yolov8m.pt")

classNames = ["person", "bicycle", "car", "motorcycle", "airplane"]

prev_frame_time = 0
new_frame_time = 0
start_time = time.time()

person_count = 0
detected_persons = set()  # Set to store unique person identifiers

while True:
    new_frame_time = time.time()
    elapsed_time = new_frame_time - start_time

    success, img = cap.read()
    
    if not success:
        # Video has ended, reset counters
        person_count = 0
        detected_persons = set()
        start_time = time.time()
        break

    results = model(img, stream=True)
    for r in results:
        person_count=0
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            currentClass = classNames[0]

            # Unique identifier for each person
            person_identifier = (x1, y1, x2, y2, conf)

            if currentClass == "person" and conf > 0.8:
                if person_identifier not in detected_persons:
                    print(f"New person detected at {x1, y1}!")
                    cvzone.cornerRect(img, (x1, y1, w, h))
                    cvzone.putTextRect(img, f'{classNames[0]} {conf}', (max(0, x1), max(35, y1)), scale=0.8, thickness=1,
                                       offset=3)
                    detected_persons.add(person_identifier)
                    person_count += 1
            

    # Display person count in the top right corner for 1 second
    if elapsed_time <= 3153600000:
        cv2.putText(img, f'Persons: {person_count}', (img.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        #person_Count=0
    else:
        #person_Count=0
        cv2.putText(img, f'Final Persons: {person_count}', (img.shape[1] - 300, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print(f"FPS: {fps}, Persons Count: {person_count}")

    cv2.imshow("Image", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
