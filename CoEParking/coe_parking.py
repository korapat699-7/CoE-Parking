import cv2
import numpy as np
from ultralytics import YOLO
import torch

# print(torch.cuda.is_available()) 
# print(torch.cuda.get_device_name(0))

model = YOLO('yolo11m.pt')

mask_path = './mask_park.png'
video_path = './footage.mov'
mask = cv2.imread(mask_path, 0)

cap = cv2.VideoCapture(video_path)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

def is_in_parking_area(box, contour):
    x1, y1, x2, y2 = box
    center_x = (x1 + x2) // 2
    center_y = (y1 + y2) // 2
    return cv2.pointPolygonTest(contour, (center_x, center_y), False) >= 0

spots_status = [False] * len(contours)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device='cuda')

    boxes = results[0].boxes.xyxy.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy()

    car_count = 0
    spots_status = [False] * len(contours)

    for box, cls in zip(boxes, classes):
        if cls == 2:
            for idx, contour in enumerate(contours):
                if is_in_parking_area(box, contour):
                    car_count += 1
                    spots_status[idx] = True
                    x1, y1, x2, y2 = map(int, box)
                    break

    for idx, contour in enumerate(contours):
        if spots_status[idx]:
            cv2.polylines(frame, [contour], True, (0, 0, 255), 2)
        else:
            cv2.polylines(frame, [contour], True, (0, 255, 0), 2)

    cv2.putText(frame, f'Cars in parking: {car_count} / 12', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow('Parking Lot', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
