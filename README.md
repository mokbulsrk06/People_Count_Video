# People_Count_Video
import cv2
import numpy as np
from ultralytics import YOLO
import cvzone

# Mouse callback function for RGB window
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        point = [x, y]
        print(point)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

# Load YOLO11 model
model = YOLO("yolo11n.pt")
names = model.names

# Open the video file
cap = cv2.VideoCapture('peoplecount1.mp4')
count = 0

# Define your areas
area1 = [(250,444),(211,448),(473,575),(514,556)]
area2 = [(201,449),(177,453),(420,581),(457,577)]

# Track IDs who enter and exit
enter = {}
exit = {}
list_enter = []
list_exit = []

# ==================== NEW: Setup video writer ====================
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('result.avi', fourcc, fps, (1020, 600))  # matches resize
# ================================================================

while True:
    ret, frame = cap.read()
    count += 1
    if count % 2 != 0:
        continue
    if not ret:
        break

    frame = cv2.resize(frame, (1020, 600))

    # Run YOLO tracking
    results = model.track(frame, persist=True)

    if results[0].boxes is not None and results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.int().cpu().tolist()
        class_ids = results[0].boxes.cls.int().cpu().tolist()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, class_id, track_id in zip(boxes, class_ids, track_ids):
            c = names[class_id]
            if c == 'person':
                x1, y1, x2, y2 = box
                
                # Check entry region
                if cv2.pointPolygonTest(np.array(area2, np.int32), (x1, y2), False) >= 0:
                    enter[track_id] = (x1, y2)

                # Check exit region and add to list
                if track_id in enter and cv2.pointPolygonTest(np.array(area1, np.int32), (x1, y2), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)
                    cv2.circle(frame, (x1, y2), 4, (255, 0, 0), -1)
                    if track_id not in list_enter:
                        list_enter.append(track_id)

                # Check exiting the shop
                if cv2.pointPolygonTest(np.array(area1, np.int32), (x1, y2), False) >= 0:
                    exit[track_id] = (x1, y2)
                
                if track_id in exit and cv2.pointPolygonTest(np.array(area2, np.int32), (x1, y2), False) >= 0:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cvzone.putTextRect(frame, f'{track_id}', (x1, y1), 1, 1)
                    cv2.circle(frame, (x1, y2), 4, (255, 0, 0), -1)
                    if track_id not in list_exit:
                        list_exit.append(track_id)

    enterinshop = len(list_enter)
    exitfromshop = len(list_exit)
    cvzone.putTextRect(frame, f'Enter: {enterinshop}', (50, 60), 2, 2)
    cvzone.putTextRect(frame, f'Exit: {exitfromshop}', (50, 100), 2, 2)

    # Draw the polygons
    cv2.polylines(
        frame, [np.array(area1, np.int32)],
        isClosed=True, color=(255, 0, 255), thickness=2
    )
    cv2.polylines(
        frame, [np.array(area2, np.int32)],
        isClosed=True, color=(255, 0, 255), thickness=2
    )

    # ==================== NEW: Save the output frame ====================
    out.write(frame)
    # ====================================================================

    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()  # ✅ Save file
cv2.destroyAllWindows()
