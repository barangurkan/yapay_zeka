from ultralytics import YOLO
import cv2


def is_sitting_on_chair(person_box, chair_box):
    # person_box ve chair_box: [x1, y1, x2, y2]
    px_center = (person_box[0] + person_box[2]) // 2
    py_bottom = person_box[3]

    if chair_box[0] <= px_center <= chair_box[2] and chair_box[1] <= py_bottom <= chair_box[3]:
        return True
    return False


# YOLO modeli yükle
model = YOLO("yolo12m.pt")

img = cv2.imread("oturanlar.jpg")

results = model(img,save=True)[0]

person_boxes = []
chair_boxes = []

for result in results.boxes:
    cls = int(result.cls[0])
    label = model.names[cls]
    x1, y1, x2, y2 = map(int, result.xyxy[0])

    if label == "person":
        person_boxes.append([x1, y1, x2, y2])
    elif label in ["chair", "bench", "couch"]:
        chair_boxes.append([x1, y1, x2, y2])

sitting_people = 0

for person_box in person_boxes:
    for chair_box in chair_boxes:
        if is_sitting_on_chair(person_box, chair_box):
            sitting_people += 1
            break

for box in person_boxes:
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    center_x = (box[0] + box[2]) // 2
    center_y = box[3]
    cv2.circle(img, (center_x, center_y), 5, (255, 0, 255), -1)

for box in chair_boxes:
    cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

cv2.putText(img, f"Oturuyor: {sitting_people}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Library Occupancy Tracking", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(sitting_people)
