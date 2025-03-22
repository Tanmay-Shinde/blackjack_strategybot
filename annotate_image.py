from ultralytics import YOLO
import cv2
import yaml

img = "C:/Users/tanma/PycharmProjects/blackjack_strategybot/playing_cards_data/train/images/001198293_jpg.rf.411db15ce8a9a42a2d51a1885f7592d2.jpg"
img_anot = "C:/Users/tanma/PycharmProjects/blackjack_strategybot/playing_cards_data/train/labels/001198293_jpg.rf.411db15ce8a9a42a2d51a1885f7592d2.txt"

data_yaml_path = "C:/Users/tanma/PycharmProjects/blackjack_strategybot/playing_cards_data/data.yaml"

with open(data_yaml_path, "r") as f:
    data = yaml.safe_load(f)

label_names = data['names']

img = cv2.imread(img)

H, W, _ = img.shape

with open(img_anot, "r") as f:
    lines = f.readlines()

annotations = []
for line in lines:
    values = line.split()
    label = values[0]

    x, y, w, h = map(float, values[1:])
    annotations.append((label, x, y, w, h))

for annotation in annotations:
    label, x, y, w, h = annotation
    label_name = label_names[int(label)]

    # Convert Yolo coordinates to pixel coordinates
    x1 = int((x - w / 2) * W)
    y1 = int((y - h / 2) * H)
    x2 = int((x + w / 2) * W)
    y2 = int((y + h / 2) * H)

    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # display label name
    cv2.putText(img, label_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)

