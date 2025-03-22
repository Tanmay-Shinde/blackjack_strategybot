import cv2
from PIL import ImageGrab
import numpy as np
from screeninfo import get_monitors

for m in get_monitors():
    x: int = m.x
    y: int = m.y
    w: int = m.width
    h: int = m.height

while True:
    img = ImageGrab.grab(bbox=((x, y, w, h)))
    np_img = np.array(img)

    conv_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)

    cv2.imshow("Screen Capture", np_img)

    key = cv2.waitKey(20)
    if key == 27:
        break

cv2.destroyAllWindows()
