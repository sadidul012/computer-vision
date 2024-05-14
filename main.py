import cv2
import numpy as np
import tqdm

from object_detection import detect_objects, model
from segmentation import segment

cap = cv2.VideoCapture('data/traffic.mp4')
count = 0
progress = tqdm.tqdm(total=1000)
while cap.isOpened():
    progress.update(1)
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=.8, fy=.8, interpolation=cv2.INTER_CUBIC)
    results = detect_objects(frame)
    frame, labels = segment(frame)

    x, y, w, h = 10, 10, 200, 400
    sub_img = frame[y:y + h, x:x + w]
    white_rect = np.ones(sub_img.shape, dtype=np.uint8) * 255

    res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)

    # Putting the image back to its position
    frame[y:y + h, x:x + w] = res

    for li, ll in enumerate(labels):
        cv2.putText(frame, ll[0], (20, (20 * li) + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (int(ll[1][0]), int(ll[1][1]), int(ll[1][2]), 255), 2)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score.item() > 0.96:
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f'{model.config.id2label[label.item()]}: {round(score.item(), 2)}', (int(x), int(y)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 222, 222), 1)
    if count == 50:
        cv2.imwrite('data/output.png', frame)
    cv2.imshow('window-name', frame)
    # cv2.imwrite("frame%d.jpg" % count, frame)
    count = count + 1
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
