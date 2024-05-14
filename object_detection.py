import cv2
import torch
from PIL import Image, ImageDraw
from transformers import AutoImageProcessor, AutoModelForObjectDetection

device = torch.device("cuda:1")
image_processor = AutoImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = AutoModelForObjectDetection.from_pretrained("facebook/detr-resnet-50")
model.to(device=torch.device(device))


def detect_objects(image):
    image = Image.fromarray(image)
    with torch.no_grad():
        inputs = image_processor(images=image, return_tensors="pt")
        inputs = inputs.to(device=torch.device(device))
        outputs = model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.5, target_sizes=target_sizes)[0]

        return results


def main():
    frame = cv2.imread("data/image.jpg")
    frame = cv2.resize(frame, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    results = detect_objects(frame)
    print(frame.shape)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        if score.item() > 0.96:
            box = [round(i, 2) for i in box.tolist()]
            x, y, x2, y2 = tuple(box)
            cv2.rectangle(frame, (int(x), int(y)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(frame, f'{model.config.id2label[label.item()]}: {round(score.item(), 2)}', (int(x), int(y) - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 2)

    # cv2.putText(frame, f'Object Detection', (200, 200), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 0), 4)
    cv2.imwrite("data/ImageDraw.png", frame)


if __name__ == "__main__":
    main()
