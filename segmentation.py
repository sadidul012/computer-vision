import cv2
import numpy as np
import torch
from PIL import Image
from transformers import MaskFormerFeatureExtractor, MaskFormerForInstanceSegmentation, AutoConfig

device = torch.device("cuda:0")
config = AutoConfig.from_pretrained("facebook/maskformer-swin-small-ade")
feature_extractor = MaskFormerFeatureExtractor.from_pretrained("facebook/maskformer-swin-small-ade")
model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-small-ade")
model.to(device=device)
colors = {}
for index, label in enumerate(config.id2label.keys()):
    colors[label] = np.random.randint(0, 255, 4)


def segment(frame):
    image = Image.fromarray(frame)
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = inputs.to(device=device)
    outputs = model(**inputs)
    # you can pass them to feature_extractor for postprocessing
    # we refer to the demo notebooks for visualization (see "Resources" section in the MaskFormer docs)
    semantic_map = feature_extractor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]
    semantic_map = semantic_map.detach().cpu().numpy()

    labels = []
    img = np.zeros((semantic_map.shape[0], semantic_map.shape[1], 4))
    for i in np.unique(semantic_map):
        img[semantic_map == i] = colors[i]
        labels.append([config.id2label[i], colors[i]])

    background = cv2.cvtColor(frame, cv2.COLOR_RGB2RGBA)
    foreground = img

    background[:, :, 3] = 255.0
    foreground[:, :, 3] = 150
    alpha_background = background[:, :, 3] / 255.0
    alpha_foreground = foreground[:, :, 3] / 255.0

    for color in range(0, 3):
        background[:, :, color] = alpha_foreground * foreground[:, :, color] + alpha_background * background[:, :, color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    background[:, :, 3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    return background, labels


def main():
    image = cv2.imread("data/image.jpg")
    # draw = ImageDraw.Draw(image)
    background, _ = segment(image)
    # semantic_map[semantic_map == 1] = (0, 255, 0)
    # image = Image.fromarray(semantic_map)
    # image.save("data/ImageDraw.png")

    cv2.imshow("image", background/255)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # h, w = semantic_map.shape[0], semantic_map.shape[1]  # image dimensions
    # img_0 = semantic_map[0].reshape(3, h, w).transpose(1, 2, 0)
    # print(img_0.shape)


if __name__ == "__main__":
    main()
