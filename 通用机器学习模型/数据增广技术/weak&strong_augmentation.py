import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 加载图像
image_path = './example.png'
image = Image.open(image_path)
image_mix_path = './example2.png'
image_mix = Image.open(image_mix_path)


# 弱增强
weak_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(300, padding=4),
    transforms.ToTensor()
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# 强增强
strong_augmentation = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(300, padding=4),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor()
#     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# 应用弱增强
weak_augmented_image = weak_augmentation(image)

# 应用强增强
strong_augmented_image = strong_augmentation(image)


# 展示原始图像和增强后的图像
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[1].imshow(transforms.ToPILImage()(weak_augmented_image))
axes[1].set_title('Weak Augmentation')
axes[2].imshow(transforms.ToPILImage()(strong_augmented_image))
axes[2].set_title('Strong Augmentation')


for ax in axes:
    ax.axis('off')

plt.show()
