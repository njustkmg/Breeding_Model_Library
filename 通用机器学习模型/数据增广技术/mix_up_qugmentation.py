import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

# 加载图像
image_path1 = './example.png'
image_path2 = './example2.png'
image1 = Image.open(image_path1)
image2 = Image.open(image_path2)

crop = transforms.Compose([
    transforms.CenterCrop(300),
    transforms.ToTensor(),
])
# MixUp增强
class MixUp:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, images):
        images = [crop(image) for image in images]
        batch = torch.stack(images)
        batch_size = batch.size(0)
        indices = torch.randperm(batch_size)
        shuffled_data = batch[indices]
        
        lam = torch.Tensor([np.random.beta(self.alpha, self.alpha)]).expand(batch_size, 1, 1, 1)
        mixed_data = lam * batch + (1 - lam) * shuffled_data
        mixed_images = [transforms.ToPILImage()(mixed_data[i]) for i in range(batch_size)]
        return mixed_images

# 应用MixUp增强
mixup = MixUp(alpha=0.6)
mixup_augmented_images = mixup([image1, image2])

# 展示原始图像和增强后的图像
fig, axes = plt.subplots(2, 2, figsize=(10, 10))
axes[0, 0].imshow(image1)
axes[0, 0].set_title('Original Image 1')
axes[0, 1].imshow(image2)
axes[0, 1].set_title('Original Image 2')
axes[1, 0].imshow(mixup_augmented_images[0])
axes[1, 0].set_title('MixUp Augmented Image 1')
axes[1, 1].imshow(mixup_augmented_images[1])
axes[1, 1].set_title('MixUp Augmented Image 2')

for ax in axes.flatten():
    ax.axis('off')

plt.show()
