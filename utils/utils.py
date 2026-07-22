from torch.utils.data import Dataset
import os
import random
import torch
from PIL import Image, ImageFile
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = root
        self.transform = transform
        self.files = [
            p for p in os.listdir(root)
            if p.lower().endswith(('.jpg', '.png', '.jpeg'))
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        max_attempts = 5

        for attempt in range(max_attempts):
            try:
                image_path = os.path.join(self.root, self.files[idx])
                image = Image.open(image_path).convert('RGB')

                # skip extremely large images
                if image.size[0] * image.size[1] > 50_000_000:
                    raise ValueError(
                        f"Image too large: {image.size}"
                    )

                if self.transform:
                    image = self.transform(image)
                return image

            except Exception as e:
                print(
                    f"Attempt {attempt+1} failed for idx {idx}: {e}"
                )
                idx = random.randint(0, len(self.files) - 1)

        # all attempts failed — return black image
        print("All attempts failed — returning black image")
        return torch.zeros(3, 256, 256)


def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))
    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)


def adaptive_instance_normalization(content_feat, style_feat):
    size = content_feat.size()
    style_mean, style_std   = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)
    normalized = (
        content_feat - content_mean.expand(size)
    ) / content_std.expand(size)
    return normalized * style_std.expand(size) + style_mean.expand(size)


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert len(size) == 4
    batch_size, channels = size[:2]
    feat_mean = (
        feat.view(batch_size, channels, -1)
            .mean(dim=2)
            .view(batch_size, channels, 1, 1)
    )
    feat_var = (
        feat.view(batch_size, channels, -1)
            .var(dim=2, unbiased=False) + eps
    )
    feat_std = feat_var.sqrt().view(batch_size, channels, 1, 1)
    return feat_mean, feat_std