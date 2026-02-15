import grain.python as grain
from datasets import load_dataset
import numpy as np
from PIL import Image


class CelebAInfo:
    url = "nielsr/CelebA-faces"
    train_length = 202599
    crop_resize = 64
    shape = (crop_resize, crop_resize, 3)


class _Normalize(grain.MapTransform):
    def map(self, element):
        img = element.astype(np.float32) / 255.0
        return img * 2.0 - 1.0


class _CenterCropAndResize(grain.MapTransform):
    def __init__(self, output_size=CelebAInfo.crop_resize):
        self.output_size = output_size

    def map(self, data):
        img = data["image"]
        img = Image.fromarray(img)

        width, height = img.size

        crop_size = min(width, height)
        left = (width - crop_size) / 2
        top = (height - crop_size) / 2
        right = (width + crop_size) / 2
        bottom = (height + crop_size) / 2

        img_cropped = img.crop((left, top, right, bottom))

        img_resized = img_cropped.resize(
            (self.output_size, self.output_size),
            Image.LANCZOS,
        )

        return np.array(img_resized)


def get_celeba_dataset(
    train_batch_size: int, num_epochs: int, seed: int, train_worker_count: int = 4
):
    ds = load_dataset(CelebAInfo.url)
    ds = ds.with_format("numpy")["train"]

    try:
        shard_options = grain.ShardByJaxProcess(drop_remainder=True)

    except ImportError:
        shard_options = grain.ShardOptions(
            shard_count=0, shard_index=1, drop_remainder=True
        )

    train_loader = grain.load(
        ds,
        num_epochs=num_epochs,
        shard_options=shard_options,
        shuffle=True,
        seed=seed,
        drop_remainder=True,
        batch_size=train_batch_size,
        worker_count=train_worker_count,
        transformations=[_CenterCropAndResize(), _Normalize()],
    )
    return train_loader
