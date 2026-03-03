import os
import random
import collections

import numpy as np
import torch
from PIL import Image
from torch.utils import data
from torchvision import transforms
from os.path import join as pjoin
from collections import Counter
from tqdm import tqdm
from config import IGNORE_INDEX


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------

def convert_mask_to_array(mask, color_map):
    """Convert an RGB PIL mask to an integer label array using *color_map*."""
    if mask.mode == 'RGBA':
        mask = mask.convert('RGB')
    array = np.array(mask).astype('int32')
    output_array = np.zeros_like(array[:, :, 0], dtype=int)
    for rgb_color, integer_value in color_map.items():
        b_mask = np.all(array == np.array(rgb_color), axis=-1)
        output_array[b_mask] = integer_value
    return output_array


def compute_class_weights(loader, n_classes, ignore_index):
    """Compute inverse-frequency class weights from mask batches."""
    class_counts = Counter()

    for _, masks in tqdm(loader, desc="Computing class weights"):
        for c in range(n_classes):
            class_counts[c] += (masks == c).sum().item()

    total = sum(class_counts[c] for c in range(n_classes) if c != ignore_index)
    weights = torch.zeros(n_classes)

    for c in range(n_classes):
        if c == ignore_index or class_counts[c] == 0:
            weights[c] = 0.0
        else:
            weights[c] = total / class_counts[c]

    weights = weights / weights.mean()
    return weights


def collect_files_from_split(split_path, mask_dir):
    """Read a split file and collect all PNG file stems from the listed folders.

    Returns:
        list of str: relative paths without extension, e.g. ``"folder/stem"``.
    """
    with open(split_path, "r") as f:
        valid_folders = set(line.strip() for line in f if line.strip())

    files = []
    for folder in valid_folders:
        folder_path = os.path.join(mask_dir, folder)
        if os.path.isdir(folder_path):
            files_in_folder = [f for f in os.listdir(folder_path) if f.endswith(".png")]
            full_paths = [os.path.join(folder, os.path.splitext(f)[0]) for f in files_in_folder]
            files.extend(full_paths)
        else:
            print(f"Warning: Folder {folder_path} does not exist!")

    return files


def center_crop(img, crop_size=252):
    """Return a centered square crop from a PIL image."""
    width, height = img.size
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    return img.crop((left, top, right, bottom))


def load_mask(
    file_name,
    device,
    mask_dir,
    color_map,
    img_dim=(252, 252),
):
    """Load a mask image, center-crop it, and map RGB colors to class indices."""
    mask_path = f"{mask_dir}/{file_name}.png"
    mask_pil = Image.open(mask_path)
    mask_pil = center_crop(mask_pil, crop_size=img_dim[0])
    mask = convert_mask_to_array(mask_pil, color_map).reshape(img_dim[0] * img_dim[1])
    return torch.tensor(mask, dtype=torch.int64, device=device)


def class_labels(class_specs, ignore_index):
    """Return class names ordered by class index for non-ignored classes."""
    indexed = [
        (name, spec["index"])
        for name, spec in class_specs.items()
        if spec["index"] != ignore_index
    ]
    indexed.sort(key=lambda x: x[1])
    return [name for name, _ in indexed]


# ---------------------------------------------------------------------------
# Crop-validation helpers used by HystoDataset
# ---------------------------------------------------------------------------

def _is_black_or_white(pixel):
    return pixel == (0, 0, 0) or pixel == (255, 255, 255)


def _is_image_black_or_white(image):
    for pixel in image.getdata():
        if not _is_black_or_white(pixel):
            return False
    return True


def sufficient_color(image, threshold=0.01, threshold_strict=0.005,
                     strict_color_list=None, ignore_color_list=None):
    """Check if the percentages of colored pixels are above certain thresholds,
    or if specific colored pixels (defined by `ignore_color_list`) are present."""
    if _is_image_black_or_white(image):
        return False

    image_array = np.array(image)
    total_pixels = image_array.size // 3

    black_mask = np.all(image_array == 0, axis=-1)
    white_mask = np.all(image_array == 255, axis=-1)
    colored_pixels_mask = ~black_mask & ~white_mask
    colored_pixel_ratio = np.sum(colored_pixels_mask) / total_pixels

    ignore_color_present = False
    if ignore_color_list is not None:
        for color in ignore_color_list:
            mask = ((image_array[:, :, 0] == color[0]) &
                    (image_array[:, :, 1] == color[1]) &
                    (image_array[:, :, 2] == color[2]))
            if np.any(mask):
                ignore_color_present = True
                break

    strict_color_sufficient = False
    if strict_color_list is not None:
        for color in strict_color_list:
            color_mask = ((image_array[:, :, 0] == color[0]) &
                          (image_array[:, :, 1] == color[1]) &
                          (image_array[:, :, 2] == color[2]))
            if np.sum(color_mask) / total_pixels >= threshold_strict:
                strict_color_sufficient = True
                break

    return (colored_pixel_ratio >= threshold) or strict_color_sufficient or ignore_color_present


# ---------------------------------------------------------------------------
# Base dataset
# ---------------------------------------------------------------------------

class BaseHystoDataset(data.Dataset):
    """Shared helpers for histology segmentation datasets."""

    def convert_mask_to_array(self, mask):
        return convert_mask_to_array(mask, self.color_map)

    def random_crop_non_empty(self, img, mask, crop_size, max_retries=50):
        img_np = np.array(img)
        mask_np = np.array(mask)
        h, w = mask_np.shape[:2]

        for _ in range(max_retries):
            top = random.randint(0, h - crop_size)
            left = random.randint(0, w - crop_size)

            img_crop = img_np[top:top + crop_size, left:left + crop_size]
            mask_crop = mask_np[top:top + crop_size, left:left + crop_size]

            mask_crop_pil = Image.fromarray(mask_crop)
            if self._is_valid_crop(mask_crop_pil):
                return Image.fromarray(img_crop), Image.fromarray(mask_crop)

        # fallback: return the last crop
        return Image.fromarray(img_crop), Image.fromarray(mask_crop)

    def _is_valid_crop(self, mask_crop_pil):
        """Override in subclasses for custom crop validation."""
        mask_array = self.convert_mask_to_array(mask_crop_pil)
        return np.any(mask_array != IGNORE_INDEX)

    def _load_split_files(self, split_dir, split, img_dir):
        """Populate ``self.files`` from split text files."""
        self.files = collections.defaultdict(list)

        if split == "test":
            splits_to_load = [split]
        else:
            splits_to_load = ["train", "val"]

        for s in splits_to_load:
            path = pjoin(split_dir, s + ".txt")
            with open(path, "r") as f:
                folder_list = [line.strip() for line in f]
            file_list = []
            for rel_folder in folder_list:
                full_path = pjoin(img_dir, rel_folder)
                if not os.path.isdir(full_path):
                    continue
                for fname in os.listdir(full_path):
                    name, _ = os.path.splitext(fname)
                    file_list.append(f"{rel_folder}/{name}")
            self.files[s] = file_list


# ---------------------------------------------------------------------------
# HystoDataset -- the primary dataset used in this project
# ---------------------------------------------------------------------------

class HystoDataset(BaseHystoDataset):
    def __init__(
        self,
        split,
        mask_dir,
        img_dir,
        split_dir,
        color_map,
        crop_dim=252,
        normalize_mean=(0.5788, 0.3551, 0.5655),
        normalize_std=(1, 1, 1),
        color_threshold=0.01,
        strict_threshold=0.005,
        strict_color_list=None,
        ignore_color_list=None,
    ):
        """Customized dataset class. Takes png tiles with offset as an input, augments by cropping tiles at random position to desired size 
        (default 252x252) while taking restrictions for certain anomaly classes into account (function sufficient_color).
        Args:
            root: Root directory, contains subfolders "img" and "mask" 
            -> each of these subfolders contains study subfolders (e.g. "20_202") 
            -> each study subfolder contains WSI subfolders (e.g. "20_202_HE_101") 
            -> each WSI subfolder contains png tiles of tissue / png tiles with color encoded masks
            split_name: Name of subfolder in root directory which contains files "test.txt", train.txt, "val.txt"
            -> Each file contains list of relevant WSI-s (in format '20_202/20_202_HE_101')
            color_map: Dict with RGB color codes and class numbers
            split: 'train', 'val' or 'test'
        """
        self.mask_dir = mask_dir
        self.img_dir = img_dir
        self.color_map = color_map
        self.split = split
        self.crop_dim = crop_dim
        self.normalize_mean = normalize_mean
        self.normalize_std = normalize_std

        self.color_threshold = color_threshold
        self.strict_threshold = strict_threshold
        self.strict_color_list = strict_color_list
        self.ignore_color_list = ignore_color_list

        self._load_split_files(split_dir, split, img_dir)

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]

        img_path = pjoin(self.img_dir, img_name + ".png")
        img = Image.open(img_path)

        mask_path = pjoin(self.mask_dir, img_name + ".png")
        mask = Image.open(mask_path)

        if self.split == "train":
            img, mask = self.random_crop_non_empty(img, mask, self.crop_dim)

            img = transforms.functional.to_tensor(img)
            img = transforms.functional.normalize(img, mean=self.normalize_mean, std=self.normalize_std)

            mask = torch.from_numpy(self.convert_mask_to_array(mask)).long()
        else:
            img = transforms.CenterCrop(self.crop_dim)(img)
            img = transforms.functional.to_tensor(img)
            img = transforms.functional.normalize(img, mean=self.normalize_mean, std=self.normalize_std)

            mask = transforms.CenterCrop(self.crop_dim)(mask)
            mask = torch.from_numpy(self.convert_mask_to_array(mask)).long()

        return img, mask

    def _is_valid_crop(self, mask_crop_pil):
        return sufficient_color(
            mask_crop_pil,
            threshold=self.color_threshold,
            threshold_strict=self.strict_threshold,
            strict_color_list=self.strict_color_list,
            ignore_color_list=self.ignore_color_list,
        )
