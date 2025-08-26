# src/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T

def get_transforms(is_train=True, image_size=(224, 224)):
    """
    Returns appropriate transforms for training or validation.
    Using albumentations for better performance.
    """
    if is_train:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
            A.Normalize(mean=[0.5], std=[0.5]),  # Normalize to [-1, 1]
            ToTensorV2(),  # Converts to tensor automatically
        ])
    else:
        return A.Compose([
            A.Resize(image_size[0], image_size[1]),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])