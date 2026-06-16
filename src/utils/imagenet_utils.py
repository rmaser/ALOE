from src.eval.zero_shot_templates import IMAGENET_CLASSNAMES

IMAGENET_1K_CLASSES = {i: c for i, c in enumerate(IMAGENET_CLASSNAMES)}

def get_imagenet_label_name(label_id: int) -> str:
    """
    Returns the ImageNet label name for a given label ID.
    
    Args:
        label_id (int): The ImageNet label ID (0-999).
        
    Returns:
        str: The class name.
    """
    if label_id in IMAGENET_1K_CLASSES:
        return IMAGENET_1K_CLASSES[label_id]
    else:
        raise ValueError(f"Label ID {label_id} not found in ImageNet classes.")
