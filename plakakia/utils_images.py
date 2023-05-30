import cv2
from pathlib import Path

def read_input_image(im_fname=None, settings=None):
    extension = settings.input_extension_images

    if extension in ['jpg', 'png']:
        image_filename = str(Path(settings.input_dir_images).joinpath(f"{im_fname}.{extension}"))
        image = cv2.imread(image_filename)
        # Pad the image if needed
        if settings.pad_image:
            image = add_border(image, settings=settings, color=[0, 0, 0])
    else:
        raise NotImplementedError(f"Extension {extension} not implemented.")

    return image

def read_input_mask(im_fname=None, settings=None):
    extension = settings.input_extension_masks

    if extension in ['jpg', 'png']:
        mask_filename = str(Path(settings.input_dir_masks).joinpath(f"{im_fname}.{extension}"))
        mask = cv2.imread(mask_filename)
        # Pad the mask if needed
        if settings.pad_image:
            mask = add_border(mask, settings=settings, color=[0, 0, 0])
    else:
        raise NotImplementedError(f"Extension {extension} not implemented.")

    return mask

def add_border(image, settings, color=[0, 0, 0]):
    """ Add border to an image. """

    top=settings.pad_size
    bottom=settings.pad_size
    left=settings.pad_size
    right=settings.pad_size

    if isinstance(image, str):
        image = cv2.imread(image)

    # Create border
    border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return border

