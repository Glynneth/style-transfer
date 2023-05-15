#!/usr/bin/python
from pathlib import Path

from PIL import Image

root_path = Path.cwd()
dir = root_path / "data" / "original"
images = Path(dir).glob('*.jpg')
out_dir = root_path / "data" / "resized"

def crop_max_square(pil_img):
    return crop_center(pil_img, min(pil_img.size), min(pil_img.size))

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


for path in Path(dir).iterdir():
    if path.is_file() and path.suffix == '.jpg':
        im = Image.open(path)
        im_cropped = crop_max_square(im)
        f_name = path.stem
        imResize = im_cropped.resize((200,200), Image.ANTIALIAS)
        imResize.save(out_dir/(f_name + '_resized.jpg'), 'JPEG', quality=90)


    