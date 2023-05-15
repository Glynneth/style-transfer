from PIL import Image  # type: ignore

from style_transfer_gs_2023 import ROOT_PATH


def style_transfer() -> None:
    resized_imgs_path = ROOT_PATH / "data" / "resized"
    content_image = Image.open(resized_imgs_path / "content_resized.jpg")
    style_image = Image.open(resized_imgs_path / "portrait_arden_resized.jpg")
    content_image.show()


if __name__ == "__main__":
    style_transfer()
