import unittest
from io import BytesIO

from PIL import Image

from image_processing import ImageProcessingError, build_emoji_image, export_png_webp


def _image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


class TestImageProcessing(unittest.TestCase):
    def test_build_emoji_image_centers_and_scales_content(self) -> None:
        src = Image.new("RGBA", (300, 200), (0, 0, 0, 0))
        for x in range(100, 200):
            for y in range(50, 150):
                src.putpixel((x, y), (255, 0, 0, 255))

        out = build_emoji_image(_image_to_bytes(src))
        self.assertEqual(out.size, (100, 100))
        self.assertEqual(out.mode, "RGBA")

        alpha = out.split()[-1]
        bbox = alpha.getbbox()
        self.assertIsNotNone(bbox)
        self.assertEqual(bbox, (5, 5, 95, 95))

    def test_export_png_webp_outputs_loadable_images(self) -> None:
        src = Image.new("RGBA", (100, 100), (0, 255, 0, 128))
        png_data, webp_data = export_png_webp(src)
        self.assertGreater(len(png_data), 0)
        self.assertGreater(len(webp_data), 0)

        png_img = Image.open(BytesIO(png_data))
        webp_img = Image.open(BytesIO(webp_data))
        self.assertEqual(png_img.size, (100, 100))
        self.assertEqual(webp_img.size, (100, 100))

    def test_corrupted_image_raises_error(self) -> None:
        with self.assertRaises(ImageProcessingError):
            build_emoji_image(b"not-an-image")

    def test_empty_transparent_image_raises_error(self) -> None:
        src = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
        with self.assertRaises(ImageProcessingError):
            build_emoji_image(_image_to_bytes(src))


if __name__ == "__main__":
    unittest.main()
