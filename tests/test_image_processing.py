import unittest
from io import BytesIO

from PIL import Image

from image_processing import (
    ImageProcessingError,
    add_outline,
    build_emoji_image,
    export_png_webp,
)


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
        left, top, right, bottom = bbox
        # We allow +/-1 px tolerance around the ideal 7..93 bounds for resampling/rounding variance.
        self.assertTrue(6 <= left <= 8)
        self.assertTrue(6 <= top <= 8)
        self.assertTrue(92 <= right <= 94)
        self.assertTrue(92 <= bottom <= 94)

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

    def test_non_transparent_image_uses_grayscale_bbox(self) -> None:
        src = Image.new("RGBA", (220, 220), (255, 255, 255, 255))
        for x in range(70, 150):
            for y in range(80, 140):
                src.putpixel((x, y), (30, 30, 30, 255))

        out = build_emoji_image(_image_to_bytes(src))
        self.assertEqual(out.size, (100, 100))
        self.assertEqual(out.mode, "RGBA")

        alpha_bbox = out.split()[-1].getbbox()
        self.assertIsNotNone(alpha_bbox)
        left, top, right, bottom = alpha_bbox
        self.assertTrue(6 <= left <= 8)
        self.assertTrue(18 <= top <= 20)
        self.assertTrue(92 <= right <= 94)
        self.assertTrue(80 <= bottom <= 82)

    def test_small_image_is_upscaled_and_output_is_rgba_100(self) -> None:
        src = Image.new("RGBA", (40, 40), (0, 0, 0, 0))
        for x in range(5, 35):
            for y in range(5, 35):
                src.putpixel((x, y), (255, 0, 0, 255))

        out = build_emoji_image(_image_to_bytes(src))
        self.assertEqual(out.size, (100, 100))
        self.assertEqual(out.mode, "RGBA")
        bbox = out.split()[-1].getbbox()
        self.assertIsNotNone(bbox)
        left, top, right, bottom = bbox
        self.assertTrue(10 <= left <= 13)
        self.assertTrue(10 <= top <= 13)
        self.assertTrue(86 <= right <= 90)
        self.assertTrue(86 <= bottom <= 90)

    def test_fast_mode_disables_background_removal_flag(self) -> None:
        src = Image.new("RGBA", (120, 120), (255, 255, 255, 255))
        for x in range(30, 90):
            for y in range(30, 90):
                src.putpixel((x, y), (0, 0, 0, 255))
        out = build_emoji_image(
            _image_to_bytes(src),
            remove_background=True,
            mode="FAST",
        )
        self.assertEqual(out.size, (100, 100))
        self.assertEqual(out.mode, "RGBA")

    def test_add_outline_creates_white_border(self) -> None:
        src = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
        for x in range(8, 12):
            for y in range(8, 12):
                src.putpixel((x, y), (255, 0, 0, 255))

        outlined = add_outline(src, thickness=2)
        self.assertEqual(outlined.mode, "RGBA")
        self.assertEqual(outlined.size, src.size)
        self.assertEqual(outlined.getpixel((9, 9)), (255, 0, 0, 255))
        self.assertEqual(outlined.getpixel((7, 9)), (255, 255, 255, 255))


if __name__ == "__main__":
    unittest.main()
