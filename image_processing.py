from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError


class ImageProcessingError(Exception):
    """Raised when input image cannot be converted to a valid emoji image."""


def _open_rgba(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(BytesIO(image_bytes))
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageProcessingError("Corrupted or unsupported image file.") from exc
    return image.convert("RGBA")


def _remove_background_if_enabled(image: Image.Image, enabled: bool) -> Image.Image:
    if not enabled:
        return image
    try:
        from rembg import remove
    except Exception as exc:  # pragma: no cover - optional dependency runtime failures
        raise ImageProcessingError("Background removal is not available.") from exc
    output = remove(image)
    if isinstance(output, bytes):
        return Image.open(BytesIO(output)).convert("RGBA")
    return output.convert("RGBA")


def _content_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    alpha = np.array(image.split()[-1])
    ys, xs = np.where(alpha > 0)
    if xs.size == 0 or ys.size == 0:
        raise ImageProcessingError("Image is empty after processing.")
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    return left, top, right + 1, bottom + 1


def _resize_to_fit(image: Image.Image, max_side: int = 90) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ImageProcessingError("Invalid image dimensions.")
    scale = min(max_side / width, max_side / height)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def build_emoji_image(image_bytes: bytes, remove_background: bool = False) -> Image.Image:
    # 1-2. Open image and normalize to RGBA.
    image = _open_rgba(image_bytes)

    # 3. Optional background removal.
    image = _remove_background_if_enabled(image, enabled=remove_background)

    # 4-5. Detect non-transparent object bbox and crop to it.
    bbox = _content_bbox(image)
    image = image.crop(bbox)

    # 6. Scale object proportionally to fit in 90x90.
    image = _resize_to_fit(image, max_side=90)

    # 7-8. Create exact 100x100 transparent canvas and center object.
    canvas = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
    x = (100 - image.width) // 2
    y = (100 - image.height) // 2
    canvas.paste(image, (x, y), image)

    # 9. Very light optional sharpening.
    canvas = canvas.filter(ImageFilter.SHARPEN)
    return canvas


def export_png_webp(image: Image.Image) -> tuple[bytes, bytes]:
    if image.mode != "RGBA":
        image = image.convert("RGBA")
    if image.size != (100, 100):
        raise ImageProcessingError("Output must be exactly 100x100.")

    png_buffer = BytesIO()
    image.save(png_buffer, format="PNG", optimize=True)

    webp_buffer = BytesIO()
    image.save(webp_buffer, format="WEBP", quality=100, lossless=True)

    return png_buffer.getvalue(), webp_buffer.getvalue()
