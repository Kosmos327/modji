from __future__ import annotations

from io import BytesIO

import numpy as np
from PIL import Image, ImageFilter, UnidentifiedImageError

CANVAS_SIZE = 100
SCALE_RATIO = 0.85
MAX_CONTENT_SIZE = int(CANVAS_SIZE * SCALE_RATIO)
BACKGROUND_THRESHOLD = 240
PROCESSING_MODE_FAST = "FAST"
PROCESSING_MODE_CLEAN = "CLEAN"


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
    if np.all(alpha == 255):
        grayscale = np.array(image.convert("L"))
        ys, xs = np.where(grayscale < BACKGROUND_THRESHOLD)
    else:
        ys, xs = np.where(alpha > 0)

    if xs.size == 0 or ys.size == 0:
        raise ImageProcessingError("Image is empty after processing.")
    left, right = int(xs.min()), int(xs.max())
    top, bottom = int(ys.min()), int(ys.max())
    return left, top, right + 1, bottom + 1


def _resize_to_fit(image: Image.Image, max_side: int = MAX_CONTENT_SIZE) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ImageProcessingError("Invalid image dimensions.")
    scale = min(max_side / width, max_side / height)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _upscale_if_small(image: Image.Image, min_side: int = CANVAS_SIZE) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ImageProcessingError("Invalid image dimensions.")
    if width >= min_side and height >= min_side:
        return image
    scale = max(min_side / width, min_side / height)
    new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def _should_remove_background(remove_background: bool, mode: str | None) -> bool:
    if mode is None:
        return remove_background
    normalized = mode.upper()
    if normalized == PROCESSING_MODE_FAST:
        return False
    if normalized == PROCESSING_MODE_CLEAN:
        return True
    raise ImageProcessingError(f"Unsupported processing mode: {mode}")


def add_outline(image: Image.Image, thickness: int = 2) -> Image.Image:
    source = image.convert("RGBA")
    if thickness <= 0:
        return source

    alpha = source.split()[-1]
    kernel_size = max(3, 2 * thickness + 1)
    expanded = alpha.filter(ImageFilter.MaxFilter(kernel_size))

    outline = Image.new("RGBA", source.size, (255, 255, 255, 0))
    outline.putalpha(expanded)
    return Image.alpha_composite(outline, source)


def _apply_unsharp(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    rgb = rgba.convert("RGB").filter(
        ImageFilter.UnsharpMask(radius=1, percent=120, threshold=3)
    )
    alpha = rgba.split()[-1]
    rgb.putalpha(alpha)
    return rgb


def build_emoji_image(
    image_bytes: bytes,
    remove_background: bool = False,
    mode: str | None = None,
    with_outline: bool = False,
    outline_thickness: int = 2,
) -> Image.Image:
    # 1-2. Open image and normalize to RGBA.
    image = _open_rgba(image_bytes)

    # 3. Optional background removal.
    image = _remove_background_if_enabled(
        image, enabled=_should_remove_background(remove_background, mode)
    )

    # 4. Upscale too-small images before object detection.
    image = _upscale_if_small(image, min_side=CANVAS_SIZE)

    # 5-6. Detect object bbox and crop to it.
    bbox = _content_bbox(image)
    image = image.crop(bbox)

    # 7. Scale object proportionally to fit based on SCALE_RATIO.
    image = _resize_to_fit(image, max_side=MAX_CONTENT_SIZE)

    if with_outline:
        image = add_outline(image, thickness=outline_thickness)

    # 8-9. Create exact 100x100 transparent canvas and center object.
    canvas = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (0, 0, 0, 0))
    x = (CANVAS_SIZE - image.width) // 2
    y = (CANVAS_SIZE - image.height) // 2
    canvas.paste(image, (x, y), image)

    # 10. Light sharpening optimized for emoji edges.
    canvas = _apply_unsharp(canvas)
    return canvas.convert("RGBA")


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
