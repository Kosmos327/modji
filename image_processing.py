from __future__ import annotations

from io import BytesIO
from typing import Iterable
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, UnidentifiedImageError

CANVAS_SIZE = 100
SCALE_RATIO = 0.85
MAX_CONTENT_SIZE = int(CANVAS_SIZE * SCALE_RATIO)
BACKGROUND_THRESHOLD = 240
PROCESSING_MODE_FAST = "FAST"
PROCESSING_MODE_CLEAN = "CLEAN"
REDRAW_DEFAULT_COLORS = 64
REDRAW_DEFAULT_BLUR = 0.35
REDRAW_DEFAULT_SHARPEN = 105
REDRAW_DEFAULT_SCALE = 0.92
REDRAW_DEFAULT_OUTLINE = True
REDRAW_DEFAULT_OUTLINE_THICKNESS = 1


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


def _expand_bbox(
    bbox: tuple[int, int, int, int], image_size: tuple[int, int], ratio: float = 0.3
) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top
    if width <= 0 or height <= 0:
        return bbox

    pad_x = int(round(width * ratio / 2))
    pad_y = int(round(height * ratio / 2))
    image_w, image_h = image_size
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(image_w, right + pad_x),
        min(image_h, bottom + pad_y),
    )


def _detect_face_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    try:
        import cv2
    except Exception:  # pragma: no cover - runtime dependency failure
        return None

    gray = np.array(image.convert("L"))
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return None

    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(20, 20),
    )
    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
    return _expand_bbox((int(x), int(y), int(x + w), int(y + h)), image.size, ratio=0.3)


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


def apply_emoji_style(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.split()[-1]

    styled_rgb = ImageEnhance.Contrast(rgba.convert("RGB")).enhance(1.3)
    styled_rgb = ImageEnhance.Color(styled_rgb).enhance(1.2)
    styled_rgb = styled_rgb.filter(ImageFilter.SMOOTH)
    styled_rgb.putalpha(alpha)
    return styled_rgb


def clamp_redraw_settings(
    *,
    colors: int,
    blur: float,
    sharpen: int,
    scale: float,
    outline_thickness: int,
) -> tuple[int, float, int, float, int]:
    clamped_colors = max(32, min(128, int(colors)))
    clamped_blur = max(0.0, min(0.8, float(blur)))
    clamped_sharpen = max(60, min(140, int(sharpen)))
    clamped_scale = max(0.82, min(0.96, float(scale)))
    clamped_outline_thickness = max(0, min(2, int(outline_thickness)))
    return (
        clamped_colors,
        clamped_blur,
        clamped_sharpen,
        clamped_scale,
        clamped_outline_thickness,
    )


def apply_redraw_style(
    image: Image.Image,
    *,
    redraw_colors: int = REDRAW_DEFAULT_COLORS,
    redraw_blur: float = REDRAW_DEFAULT_BLUR,
    redraw_sharpen: int = REDRAW_DEFAULT_SHARPEN,
) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.split()[-1]
    quantized = rgba.convert("RGB").quantize(
        colors=redraw_colors,
        method=Image.Quantize.MEDIANCUT,
    )
    redrawn = quantized.convert("RGB").filter(ImageFilter.GaussianBlur(radius=redraw_blur))
    redrawn = redrawn.filter(
        ImageFilter.UnsharpMask(radius=1, percent=redraw_sharpen, threshold=3)
    )
    redrawn.putalpha(alpha)
    return redrawn


def build_emoji_image(
    image_bytes: bytes,
    remove_background: bool = False,
    mode: str | None = None,
    use_face_detection: bool = False,
    apply_style: bool = False,
    with_outline: bool = False,
    outline_thickness: int = 2,
    redraw_mode: bool = False,
    redraw_colors: int = REDRAW_DEFAULT_COLORS,
    redraw_blur: float = REDRAW_DEFAULT_BLUR,
    redraw_sharpen: int = REDRAW_DEFAULT_SHARPEN,
    redraw_scale: float = REDRAW_DEFAULT_SCALE,
    redraw_outline: bool = REDRAW_DEFAULT_OUTLINE,
    redraw_outline_thickness: int = REDRAW_DEFAULT_OUTLINE_THICKNESS,
) -> Image.Image:
    # 1-2. Open image and normalize to RGBA.
    image = _open_rgba(image_bytes)

    # 3. Optional background removal.
    image = _remove_background_if_enabled(
        image, enabled=_should_remove_background(remove_background, mode)
    )

    # 4. Upscale too-small images before object detection.
    image = _upscale_if_small(image, min_side=CANVAS_SIZE)

    # 5-6. Face-centering priority with fallback bbox detection.
    bbox: tuple[int, int, int, int] | None = None
    if use_face_detection:
        bbox = _detect_face_bbox(image)
    if bbox is None:
        bbox = _content_bbox(image)
    image = image.crop(bbox)

    # 7. Scale object proportionally to fit based on mode-specific ratio.
    if redraw_mode:
        (
            redraw_colors,
            redraw_blur,
            redraw_sharpen,
            redraw_scale,
            redraw_outline_thickness,
        ) = clamp_redraw_settings(
            colors=redraw_colors,
            blur=redraw_blur,
            sharpen=redraw_sharpen,
            scale=redraw_scale,
            outline_thickness=redraw_outline_thickness,
        )

    max_content_size = (
        int(CANVAS_SIZE * redraw_scale) if redraw_mode else MAX_CONTENT_SIZE
    )
    image = _resize_to_fit(image, max_side=max_content_size)

    if redraw_mode:
        image = apply_redraw_style(
            image,
            redraw_colors=redraw_colors,
            redraw_blur=redraw_blur,
            redraw_sharpen=redraw_sharpen,
        )
    elif apply_style:
        image = apply_emoji_style(image)

    if redraw_mode and redraw_outline:
        image = add_outline(image, thickness=redraw_outline_thickness)

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


def export_batch_zip(images: Iterable[Image.Image]) -> bytes:
    zip_buffer = BytesIO()
    with ZipFile(zip_buffer, mode="w", compression=ZIP_DEFLATED) as zip_file:
        for index, image in enumerate(images, start=1):
            png_data, _ = export_png_webp(image)
            zip_file.writestr(f"emoji_{index}.png", png_data)
    return zip_buffer.getvalue()
