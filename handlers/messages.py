from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

from aiogram import F, Router
from aiogram.filters import Command
from aiogram.types import (
    BufferedInputFile,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from PIL import Image

from image_processing import (
    ImageProcessingError,
    REDRAW_DEFAULT_BLUR,
    REDRAW_DEFAULT_COLORS,
    REDRAW_DEFAULT_OUTLINE,
    REDRAW_DEFAULT_OUTLINE_THICKNESS,
    REDRAW_DEFAULT_SCALE,
    REDRAW_DEFAULT_SHARPEN,
    build_emoji_image,
    clamp_redraw_settings,
    export_batch_zip,
    export_png_webp,
)

router = Router()
logger = logging.getLogger(__name__)

SUPPORTED_MIME_PREFIXES = ("image/",)
ENABLE_BACKGROUND_REMOVAL = os.getenv("ENABLE_BACKGROUND_REMOVAL", "0") == "1"
BATCH_GROUP_WAIT_TIMEOUT = 1.2
NO_IMAGE_FOUND_MESSAGE = "Сначала отправь изображение"
REDRAW_BLUR_STEP = 0.05
REDRAW_SHARPEN_STEP = 5
REDRAW_SCALE_STEP = 0.01
REDRAW_COLORS_STEP = 8


@dataclass
class ProcessingOptions:
    mode: str = "FAST"
    use_face_detection: bool = False
    apply_style: bool = False
    with_outline: bool = False


@dataclass
class RedrawSettings:
    colors: int = REDRAW_DEFAULT_COLORS
    blur: float = REDRAW_DEFAULT_BLUR
    sharpen: int = REDRAW_DEFAULT_SHARPEN
    scale: float = REDRAW_DEFAULT_SCALE
    outline: bool = REDRAW_DEFAULT_OUTLINE
    outline_thickness: int = REDRAW_DEFAULT_OUTLINE_THICKNESS


USER_OPTIONS: dict[int, ProcessingOptions] = {}
USER_REDRAW_SETTINGS: dict[int, RedrawSettings] = {}
USER_LAST_FILE_IDS: dict[int, str] = {}
USER_LAST_ORIGINAL_IMAGES: dict[int, bytes] = {}
MEDIA_GROUP_ITEMS: dict[str, list[tuple[Message, bytes]]] = {}
MEDIA_GROUP_TASKS: dict[str, asyncio.Task[Any]] = {}


def _mode_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Обычный", callback_data="mode:normal"),
                InlineKeyboardButton(text="По лицу", callback_data="mode:face"),
            ],
            [
                InlineKeyboardButton(text="Стиль", callback_data="mode:style"),
                InlineKeyboardButton(text="Очистить фон", callback_data="mode:clean"),
            ],
            [
                InlineKeyboardButton(text="Срисовать", callback_data="mode:redraw"),
            ],
            [
                InlineKeyboardButton(text="Мягче", callback_data="redraw:softer"),
                InlineKeyboardButton(text="Чётче", callback_data="redraw:sharper"),
            ],
            [
                InlineKeyboardButton(text="Плотнее", callback_data="redraw:denser"),
                InlineKeyboardButton(text="Цветов −", callback_data="redraw:colors_down"),
                InlineKeyboardButton(text="Цветов +", callback_data="redraw:colors_up"),
            ],
            [
                InlineKeyboardButton(text="Контур", callback_data="redraw:outline"),
                InlineKeyboardButton(text="Сброс", callback_data="redraw:reset"),
            ],
        ]
    )


def _get_options(user_id: int | None) -> ProcessingOptions:
    if user_id is None:
        return ProcessingOptions()
    return USER_OPTIONS.setdefault(user_id, ProcessingOptions())


def _get_redraw_settings(user_id: int | None) -> RedrawSettings:
    if user_id is None:
        return RedrawSettings()
    return USER_REDRAW_SETTINGS.setdefault(user_id, RedrawSettings())


def _clamp_redraw_settings(settings: RedrawSettings) -> RedrawSettings:
    (
        settings.colors,
        settings.blur,
        settings.sharpen,
        settings.scale,
        settings.outline_thickness,
    ) = clamp_redraw_settings(
        colors=settings.colors,
        blur=settings.blur,
        sharpen=settings.sharpen,
        scale=settings.scale,
        outline_thickness=settings.outline_thickness,
    )
    return settings


def _redraw_status_text(settings: RedrawSettings) -> str:
    outline_state = "включён" if settings.outline else "выключен"
    return (
        "Срисовать:\n"
        f"Цветов: {settings.colors}\n"
        f"Мягкость: {settings.blur:.2f}\n"
        f"Резкость: {settings.sharpen}\n"
        f"Размер: {settings.scale:.2f}\n"
        f"Контур: {outline_state}"
    )


async def _download_message_file(message: Message) -> bytes:
    if message.photo:
        photo = message.photo[-1]
        file = await message.bot.get_file(photo.file_id)
        if not file.file_path:
            raise ImageProcessingError("Failed to download image file.")
        data = await message.bot.download_file(file.file_path)
        return data.read()

    if message.document:
        document = message.document
        if document.mime_type and not any(
            document.mime_type.startswith(prefix) for prefix in SUPPORTED_MIME_PREFIXES
        ):
            raise ImageProcessingError("Неподдерживаемый тип файла. Отправь изображение.")
        file = await message.bot.get_file(document.file_id)
        if not file.file_path:
            raise ImageProcessingError("Failed to download image file.")
        data = await message.bot.download_file(file.file_path)
        return data.read()

    raise ImageProcessingError("Отправь изображение как фото или документ.")


def _build_emoji_from_bytes(
    image_bytes: bytes, options: ProcessingOptions
) -> Image.Image:
    return build_emoji_image(
        image_bytes=image_bytes,
        remove_background=ENABLE_BACKGROUND_REMOVAL,
        mode=options.mode,
        use_face_detection=options.use_face_detection,
        apply_style=options.apply_style,
        with_outline=options.with_outline,
    )


async def _download_file_by_id(message: Message, file_id: str) -> bytes:
    file = await message.bot.get_file(file_id)
    if not file.file_path:
        raise ImageProcessingError("Failed to download image file.")
    data = await message.bot.download_file(file.file_path)
    return data.read()


async def _process_redraw_callback(
    callback: CallbackQuery,
    *,
    options: ProcessingOptions,
    settings: RedrawSettings,
    user_id: int | None,
) -> None:
    if callback.message is None or user_id is None:
        await callback.answer(NO_IMAGE_FOUND_MESSAGE, show_alert=True)
        return

    image_bytes = USER_LAST_ORIGINAL_IMAGES.get(user_id)
    if image_bytes is None:
        file_id = USER_LAST_FILE_IDS.get(user_id)
        if not file_id:
            await callback.answer(NO_IMAGE_FOUND_MESSAGE, show_alert=True)
            return
        image_bytes = await _download_file_by_id(callback.message, file_id)
        USER_LAST_ORIGINAL_IMAGES[user_id] = image_bytes

    processed = build_emoji_image(
        image_bytes=image_bytes,
        remove_background=ENABLE_BACKGROUND_REMOVAL,
        mode=options.mode,
        use_face_detection=options.use_face_detection,
        apply_style=False,
        with_outline=False,
        redraw_mode=True,
        redraw_colors=settings.colors,
        redraw_blur=settings.blur,
        redraw_sharpen=settings.sharpen,
        redraw_scale=settings.scale,
        redraw_outline=settings.outline,
        redraw_outline_thickness=settings.outline_thickness,
    )
    png_bytes, webp_bytes = export_png_webp(processed)
    await callback.message.answer_document(
        BufferedInputFile(png_bytes, filename="emoji.png"),
        caption="Готово 👇",
    )
    await callback.message.answer_document(
        BufferedInputFile(webp_bytes, filename="emoji.webp"),
    )
    await callback.message.answer(_redraw_status_text(settings))
    await callback.answer("Готово 👇")


async def _process_single_image(
    message: Message, image_bytes: bytes, options: ProcessingOptions
) -> None:
    processed = _build_emoji_from_bytes(image_bytes, options)
    png_bytes, webp_bytes = export_png_webp(processed)
    await message.answer_document(
        BufferedInputFile(png_bytes, filename="emoji.png"),
        caption="Готово 👇",
    )
    await message.answer_document(
        BufferedInputFile(webp_bytes, filename="emoji.webp"),
    )


async def _flush_media_group(media_group_id: str) -> None:
    await asyncio.sleep(BATCH_GROUP_WAIT_TIMEOUT)
    items = MEDIA_GROUP_ITEMS.pop(media_group_id, [])
    MEDIA_GROUP_TASKS.pop(media_group_id, None)
    if not items:
        return

    message = items[0][0]
    options = _get_options(message.from_user.id if message.from_user else None)
    processed_images = [_build_emoji_from_bytes(image_bytes, options) for _, image_bytes in items]
    zip_bytes = export_batch_zip(processed_images)
    await message.answer_document(
        BufferedInputFile(zip_bytes, filename="emoji_batch.zip"),
        caption="Готово 👇",
    )


@router.message(Command("start"))
async def start_command(message: Message) -> None:
    await message.answer(
        "Отправь изображение, и я сделаю из него эмодзи 👇",
        reply_markup=_mode_keyboard(),
    )


@router.message(Command("batch"))
async def batch_command(message: Message) -> None:
    await message.answer(
        "Отправь несколько изображений альбомом, и я соберу emoji_batch.zip 👇",
        reply_markup=_mode_keyboard(),
    )


@router.callback_query(F.data.startswith("mode:"))
async def set_mode(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id if callback.from_user else None
    options = _get_options(user_id)
    mode = "normal"
    if callback.data and ":" in callback.data:
        mode = callback.data.split(":", maxsplit=1)[1]
    if mode == "normal":
        options.mode = "FAST"
        options.use_face_detection = False
        options.apply_style = False
    elif mode == "face":
        options.mode = "FAST"
        options.use_face_detection = True
        options.apply_style = False
    elif mode == "style":
        options.mode = "FAST"
        options.use_face_detection = False
        options.apply_style = True
    elif mode == "clean":
        options.mode = "CLEAN"
        options.use_face_detection = False
        options.apply_style = False
    elif mode == "redraw":
        settings = _clamp_redraw_settings(_get_redraw_settings(user_id))
        await _process_redraw_callback(
            callback,
            options=options,
            settings=settings,
            user_id=user_id,
        )
        return

    await callback.answer("Режим обновлён")
    if callback.message:
        await callback.message.answer("Режим выбран. Отправь изображение или альбом 👇")


@router.message(F.photo | F.document)
async def process_image(message: Message) -> None:
    try:
        image_bytes = await _download_message_file(message)
        if message.from_user:
            USER_LAST_ORIGINAL_IMAGES[message.from_user.id] = image_bytes
            if message.photo:
                USER_LAST_FILE_IDS[message.from_user.id] = message.photo[-1].file_id
            elif message.document:
                USER_LAST_FILE_IDS[message.from_user.id] = message.document.file_id
        options = _get_options(message.from_user.id if message.from_user else None)
        if message.media_group_id:
            group_id = str(message.media_group_id)
            MEDIA_GROUP_ITEMS.setdefault(group_id, []).append((message, image_bytes))
            if group_id not in MEDIA_GROUP_TASKS:
                MEDIA_GROUP_TASKS[group_id] = asyncio.create_task(_flush_media_group(group_id))
            return

        await _process_single_image(message, image_bytes, options)
    except ImageProcessingError as exc:
        await message.answer(str(exc))
        return
    except Exception:
        logger.exception("Unexpected error while processing uploaded image")
        await message.answer("Не удалось обработать изображение. Попробуй другой файл.")
        return


@router.callback_query(F.data.startswith("redraw:"))
async def tune_redraw(callback: CallbackQuery) -> None:
    user_id = callback.from_user.id if callback.from_user else None
    settings = _clamp_redraw_settings(_get_redraw_settings(user_id))
    options = _get_options(user_id)

    action = "apply"
    if callback.data and ":" in callback.data:
        action = callback.data.split(":", maxsplit=1)[1]

    if action == "softer":
        settings.blur += REDRAW_BLUR_STEP
        settings.sharpen -= REDRAW_SHARPEN_STEP
    elif action == "sharper":
        settings.blur -= REDRAW_BLUR_STEP
        settings.sharpen += REDRAW_SHARPEN_STEP
    elif action == "denser":
        settings.scale += REDRAW_SCALE_STEP
    elif action == "colors_down":
        settings.colors -= REDRAW_COLORS_STEP
    elif action == "colors_up":
        settings.colors += REDRAW_COLORS_STEP
    elif action == "outline":
        settings.outline = not settings.outline
    elif action == "reset":
        settings = RedrawSettings()
        if user_id is not None:
            USER_REDRAW_SETTINGS[user_id] = settings

    settings = _clamp_redraw_settings(settings)
    await _process_redraw_callback(
        callback,
        options=options,
        settings=settings,
        user_id=user_id,
    )


@router.message()
async def help_message(message: Message) -> None:
    await message.answer(
        "Отправь изображение, и я сделаю из него эмодзи 👇\n"
        "Один файл: emoji.png + emoji.webp\n"
        "Альбом (/batch): emoji_batch.zip",
        reply_markup=_mode_keyboard(),
    )
