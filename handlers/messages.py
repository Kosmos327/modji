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
    build_emoji_image,
    export_batch_zip,
    export_png_webp,
)

router = Router()
logger = logging.getLogger(__name__)

SUPPORTED_MIME_PREFIXES = ("image/",)
ENABLE_BACKGROUND_REMOVAL = os.getenv("ENABLE_BACKGROUND_REMOVAL", "0") == "1"
BATCH_GROUP_WAIT_TIMEOUT = 1.2


@dataclass
class ProcessingOptions:
    mode: str = "FAST"
    use_face_detection: bool = False
    apply_style: bool = False
    with_outline: bool = False


USER_OPTIONS: dict[int, ProcessingOptions] = {}
MEDIA_GROUP_ITEMS: dict[str, list[tuple[Message, bytes]]] = {}
MEDIA_GROUP_TASKS: dict[str, asyncio.Task[Any]] = {}


def _mode_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [
                InlineKeyboardButton(text="Normal", callback_data="mode:normal"),
                InlineKeyboardButton(text="Face mode", callback_data="mode:face"),
            ],
            [
                InlineKeyboardButton(text="Style mode", callback_data="mode:style"),
                InlineKeyboardButton(text="Clean mode", callback_data="mode:clean"),
            ],
        ]
    )


def _get_options(user_id: int | None) -> ProcessingOptions:
    if user_id is None:
        return ProcessingOptions()
    return USER_OPTIONS.setdefault(user_id, ProcessingOptions())


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
            raise ImageProcessingError("Unsupported file type. Please send an image.")
        file = await message.bot.get_file(document.file_id)
        if not file.file_path:
            raise ImageProcessingError("Failed to download image file.")
        data = await message.bot.download_file(file.file_path)
        return data.read()

    raise ImageProcessingError("Please send an image as photo or document.")


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


async def _process_single_image(
    message: Message, image_bytes: bytes, options: ProcessingOptions
) -> None:
    processed = _build_emoji_from_bytes(image_bytes, options)
    png_bytes, webp_bytes = export_png_webp(processed)
    await message.answer_document(
        BufferedInputFile(png_bytes, filename="emoji.png"),
        caption="Done! Here is your Telegram-compatible static emoji.",
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
        caption="Done! Batch archive is ready.",
    )


@router.message(Command("start"))
async def start_command(message: Message) -> None:
    await message.answer(
        "Hi! Send an image (or album) and I will convert it to emoji.\n"
        "Use the buttons below to switch processing mode.",
        reply_markup=_mode_keyboard(),
    )


@router.message(Command("batch"))
async def batch_command(message: Message) -> None:
    await message.answer(
        "Send multiple images as a media group/album and I will return emoji_batch.zip.",
        reply_markup=_mode_keyboard(),
    )


@router.callback_query(F.data.startswith("mode:"))
async def set_mode(callback: CallbackQuery) -> None:
    options = _get_options(callback.from_user.id if callback.from_user else None)
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

    await callback.answer("Mode updated")
    if callback.message:
        await callback.message.answer("Mode set. Send an image or album.")


@router.message(F.photo | F.document)
async def process_image(message: Message) -> None:
    try:
        image_bytes = await _download_message_file(message)
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
        await message.answer("Failed to process image. Please try another file.")
        return


@router.message()
async def help_message(message: Message) -> None:
    await message.answer(
        "Send an image and I will convert it to emoji.\n"
        "Single image: emoji.png + emoji.webp\n"
        "Album (/batch): emoji_batch.zip",
        reply_markup=_mode_keyboard(),
    )
