from __future__ import annotations

import logging
import os

from aiogram import F, Router
from aiogram.types import BufferedInputFile, Message

from image_processing import ImageProcessingError, build_emoji_image, export_png_webp

router = Router()
logger = logging.getLogger(__name__)

SUPPORTED_MIME_PREFIXES = ("image/",)
ENABLE_BACKGROUND_REMOVAL = os.getenv("ENABLE_BACKGROUND_REMOVAL", "0") == "1"


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


@router.message(F.photo | F.document)
async def process_image(message: Message) -> None:
    try:
        image_bytes = await _download_message_file(message)
        processed = build_emoji_image(
            image_bytes=image_bytes,
            remove_background=ENABLE_BACKGROUND_REMOVAL,
        )
        png_bytes, webp_bytes = export_png_webp(processed)
    except ImageProcessingError as exc:
        await message.answer(str(exc))
        return
    except Exception:
        logger.exception("Unexpected error while processing uploaded image")
        await message.answer("Failed to process image. Please try another file.")
        return

    await message.answer_document(
        BufferedInputFile(png_bytes, filename="emoji.png"),
        caption="Done! Here is your Telegram-compatible static emoji.",
    )
    await message.answer_document(
        BufferedInputFile(webp_bytes, filename="emoji.webp"),
    )


@router.message()
async def help_message(message: Message) -> None:
    await message.answer("Send an image (photo or file), and I will convert it to emoji.png and emoji.webp.")
