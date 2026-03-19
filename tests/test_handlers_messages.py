import asyncio
import unittest
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from PIL import Image

from handlers import messages


def _image_to_bytes(image: Image.Image, fmt: str = "PNG") -> bytes:
    buf = BytesIO()
    image.save(buf, format=fmt)
    return buf.getvalue()


class TestMessagesHandlers(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        messages.USER_OPTIONS.clear()
        messages.USER_REDRAW_SETTINGS.clear()
        messages.USER_LAST_FILE_IDS.clear()
        messages.USER_LAST_ORIGINAL_IMAGES.clear()
        messages.MEDIA_GROUP_ITEMS.clear()
        for task in messages.MEDIA_GROUP_TASKS.values():
            task.cancel()
        messages.MEDIA_GROUP_TASKS.clear()

    async def test_redraw_callback_uses_stored_last_file_reference(self) -> None:
        user_id = 123
        file_id = "file-1"
        messages.USER_LAST_FILE_IDS[user_id] = file_id

        src = Image.new("RGBA", (120, 120), (0, 0, 0, 0))
        for x in range(20, 100):
            for y in range(20, 100):
                src.putpixel((x, y), (100, 140, 200, 255))
        image_bytes = _image_to_bytes(src)

        bot = SimpleNamespace()
        bot.get_file = AsyncMock(return_value=SimpleNamespace(file_path="abc/path.png"))
        bot.download_file = AsyncMock(return_value=BytesIO(image_bytes))

        message = SimpleNamespace()
        message.bot = bot
        message.answer_document = AsyncMock()
        message.answer = AsyncMock()

        callback = SimpleNamespace(
            from_user=SimpleNamespace(id=user_id),
            data="mode:redraw",
            message=message,
            answer=AsyncMock(),
        )

        with patch("handlers.messages.build_emoji_image", wraps=messages.build_emoji_image) as mocked:
            await messages.set_mode(callback)
        self.assertTrue(mocked.called)
        self.assertTrue(mocked.call_args.kwargs.get("redraw_mode"))
        self.assertEqual(mocked.call_args.kwargs.get("image_bytes"), image_bytes)
        self.assertEqual(message.answer_document.await_count, 2)
        callback.answer.assert_awaited()

    async def test_redraw_tune_updates_only_current_user_settings(self) -> None:
        user_a = 101
        user_b = 202
        payload = b"image-bytes"
        messages.USER_LAST_ORIGINAL_IMAGES[user_a] = payload
        messages.USER_LAST_ORIGINAL_IMAGES[user_b] = payload

        message = SimpleNamespace()
        message.answer_document = AsyncMock()
        message.answer = AsyncMock()

        callback = SimpleNamespace(
            from_user=SimpleNamespace(id=user_a),
            data="redraw:colors_up",
            message=message,
            answer=AsyncMock(),
        )

        with patch("handlers.messages.build_emoji_image") as mocked_build, patch(
            "handlers.messages.export_png_webp", return_value=(b"png", b"webp")
        ):
            await messages.tune_redraw(callback)

        settings_a = messages.USER_REDRAW_SETTINGS[user_a]
        settings_b = messages.USER_REDRAW_SETTINGS.get(user_b)
        self.assertEqual(settings_a.colors, 72)
        self.assertIsNone(settings_b)
        self.assertEqual(mocked_build.call_args.kwargs["redraw_colors"], 72)
        callback.answer.assert_awaited()

    async def test_redraw_tune_reuses_cached_original_bytes(self) -> None:
        user_id = 303
        original_bytes = b"original-image"
        messages.USER_LAST_ORIGINAL_IMAGES[user_id] = original_bytes
        messages.USER_LAST_FILE_IDS[user_id] = "file-x"

        bot = SimpleNamespace()
        bot.get_file = AsyncMock()
        bot.download_file = AsyncMock()

        message = SimpleNamespace()
        message.bot = bot
        message.answer_document = AsyncMock()
        message.answer = AsyncMock()

        callback = SimpleNamespace(
            from_user=SimpleNamespace(id=user_id),
            data="redraw:sharper",
            message=message,
            answer=AsyncMock(),
        )

        with patch("handlers.messages.build_emoji_image") as mocked_build, patch(
            "handlers.messages.export_png_webp", return_value=(b"png", b"webp")
        ):
            await messages.tune_redraw(callback)

        self.assertEqual(mocked_build.call_args.kwargs["image_bytes"], original_bytes)
        bot.get_file.assert_not_called()
        bot.download_file.assert_not_called()

    def test_mode_keyboard_texts_and_callback_data(self) -> None:
        keyboard = messages._mode_keyboard()
        buttons = [button for row in keyboard.inline_keyboard for button in row]
        text_to_callback = {button.text: button.callback_data for button in buttons}

        self.assertEqual(text_to_callback["Обычный"], "mode:normal")
        self.assertEqual(text_to_callback["По лицу"], "mode:face")
        self.assertEqual(text_to_callback["Стиль"], "mode:style")
        self.assertEqual(text_to_callback["Очистить фон"], "mode:clean")
        self.assertEqual(text_to_callback["Срисовать"], "mode:redraw")
        self.assertEqual(text_to_callback["Мягче"], "redraw:softer")
        self.assertEqual(text_to_callback["Чётче"], "redraw:sharper")
        self.assertEqual(text_to_callback["Плотнее"], "redraw:denser")
        self.assertEqual(text_to_callback["Цветов −"], "redraw:colors_down")
        self.assertEqual(text_to_callback["Цветов +"], "redraw:colors_up")
        self.assertEqual(text_to_callback["Контур"], "redraw:outline")
        self.assertEqual(text_to_callback["Сброс"], "redraw:reset")

    def test_redraw_status_text_is_russian_human_readable(self) -> None:
        status = messages._redraw_status_text(
            messages.RedrawSettings(
                colors=64,
                blur=0.35,
                sharpen=105,
                scale=0.92,
                outline=True,
            )
        )
        self.assertEqual(
            status,
            "Срисовать:\n"
            "Цветов: 64\n"
            "Мягкость: 0.35\n"
            "Резкость: 105\n"
            "Размер: 0.92\n"
            "Контур: включён",
        )


if __name__ == "__main__":
    unittest.main()
