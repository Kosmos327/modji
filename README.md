# modji

Telegram bot for converting images into Telegram-compatible emoji assets.

## Features

- 100x100 RGBA emoji output with transparency preserved
- Optional FAST/CLEAN processing modes
- Optional face-centered crop (OpenCV Haar cascade)
- Optional emoji style enhancement (contrast/saturation/smoothing)
- Batch processing for media groups with ZIP export

## Bot commands

- `/start` — show quick help and mode buttons
- `/batch` — explain album upload + ZIP export flow

## Modes via inline buttons

- Обычный
- По лицу
- Стиль
- Очистить фон
- Срисовать
- Настройка срисовки: Мягче, Чётче, Плотнее, Цветов −, Цветов +, Контур, Сброс
