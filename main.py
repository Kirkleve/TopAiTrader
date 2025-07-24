
import threading

from bot.order_notifier import OrderNotifier
from bot.telegram_bot import TelegramBot
from config import TELEGRAM_CHAT_ID


def send_welcome(bot):
    commands = [
        "/predict — 📈 Прогноз цены BTC (LSTM, NeuralProphet, XGB, PPO)",
        "/accuracy — 📊 Проверить точность текущей стратегии",
        "/sentiment — 🗞️ Сантимент рынка (новости + Fear & Greed)",
        "/market — 📌 Подробный обзор рынка и трендов",
        "/topnews — 🚀 ТОП новости крипторынка",
        "/manage — 💎 Управление монетами",
        "/autotrade — 🤖 Запустить автотрейдинг",
        "/help — 📖 Показать список команд"
    ]

    welcome_text = (
        "🚀 <b>Торговый Telegram-бот успешно запущен!</b>\n\n"
        "✨ Доступные команды:\n" + "\n".join(commands)
    )

    bot.bot.send_message(TELEGRAM_CHAT_ID, welcome_text, parse_mode="HTML")


def main():
    bot = TelegramBot()

    # Запуск мониторинга позиций в отдельном потоке
    notifier = OrderNotifier(bot, bot.trader)
    threading.Thread(target=notifier.start_monitoring, daemon=True).start()

    # Отправка приветственного сообщения
    send_welcome(bot)

    # Запуск бота (polling)
    print("🤖 Telegram бот успешно запущен!")
    bot.bot.infinity_polling()

if __name__ == '__main__':
    print("🚀 Запуск торгового Telegram-бота...")
    main()
