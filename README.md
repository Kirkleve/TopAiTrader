# 🚀 AI Crypto Trading Bot

🤖 **Интеллектуальный торговый бот**, который использует мощную комбинацию моделей **LSTM и DQN** для эффективного трейдинга криптовалют на бирже Binance (Futures). Включает мульти-таймфреймный анализ, sentiment-анализ новостей, а также данные Fear & Greed Index для принятия максимально точных торговых решений.

---

## 📌 Основные возможности:
- 📈 **LSTM-модель** для точного прогнозирования цены.
- 🧠 **DQN-агент** для оптимизации торговых решений.
- 📊 Адаптивная стратегия торговли на основе метрик.
- 📅 Использование нескольких таймфреймов (15m, 1h, 4h, 1d).
- 📰 Анализ новостного фона и социального Sentiment анализа.
- 🔥 Индекс страха и жадности (Fear & Greed Index).

---

## 🛠 Стек технологий:

- Python
- PyTorch
- Gymnasium
- ccxt (Binance Futures API)
- Telebot (Telegram Bot API)
- Pandas, NumPy, Scikit-learn
- Transformers (для анализа sentiment и summarization)

---

## 🔧 Как запустить проект:

1. Установить зависимости:
```bash
pip install -r requirements.txt
```

2. Настроить конфигурационный файл:
```python
# config.py
BINANCE_TESTNET_API_KEY = "YOUR_KEY"
BINANCE_TESTNET_API_SECRET = "your_secret"
TELEGRAM_TOKEN = 'your_telegram_bot_token'
TELEGRAM_CHAT_ID = 'your_chat_id'
```

3. Запустить обучение моделей:
```bash
python main.py
```

---

## 🤝 Поддерживаемые команды Telegram бота:

- `/manage` — Управление списком монет для торговли
- `/predict` — Получение прогноза по BTC
- `/accuracy` — Проверка точности модели
- `/sentiment` — Анализ новостей
- `/market` — Обзор текущего рынка
- `/topnews` — важные крипто-новости
- `/autotrade` — запустить автоматическую торговлю
- `/stop` — остановить торговлю

---

⭐️ **Успешных торгов и профита!**

