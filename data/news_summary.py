class NewsSummarizer:
    def __init__(self, translator, summarizer):
        self.translator = translator
        self.summarizer = summarizer

    def summarize_and_translate(self, text):
        try:
            # Если текст короткий — просто переводим
            if len(text.split()) <= 30:
                translation = self.translator(text)[0]['translation_text']
                return translation

            # Если текст длинный — делаем краткое резюме
            max_len = min(50, len(text.split()) // 2)
            summary_en = self.summarizer(
                text, max_length=max_len, min_length=5, do_sample=False
            )[0]['summary_text']

            # Переводим резюме на русский
            translation = self.translator(summary_en)[0]['translation_text']
            return translation

        except Exception as e:
            print(f"⚠️ Ошибка при суммаризации и переводе: {e}")
            return "Не удалось подготовить краткое описание новости."
