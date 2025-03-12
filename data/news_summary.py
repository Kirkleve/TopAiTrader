class NewsSummarizer:
    def __init__(self, translator, summarizer):
        self.translator = translator
        self.summarizer = summarizer

    def summarize_and_translate(self, text):
        if len(text.split()) <= 30:
            return self.translator(text)[0]['translation_text']

        max_len = min(50, len(text.split()) // 2)
        summary_en = self.summarizer(
            text, max_length=max_len, min_length=5, do_sample=False
        )[0]['summary_text']

        return self.translator(summary_en)[0]['translation_text']

