# languages.py
from googletrans import Translator

translator = Translator()

def translate_text(text, lang_code):
    try:
        translated = translator.translate(text, dest=lang_code)
        return translated.text
    except Exception as e:
        return f"[Translation error: {e}]"
