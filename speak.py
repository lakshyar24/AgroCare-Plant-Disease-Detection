from gtts import gTTS
import os
import uuid
import playsound
import tempfile

def speak_text(text, lang='en'):
    try:
        # Create a unique file path
        temp_dir = tempfile.gettempdir()
        filename = os.path.join(temp_dir, f"{uuid.uuid4().hex}.mp3")

        # Save the speech audio
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)

        # Play the audio
        playsound.playsound(filename)

        # Delete the file after playing
        os.remove(filename)

    except Exception as e:
        print("❌ TTS failed:", e)
