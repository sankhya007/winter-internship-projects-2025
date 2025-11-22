

# speech_engine.py
import subprocess
import sys

class SpeechEngine:
    def __init__(self):
        pass

    def speak_sync(self, text):
        safe = str(text).replace('"', '`"')
        ps = (
            'Add-Type -AssemblyName System.speech;'
            '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;'
            '$s.Rate = 0; $s.Volume = 100;'
            f'$s.Speak("{safe}")'
        )
        try:
            subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=False)
        except Exception as e:
            print("[TTS ERROR]", e)
            print("Assistant:", text)

    def listen(self, timeout=6, phrase_time_limit=5):
        try:
            import speech_recognition as sr
            r = sr.Recognizer()
            try:
                with sr.Microphone() as source:
                    print("🎙️ Adjusting for ambient noise...")
                    r.adjust_for_ambient_noise(source, duration=0.4)
                    print("🎙️ Listening...")
                    audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)
            except Exception as mic_err:
                print(f"[MIC ERROR] {mic_err}")
                return self._fallback()

            try:
                text = r.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                print("❌ I couldn't understand that.")
                return None
            except sr.RequestError as api_err:
                print(f"[SPEECH API ERROR] {api_err}")
                return self._fallback()

        except Exception as e:
            print(f"[SPEECH_RECOGNITION ERROR] {e}")
            return self._fallback()

    def _fallback(self):
        try:
            return input("⌨️ Type instead: ").strip()
        except Exception:
            return None