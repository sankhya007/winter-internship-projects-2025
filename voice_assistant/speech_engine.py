# speech_engine.py
import subprocess

class SpeechEngine:
    def __init__(self):
        pass

    # ----------------------------------
    # SPEAK
    # ----------------------------------
    def speak_sync(self, text):
        safe = text.replace('"', '`"')
        ps = (
            'Add-Type -AssemblyName System.speech;'
            '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;'
            '$s.Rate = 0; $s.Volume = 100;'
            f'$s.Speak("{safe}")'
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except Exception as e:
            print("TTS Error:", e)
            print("Assistant:", text)

    # ----------------------------------
    # LISTEN WITH MIC + KEYBOARD FALLBACK
    # ----------------------------------
    def listen(self, timeout=6, phrase_time_limit=4):
        """
        Order:
        1. Try microphone
        2. If error → fallback to keyboard input
        """

        # Try speech_recognition
        try:
            import speech_recognition as sr
            r = sr.Recognizer()

            # Try opening microphone
            try:
                with sr.Microphone() as source:
                    print("🎙️ Adjusting noise...")
                    r.adjust_for_ambient_noise(source, duration=0.4)

                    print("🎙️ Listening...")
                    audio = r.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            except Exception as mic_error:
                print(f"[MIC ERROR] {mic_error}")
                return self._fallback()

            # Convert audio → text
            try:
                text = r.recognize_google(audio)
                return text

            except sr.UnknownValueError:
                print("❌ I couldn't understand that.")
                return None

            except sr.RequestError as api_error:
                print(f"[SPEECH API ERROR] {api_error}")
                return self._fallback()

        except Exception as e:
            print(f"[SPEECH_RECOGNITION ERROR] {e}")
            return self._fallback()

    # ----------------------------------
    # KEYBOARD FALLBACK
    # ----------------------------------
    def _fallback(self):
        print("⌨️ No mic available — type instead:")
        try:
            return input("> ").strip()
        except:
            return None
