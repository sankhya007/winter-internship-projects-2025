
# wake_engine.py
import speech_recognition as sr

class WakeWordEngine:
    def __init__(self, wake_word='hey nova'):
        self.wake_word = wake_word.lower()
        self.r = sr.Recognizer()
        self.mic = None
        try:
            self.mic = sr.Microphone()
        except Exception as e:
            print('[WAKE ENGINE] Microphone init error:', e)

    def listen_for_wake_word(self):
        if not self.mic:
            # fallback to typed wake-word
            typed = input("Type wake word to activate: ").strip().lower()
            return typed == self.wake_word

        with self.mic as source:
            print('🎤 Wake-word engine: adjusting for ambient noise...')
            self.r.adjust_for_ambient_noise(source, duration=0.6)

        print(f"🎤 Wake-word engine running... say '{self.wake_word}'")

        while True:
            try:
                with self.mic as source:
                    audio = self.r.listen(source, timeout=None, phrase_time_limit=3)
                try:
                    text = self.r.recognize_google(audio).lower()
                    print('Heard:', text)
                    if self.wake_word in text:
                        print('✨ Wake word detected!')
                        return True
                except sr.UnknownValueError:
                    continue
                except sr.RequestError:
                    # API problems — fallback to typed wake word
                    typed = input("Speech API error. Type wake word to activate: ").strip().lower()
                    if typed == self.wake_word:
                        return True
            except Exception as e:
                print('[WAKE ENGINE] Listening error:', e)
                typed = input("Type wake word to activate: ").strip().lower()
                if typed == self.wake_word:
                    return True