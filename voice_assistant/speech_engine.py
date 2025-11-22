# speech_engine.py
import subprocess

class SpeechEngine:
    def speak_sync(self, text):
        safe = text.replace('"', '`"')
        ps = (
            'Add-Type -AssemblyName System.speech;'
            '$s = New-Object System.Speech.Synthesis.SpeechSynthesizer;'
            f'$s.Speak("{safe}")'
        )
        try:
            subprocess.run(
                ["powershell", "-NoProfile", "-Command", ps],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
        except:
            print("Assistant:", text)

    def listen(self):
        """
        MIC removed because it was causing all issues.
        Now always uses keyboard input.
        """
        try:
            return input("You: ").strip()
        except:
            return None
