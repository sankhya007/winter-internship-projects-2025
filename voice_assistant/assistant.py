

# assistant.py
import sys
import time
from datetime import datetime, timedelta

from speech_engine import SpeechEngine
from task_manager import TaskManager
from wake_engine import WakeWordEngine
from config import ASSISTANT_NAME, WAKE_WORD


def type_out(text, delay=0.01):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print()


class VoiceAssistant:
    def __init__(self):
        self.name = ASSISTANT_NAME
        self.wake_word = WAKE_WORD.lower()
        self.speech = SpeechEngine()
        self.task = TaskManager()
        self.wake_engine = WakeWordEngine(wake_word=self.wake_word)

    def greet(self):
        print("\n================ VOICE ASSISTANT STARTED ================\n")
        intro = (
            f"{self.name} (wake word: '{self.wake_word}')\n"
            "Capabilities:\n"
            " - Tell time\n"
            " - Tell date\n"
            " - Check weather (e.g., weather in Mumbai)\n"
            " - Read news\n"
            " - Set reminders (use: 'remind me to <task> in X minutes' or 'at HH:MM')\n"
            " - Search the web\n"
            " - Exit by saying: goodbye\n"
        )
        type_out(intro)
        self.speech.speak_sync(f"{self.name} ready. Say '{self.wake_word}' to activate me.")

    def handle_command(self, cmd):
        cmd = cmd.lower()

        if "time" in cmd:
            return self.task.tell_time()

        if "date" in cmd:
            return self.task.tell_date()

        if "weather" in cmd:
            m = None
            if "weather in" in cmd:
                try:
                    m = cmd.split("weather in",1)[1].strip()
                except Exception:
                    m = None
            return self.task.check_weather(m if m else "London")

        if "news" in cmd:
            return self.task.get_news()

        if "search for" in cmd:
            q = cmd.split("search for", 1)[1].strip()
            return self.task.search_web(q)

        if "remind me" in cmd:
            # simple parsing: support 'in X minutes/hours' and 'at HH:MM'
            time_dt = None
            # in X minutes/hours
            import re
            m = re.search(r"in (\d+) (minute|minutes|hour|hours)", cmd)
            if m:
                num = int(m.group(1))
                unit = m.group(2)
                if 'hour' in unit:
                    time_dt = datetime.now() + timedelta(hours=num)
                else:
                    time_dt = datetime.now() + timedelta(minutes=num)
            else:
                m2 = re.search(r"at (\d{1,2}):(\d{2})", cmd)
                if m2:
                    hr = int(m2.group(1)); mn = int(m2.group(2))
                    now = datetime.now()
                    dt = now.replace(hour=hr, minute=mn, second=0, microsecond=0)
                    if dt < now:
                        dt += timedelta(days=1)
                    time_dt = dt

            if not time_dt:
                return "I couldn't understand the time. Use: 'in 5 minutes' or 'at 18:30'."

            # extract the task text
            tmatch = re.search(r"remind me to (.+?)(?: in | at |$)", cmd)
            task_text = tmatch.group(1).strip() if tmatch else "your reminder"
            return self.task.set_reminder(task_text, time_dt)

        if "goodbye" in cmd or "exit" in cmd:
            return "Goodbye!"

        return "I didn't understand that."

    def run(self):
        self.greet()

        try:
            while True:
                # BLOCK until wake word is detected
                print("\n🎧 Waiting for wake word... (say the wake phrase)")
                self.wake_engine.listen_for_wake_word()

                # Activated
                prompt = f"{self.name} activated. What can I do?"
                type_out(prompt)
                self.speech.speak_sync(prompt)

                # Listen for the actual command
                cmd = self.speech.listen()
                if not cmd:
                    continue

                type_out(f"You said: {cmd}")
                response = self.handle_command(cmd)

                type_out(response)
                self.speech.speak_sync(response)

                if "goodbye" in response.lower():
                    break

                # Check reminders
                for r in self.task.pending():
                    msg = f"Reminder: {r['text']}"
                    type_out(msg)
                    self.speech.speak_sync(msg)
                    r['done'] = True
                    self.task.save()

        except KeyboardInterrupt:
            print("\nExiting.")
            self.speech.speak_sync("Goodbye.")


if __name__ == "__main__":
    VoiceAssistant().run()