# assistant.py

import re
import sys
import time
from datetime import datetime, timedelta

from speech_engine import SpeechEngine
from task_manager import TaskManager
from wake_engine import WakeWordEngine
from ai_engine import AIEngine
from config import ASSISTANT_NAME, WAKE_WORD


# -----------------------------------------------------
# Minimal Typewriter Effect
# -----------------------------------------------------
def type_out(text, delay=0.01):
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()


# -----------------------------------------------------
# Time Parsing without external libraries
# -----------------------------------------------------
def parse_time_expression(text):
    text = text.lower()

    # in X minutes
    m = re.search(r"in (\d+) (minute|minutes)", text)
    if m:
        return datetime.now() + timedelta(minutes=int(m.group(1)))

    # in X hours
    m = re.search(r"in (\d+) (hour|hours)", text)
    if m:
        return datetime.now() + timedelta(hours=int(m.group(1)))

    # at HH:MM
    m = re.search(r"at (\d{1,2}):(\d{2})", text)
    if m:
        now = datetime.now()
        hour = int(m.group(1))
        minute = int(m.group(2))
        dt = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if dt < now:
            dt += timedelta(days=1)
        return dt

    # at 5pm / 5 am
    m = re.search(r"at (\d{1,2})\s*(am|pm)", text)
    if m:
        hour = int(m.group(1))
        ap = m.group(2)
        if ap == "pm" and hour != 12:
            hour += 12
        if ap == "am" and hour == 12:
            hour = 0
        now = datetime.now()
        dt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if dt < now:
            dt += timedelta(days=1)
        return dt

    return None


# -----------------------------------------------------
# Voice Assistant Class
# -----------------------------------------------------
class VoiceAssistant:
    def __init__(self):
        self.name = ASSISTANT_NAME
        self.wake_word = WAKE_WORD.lower()

        self.speech = SpeechEngine()
        self.task = TaskManager()
        self.ai = AIEngine()

        # wake-word engine
        self.wake_engine = WakeWordEngine(self.wake_word)

    # -------------------------------------------------
    def startup(self):
        print("\n================ VOICE ASSISTANT STARTED ================\n")

        intro = (
            f"{self.name} (wake word: '{self.wake_word}')\n"
            "Capabilities:\n"
            " - Tell time/date\n"
            " - Weather (e.g., weather in Mumbai)\n"
            " - News\n"
            " - AI Chat / Questions\n"
            " - Web search\n"
            " - Reminders:\n"
            "     → remind me to drink water in 5 minutes\n"
            "     → remind me to study at 6pm\n"
        )

        type_out(intro)
        try:
            self.speech.speak_sync(f"{self.name} online. Say '{self.wake_word}' to wake me.")
        except Exception:
            # speak_sync already prints errors; continue anyway
            pass

    # -------------------------------------------------
    # Command Handler
    # -------------------------------------------------
    def handle_command(self, cmd):
        if not cmd:
            return "I didn't catch that."

        cmd = cmd.lower()

        # TIME
        if "time" in cmd:
            return self.task.tell_time()

        # DATE
        if "date" in cmd:
            return self.task.tell_date()

        # WEATHER
        if "weather" in cmd:
            m = re.search(r"weather in (.+)", cmd)
            if m:
                return self.task.check_weather(m.group(1).strip())
            return self.task.check_weather("London")

        # NEWS
        if "news" in cmd:
            return self.task.get_news()

        # SEARCH
        if "search for" in cmd:
            q = cmd.split("search for", 1)[1].strip()
            return self.task.search_web(q)

        # REMIND ME
        if "remind me" in cmd:
            dt = parse_time_expression(cmd)
            if dt is None:
                return "I couldn't understand the time format."

            m = re.search(r"remind me to (.+?) (in|at)", cmd)
            task_text = m.group(1).strip() if m else re.sub(r"remind me (to|about)\s*", "", cmd).strip()

            return self.task.set_reminder(task_text, dt)

        # GOODBYE
        if "goodbye" in cmd or "exit" in cmd or "quit" in cmd:
            return "Goodbye!"

        # DEFAULT: SEND TO AI
        return self.ai.ask(cmd)

    # -------------------------------------------------
    # Main Loop (Wake Word → Command)
    # -------------------------------------------------
    def run(self):
        self.startup()

        try:
            while True:
                # Wake-word detection (returns True/False)
                print("\n🎤 Waiting for wake word...")
                activated = self.wake_engine.listen_for_wake_word()
                if not activated:
                    # if listening fallback returned False, continue waiting
                    continue

                # Wake-word triggered
                prompt = f"{self.name} activated. What can I do?"
                type_out(prompt)
                self.speech.speak_sync(prompt)

                # Listen to user command
                cmd = self.speech.listen()
                if not cmd:
                    type_out("No command received.")
                    continue

                type_out(f"You said: {cmd}")
                response = self.handle_command(cmd)

                # Output
                type_out(response)
                self.speech.speak_sync(response)

                if "goodbye" in response.lower():
                    break

                # Check reminders (quick check after each command)
                for r in self.task.pending():
                    msg = f"Reminder: {r['text']}"
                    type_out(msg)
                    self.speech.speak_sync(msg)
                    r["done"] = True
                self.task.save()

        except KeyboardInterrupt:
            print("\nExiting.")
            self.speech.speak_sync("Goodbye.")
            return


if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.run()
