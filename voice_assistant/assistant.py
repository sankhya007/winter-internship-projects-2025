# assistant.py
import sys
import time
from datetime import datetime, timedelta

from speech_engine import SpeechEngine
from task_manager import TaskManager
from config import ASSISTANT_NAME, WAKE_WORD


def type_out(text, delay=0.01):
    for c in text:
        sys.stdout.write(c)
        sys.stdout.flush()
        time.sleep(delay)
    print()


def parse_time(cmd):
    cmd = cmd.lower()

    if "in" in cmd and "minute" in cmd:
        num = int(cmd.split("in")[1].split("minute")[0].strip())
        return datetime.now() + timedelta(minutes=num)

    if "in" in cmd and "hour" in cmd:
        num = int(cmd.split("in")[1].split("hour")[0].strip())
        return datetime.now() + timedelta(hours=num)

    if "at" in cmd and ":" in cmd:
        t = cmd.split("at")[1].strip()
        hr, mn = t.split(":")
        now = datetime.now()
        dt = now.replace(hour=int(hr), minute=int(mn), second=0)
        if dt < now:
            dt += timedelta(days=1)
        return dt

    return None


class Assistant:
    def __init__(self):
        self.speech = SpeechEngine()
        self.task = TaskManager()
        self.name = ASSISTANT_NAME
        self.wake = WAKE_WORD

    def greet(self):
        print("\n================ VOICE ASSISTANT STARTED ================\n")
        intro = (
            f"{self.name} (wake word: '{self.wake}')\n"
            "Capabilities:\n"
            " - Tell time\n"
            " - Tell date\n"
            " - Get weather\n"
            " - Read news\n"
            " - Set reminders\n"
            " - Web search\n"
            " - Type 'exit' to stop\n"
        )
        type_out(intro)
        self.speech.speak_sync(f"{self.name} ready. Say {self.wake} to activate me.")

    def handle(self, cmd):
        cmd = cmd.lower()

        if "time" in cmd:
            return self.task.tell_time()

        if "date" in cmd:
            return self.task.tell_date()

        if "weather" in cmd:
            return self.task.check_weather("India")

        if "news" in cmd:
            return self.task.get_news()

        if "search" in cmd:
            q = cmd.replace("search", "").strip()
            return self.task.search_web(q)

        if "remind me" in cmd:
            when = parse_time(cmd)
            if not when:
                return "Invalid time format."
            msg = cmd.split("remind me to")[-1].strip()
            return self.task.set_reminder(msg, when)

        if "goodbye" in cmd or "exit" in cmd:
            return "Goodbye!"

        return "I didn't understand that."

    def run(self):
        self.greet()

        while True:
            heard = self.speech.listen()
            if not heard:
                continue

            type_out(f"You said: {heard}")

            if self.wake in heard.lower():
                self.speech.speak_sync("Yes?")
                cmd = self.speech.listen()
                if not cmd:
                    continue

                type_out(f"You said: {cmd}")
                res = self.handle(cmd)
                type_out(res)
                self.speech.speak_sync(res)

                if "goodbye" in res.lower():
                    break

            # check reminders
            for r in self.task.pending():
                msg = f"Reminder: {r['text']}"
                type_out(msg)
                self.speech.speak_sync(msg)
                r["done"] = True
                self.task.save()


if __name__ == "__main__":
    Assistant().run()
