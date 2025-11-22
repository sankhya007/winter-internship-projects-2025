# task_manager.py
import json
import os
import requests
from datetime import datetime

REM_FILE = "reminders.json"


class TaskManager:
    def __init__(self):
        if not os.path.exists(REM_FILE):
            with open(REM_FILE, "w") as f:
                json.dump([], f)

    def save(self, data=None):
        if data is None:
            data = self.load()
        with open(REM_FILE, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        with open(REM_FILE) as f:
            return json.load(f)

    def tell_time(self):
        return datetime.now().strftime("The time is %H:%M")

    def tell_date(self):
        return datetime.now().strftime("Today is %d %B %Y")

    def check_weather(self, city):
        try:
            url = f"https://wttr.in/{city}?format=3"
            return requests.get(url, timeout=3).text
        except:
            return "Weather unavailable."

    def get_news(self):
        return "Top news: India is progressing rapidly."

    def search_web(self, query):
        return f"Here’s your search result for: {query}"

    def set_reminder(self, text, when):
        data = self.load()
        data.append({"text": text, "time": when.isoformat(), "done": False})
        self.save(data)
        return f"Reminder set for {when.strftime('%H:%M')}."

    def pending(self):
        data = self.load()
        now = datetime.now()
        due = []
        for r in data:
            t = datetime.fromisoformat(r["time"])
            if not r["done"] and t <= now:
                due.append(r)
        return due
