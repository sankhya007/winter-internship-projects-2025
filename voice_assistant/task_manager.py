
# task_manager.py
import json
from pathlib import Path
from datetime import datetime
import requests
import webbrowser
from config import REMINDERS_FILENAME

BASE = Path(__file__).parent

class TaskManager:
    def __init__(self):
        self.file = BASE / REMINDERS_FILENAME
        if not self.file.exists():
            self.file.write_text('[]')
        self.reminders = self._load()

    def _load(self):
        try:
            return json.loads(self.file.read_text())
        except Exception:
            return []

    def save(self):
        try:
            self.file.write_text(json.dumps(self.reminders, indent=2))
        except Exception as e:
            print("Could not save reminders:", e)

    def tell_time(self):
        return datetime.now().strftime("The time is %I:%M %p")

    def tell_date(self):
        return datetime.now().strftime("Today is %A, %B %d, %Y")

    def _geocode(self, place):
        try:
            r = requests.get('https://geocoding-api.open-meteo.com/v1/search', params={'name': place, 'count': 1}, timeout=6)
            r.raise_for_status()
            j = r.json()
            if 'results' in j and j['results']:
                first = j['results'][0]
                return first.get('name'), first.get('latitude'), first.get('longitude')
        except Exception as e:
            print('Geocode error:', e)
        return None

    def check_weather(self, place='London'):
        geo = self._geocode(place)
        if not geo:
            return f"Couldn't find location '{place}'."
        name, lat, lon = geo
        try:
            r = requests.get('https://api.open-meteo.com/v1/forecast', params={'latitude': lat, 'longitude': lon, 'current_weather': True, 'timezone': 'auto'}, timeout=6)
            r.raise_for_status()
            j = r.json()
            cw = j.get('current_weather', {})
            if cw:
                temp = cw.get('temperature')
                wind = cw.get('windspeed')
                return f"Weather in {name}: {temp}°C, wind {wind} m/s."
            return 'Weather data unavailable.'
        except Exception as e:
            print('Weather fetch error:', e)
            return 'Weather service unavailable.'

    def set_reminder(self, text, when_dt):
        self.reminders.append({"text": text, "time": when_dt.isoformat(), "done": False})
        self.save()
        return f"Reminder set for {when_dt.strftime('%Y-%m-%d %I:%M %p')}: {text}"

    def pending(self):
        now = datetime.now()
        due = []
        for r in self.reminders:
            try:
                if not r.get('done', False) and datetime.fromisoformat(r['time']) <= now:
                    due.append(r)
            except Exception:
                r['done'] = True
        return due

    def search_web(self, query):
        try:
            url = f"https://www.google.com/search?q={requests.utils.requote_uri(query)}"
            webbrowser.open(url)
            return f"Searching the web for {query}"
        except Exception:
            return "Unable to open web browser."

    def get_news(self):
        try:
            webbrowser.open('https://news.google.com')
            return 'Opening the news in your web browser.'
        except Exception:
            return 'Unable to open news.'