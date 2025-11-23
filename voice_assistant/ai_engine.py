# ai_engine.py
"""
Simple local AI fallback engine — intentionally avoids any `google.generativeai`
imports so your project won't crash if the Gemini package is missing.

Behaviour:
 - If the prompt looks like a web lookup (contains 'who is', 'what is', 'search for', 'define'),
   this engine will open the default browser with a Google search and return a short message.
 - For greetings / thanks it returns canned replies.
 - Otherwise it echoes the prompt with a short comment.
 - If you want real Gemini-powered responses later, install the official package and
   replace or extend this module. See the message printed when unavailable.
"""

import webbrowser
import random
from config import GEMINI_API_KEY

class AIEngine:
    def __init__(self):
        # We intentionally do NOT import or use google.generativeai here.
        self.available = False
        if GEMINI_API_KEY and GEMINI_API_KEY != "my api key":
            # We detect the presence of a key but do not attempt to use it.
            print("[AI ENGINE] GEMINI_API_KEY present, but this local engine does not use Gemini.")
            print("[AI ENGINE] To enable Gemini, install 'google-generativeai' and replace this module.")
        else:
            print("[AI ENGINE] No GEMINI_API_KEY configured. Running local fallback AI.")

        # Small canned replies to feel conversational
        self.greetings = ["Hey!", "Hello!", "Hi there!"]
        self.farewells = ["Goodbye!", "See you!", "Bye!"]
        self.thanks = ["You're welcome!", "No problem!", "Anytime!"]

    def _looks_like_search(self, text):
        t = text.lower()
        triggers = ["who is", "what is", "search for", "define", "look up", "wiki", "wikipedia", "meaning of"]
        return any(trigger in t for trigger in triggers)

    def _handle_search(self, text):
        q = text.strip()
        if not q:
            return "I couldn't form a search query."
        url = f"https://www.google.com/search?q={q.replace(' ', '+')}"
        try:
            webbrowser.open(url)
            return f"I opened a web search for: {q}"
        except Exception:
            return f"I couldn't open the browser, but you can search for: {q}"

    def ask(self, prompt):
        """
        Return a text reply for the given prompt.
        This function is intentionally synchronous and simple.
        """
        if not prompt or not str(prompt).strip():
            return "I didn't get a question. Try asking something."

        text = str(prompt).strip()

        # quick rule-based replies:
        tl = text.lower()

        if any(g in tl for g in ("hello", "hi ", "hey ", "good morning", "good evening")):
            return random.choice(self.greetings)

        if any(g in tl for g in ("thank you", "thanks", "thx")):
            return random.choice(self.thanks)

        if any(g in tl for g in ("goodbye", "bye", "see you")):
            return random.choice(self.farewells)

        # If it looks like the user wants a web lookup, open the browser
        if self._looks_like_search(text):
            return self._handle_search(text)

        # Default: echo + small transformation to feel helpful
        # Keep responses safely short so the assistant output isn't huge.
        examples = [
            "I heard you. You asked: {}",
            "You said: {}. I don't have a remote AI here, but I can help open a web search if you want.",
            "I'll repeat that: {}",
            "Local AI (fallback): {}"
        ]
        template = random.choice(examples)
        return template.format(text)
