import json
import uuid
import time
from pathlib import Path
from typing import List, Dict

HISTORY_FILE = Path("chat_history.json")

class HistoryManager:
    def __init__(self):
        self._load()

    def _load(self):
        if HISTORY_FILE.exists():
            try:
                with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                    self.sessions = json.load(f)
            except Exception:
                self.sessions = []
        else:
            self.sessions = []

    def _save(self):
        self.sessions.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.sessions, f, ensure_ascii=False, indent=2)

    def create_session(self, title="新对话") -> str:
        thread_id = str(uuid.uuid4())
        new_session = {
            "id": thread_id,
            "title": title,
            "timestamp": time.time(),
            "is_new": True
        }
        self.sessions.insert(0, new_session)
        self._save()
        return thread_id

    def update_title(self, thread_id: str, new_title: str):
        for session in self.sessions:
            if session["id"] == thread_id:
                session["title"] = new_title[:20] + "..." if len(new_title) > 20 else new_title
                session["is_new"] = False
                session["timestamp"] = time.time()
                break
        self._save()

    def delete_session(self, thread_id: str):
        self.sessions = [s for s in self.sessions if s["id"] != thread_id]
        self._save()

    def get_all_sessions(self) -> List[Dict]:
        return self.sessions

    def get_session(self, thread_id: str) -> Dict:
        for s in self.sessions:
            if s["id"] == thread_id:
                return s
        return None

history_manager = HistoryManager()
