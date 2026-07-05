from collections import defaultdict, deque
from threading import Lock


MAX_MESSAGES_PER_SESSION = 8
_history: dict[str, deque[dict]] = defaultdict(lambda: deque(maxlen=MAX_MESSAGES_PER_SESSION))
_lock = Lock()


def get_recent_history(session_id: str | None, limit: int = MAX_MESSAGES_PER_SESSION) -> list[dict]:
    if not session_id:
        return []

    with _lock:
        return list(_history[str(session_id)])[-limit:]


def add_message(session_id: str | None, role: str, content: str) -> None:
    if not session_id:
        return

    content = str(content or "").strip()
    if not content:
        return

    with _lock:
        _history[str(session_id)].append({"role": role, "content": content})


def add_exchange(session_id: str | None, user_text: str, assistant_text: str) -> None:
    add_message(session_id, "user", user_text)
    add_message(session_id, "assistant", assistant_text)
