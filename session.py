"""
session.py - per-user state, so that several people can use the web app at the same time.

Every browser tab of app.py gets its own Session: its own states (rho_1, rho_2, ...), tomography
reports, figures folder and data folder. The command line (run.py, benchmark.py, the tests) uses
one default session, which behaves exactly like before: figures/ and data/.

How it works: the current session is kept in a context variable. app.py runs each question inside
    with session.use(session_id): ...
and every tool reads and writes the session that is current at that moment. LangGraph copies the
context into its worker threads, so the tools see the right session there too.

Sessions that are idle for SESSION_TTL_S seconds are deleted together with their files.
"""
import contextlib
import contextvars
import os
import re
import shutil
import threading
import time
from collections.abc import MutableMapping
from dataclasses import dataclass, field

ON_SPACE = bool(os.getenv("SPACE_ID"))                       # set by Hugging Face on a Space
SESSION_TTL_S = int(os.getenv("SESSION_TTL_S", "7200"))      # 2 hours without a question
MAX_SESSIONS = int(os.getenv("MAX_SESSIONS", "200"))
MAX_STATES = int(os.getenv("MAX_STATES_PER_SESSION", "300"))  # stored density matrices per session
SESSIONS_DIR = "sessions"                                      # under figures/ and data/
DEFAULT_ID = "default"


@dataclass
class Session:
    id: str
    store: dict = field(default_factory=dict)        # handle -> density matrix
    desc: dict = field(default_factory=dict)         # handle -> short description
    tomo: dict = field(default_factory=dict)         # handle -> tomography report
    fit: dict = field(default_factory=dict)          # handle -> fitted counts (bootstrap)
    samples: dict = field(default_factory=dict)      # handle -> bootstrap states
    question: dict = field(default_factory=lambda: {"text": ""})   # the user's latest question
    calls: list = field(default_factory=list)        # MCP clients: their tool calls + results (mcp_server.py)
    check: dict = field(default_factory=dict)        # MCP: where the last check_answer started and ended
    last_provider: str | None = None                 # which model answered this user's last call
    asked: list = field(default_factory=list)        # times of this user's recent questions (rate limit)
    last_used: float = field(default_factory=time.time)

    @property
    def is_default(self) -> bool:
        return self.id == DEFAULT_ID

    @property
    def fig_dir(self) -> str:
        return "figures" if self.is_default else f"figures/{SESSIONS_DIR}/{self.id}"

    @property
    def data_dir(self) -> str:
        """Where this session's own files go (uploads, simulated counts)."""
        return "data" if self.is_default else f"data/{SESSIONS_DIR}/{self.id}"

    def clear(self):
        for d in (self.store, self.desc, self.tomo, self.fit, self.samples):
            d.clear()
        self.question["text"] = ""
        self.calls.clear()
        self.check.clear()


# numpy/scipy/matplotlib work is serialized: matplotlib's pyplot keeps global state and is not thread-safe,
# and several users' requests (chat and Workbench) can run at the same time. Only the computation holds
# the lock, not the (slow) language-model calls. Re-entrant: a tool may call another tool.
COMPUTE_LOCK = threading.RLock()

_DEFAULT = Session(DEFAULT_ID)
_CURRENT: contextvars.ContextVar = contextvars.ContextVar("qi_session", default=_DEFAULT)
_SESSIONS: dict[str, Session] = {}
_LOCK = threading.Lock()
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def current() -> Session:
    return _CURRENT.get()


def get(session_id: str) -> Session:
    """The session with this id (created on first use). Idle sessions are cleaned up here."""
    if session_id == DEFAULT_ID:
        return _DEFAULT
    if not _SAFE_ID.match(session_id or ""):
        raise ValueError("invalid session id")
    with _LOCK:
        _expire_locked()
        s = _SESSIONS.get(session_id)
        if s is None:
            if len(_SESSIONS) >= MAX_SESSIONS:          # drop the longest-idle session
                oldest = min(_SESSIONS.values(), key=lambda x: x.last_used)
                _drop_locked(oldest.id)
            s = _SESSIONS[session_id] = Session(session_id)
        s.last_used = time.time()
        return s


@contextlib.contextmanager
def use(session_id: str):
    """Run the enclosed code with this session as the current one."""
    token = _CURRENT.set(get(session_id))
    try:
        yield _CURRENT.get()
    finally:
        _CURRENT.reset(token)


def drop(session_id: str):
    with _LOCK:
        _drop_locked(session_id)


def _drop_locked(session_id: str):
    s = _SESSIONS.pop(session_id, None)
    if s is None:
        return
    s.clear()
    for d in (s.fig_dir, s.data_dir):
        shutil.rmtree(d, ignore_errors=True)
    for hook in list(ON_DROP):
        try:
            hook(session_id)
        except Exception:  # noqa: BLE001 - cleanup must never break a request
            pass


def _expire_locked():
    now = time.time()
    for sid in [k for k, s in _SESSIONS.items() if now - s.last_used > SESSION_TTL_S]:
        _drop_locked(sid)


def active_ids() -> list[str]:
    with _LOCK:
        return list(_SESSIONS)


def exists(session_id: str) -> bool:
    """Is this session alive (not expired, not dropped)? Does not create it or refresh its idle time."""
    with _LOCK:
        _expire_locked()
        return session_id in _SESSIONS


def make_room(prefix: str, limit: int):
    """Before a new session with this id prefix is created: drop the longest-idle sessions with the
    prefix until fewer than `limit` remain. MCP clients (mcp_server.py) can open sessions cheaply, so
    they get their own share and can never push web visitors' sessions out."""
    with _LOCK:
        _expire_locked()
        mine = sorted((s for k, s in _SESSIONS.items() if k.startswith(prefix)), key=lambda s: s.last_used)
        while mine and len(mine) >= max(1, limit):
            _drop_locked(mine.pop(0).id)


ON_DROP: list = []          # callbacks(session_id), e.g. to forget the chat memory of that session


class SessionDict(MutableMapping):
    """A dict-like view of one dict of the CURRENT session (agent_tools.STORE, DESC, ...). Existing code
    (STORE[h], h in STORE, STORE.clear(), ...) keeps working and is per-user automatically."""

    def __init__(self, attr: str):
        self._attr = attr

    def _d(self) -> dict:
        return getattr(current(), self._attr)

    def __getitem__(self, k):
        return self._d()[k]

    def __setitem__(self, k, v):
        self._d()[k] = v

    def __delitem__(self, k):
        del self._d()[k]

    def __iter__(self):
        return iter(list(self._d()))

    def __len__(self):
        return len(self._d())

    def __contains__(self, k):
        return k in self._d()

    def __repr__(self):
        return repr(self._d())

    def clear(self):
        self._d().clear()

    def copy(self):
        return dict(self._d())


UPLOADS = "uploads"                     # what a web user sees instead of data/sessions/<id>
_SESSION_PATH = re.compile(r"data[/\\]" + SESSIONS_DIR + r"[/\\][^/\\\s'\"()]+[/\\]")


def real_path(path: str) -> str:
    """'uploads/x.csv' -> this web session's own folder; every other path is returned unchanged."""
    p = path.strip().strip('"').strip("'").replace("\\", "/")
    s = current()
    if not s.is_default and p.startswith(UPLOADS + "/"):
        return f"{s.data_dir}/{p[len(UPLOADS) + 1:]}"
    return path


def display_path(text: str) -> str:
    """Replace the long session folder 'data/sessions/<id>/' by 'uploads/' in text shown to a user."""
    return _SESSION_PATH.sub(UPLOADS + "/", text)


def allowed_csv(path: str) -> bool:
    """May the current session read this CSV? The command line may read any file; a web-app session
    may read only the shared examples in data/ and its own uploads and simulations."""
    s = current()
    if s.is_default:
        return True
    real = os.path.realpath(path)
    shared = os.path.realpath("data")
    own = os.path.realpath(s.data_dir)
    return os.path.dirname(real) == shared or real.startswith(own + os.sep)


def readable_csvs() -> list[str]:
    """CSV files the current session may use (for 'file not found' messages)."""
    s = current()
    dirs = (".", "data") if s.is_default else ("data", s.data_dir)
    return sorted({display_path(os.path.join(d, f).replace("\\", "/")) for d in dict.fromkeys(dirs)
                   if os.path.isdir(d) for f in os.listdir(d) if f.lower().endswith(".csv")})
