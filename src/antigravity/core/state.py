import json
import time
from pathlib import Path
from typing import Any
from contextlib import contextmanager

try:
    import fcntl
    HAS_LOCKING = "fcntl"
except ImportError:
    try:
        import msvcrt
        HAS_LOCKING = "msvcrt"
    except ImportError:
        HAS_LOCKING = None

def lock_file_fn(f):
    if HAS_LOCKING == "fcntl":
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
    elif HAS_LOCKING == "msvcrt":
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, 1)

def unlock_file_fn(f):
    if HAS_LOCKING == "fcntl":
        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    elif HAS_LOCKING == "msvcrt":
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)

class AtomicStateManager:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock_path = self.path.parent / f"{self.path.name}.lock"
    
    @contextmanager
    def lock(self):
        """Acquire exclusive file lock."""
        with open(self.lock_path, "w") as lock_file:
            lock_file_fn(lock_file)
            try:
                yield
            finally:
                unlock_file_fn(lock_file)
    
    def read(self) -> dict[str, Any]:
        with self.lock():
            if self.path.exists():
                try:
                    with open(self.path, "r") as f:
                        return json.load(f)
                except Exception:
                    return {}
            return {}
    
    def write(self, data: dict[str, Any]) -> None:
        with self.lock():
            temp_file = self.path.parent / f"{self.path.name}.tmp"
            with open(temp_file, "w") as f:
                json.dump(data, f, indent=2)
            
            if temp_file.exists():
                for _ in range(5):
                    try:
                        temp_file.replace(self.path)
                        break
                    except Exception:
                        time.sleep(0.1)
                else:
                    temp_file.replace(self.path)
