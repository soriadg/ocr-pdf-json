import sys
import time

LOG_ENABLED = False
LOG_HANDLE = None


def setup_logger(enabled, path=None):
    global LOG_ENABLED, LOG_HANDLE
    LOG_ENABLED = enabled
    if enabled and path:
        try:
            LOG_HANDLE = open(path, "a", encoding="utf-8")
        except OSError as exc:
            raise SystemExit(f"Cannot open log file: {path}. {exc}") from exc


def close_logger():
    global LOG_HANDLE
    if LOG_HANDLE:
        LOG_HANDLE.close()
        LOG_HANDLE = None


def log(message):
    if not LOG_ENABLED:
        return
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    if LOG_HANDLE:
        LOG_HANDLE.write(line + "\n")
        LOG_HANDLE.flush()
    else:
        print(line, file=sys.stderr, flush=True)
