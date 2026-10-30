import logging
import re
import sys

# Patterns to redact from logs
SENSITIVE_PATTERNS = [
    re.compile(r"(Bearer\s+)[A-Za-z0-9_\-\.]+", re.IGNORECASE),
    re.compile(r"(github_pat_[A-Za-z0-9_]+)", re.IGNORECASE),
    re.compile(r"('password':\s*')[^']+'", re.IGNORECASE),
    re.compile(r'("password":\s*")[^"]+"', re.IGNORECASE),
    re.compile(r"('token':\s*')[^']+'", re.IGNORECASE),
    re.compile(r'("token":\s*")[^"]+"', re.IGNORECASE),
]


class SafeLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        for pattern in SENSITIVE_PATTERNS:
            msg = pattern.sub(r"\1[REDACTED]", msg)
        return msg


def setup_logging(debug: bool = False) -> logging.Logger:
    level = logging.DEBUG if debug else logging.INFO
    logger = logging.getLogger("memoryos")
    logger.setLevel(level)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = SafeLogFormatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] [request_id=%(request_id)s] %(message)s",
            defaults={"request_id": "-"},
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


logger = setup_logging()
