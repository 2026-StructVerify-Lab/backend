"""
utils/logger.py — 로깅 + OpenTelemetry/Langfuse 연동 기반
"""
import logging, sys

def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger
