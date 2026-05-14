from __future__ import annotations 

import sys 
from datetime import datetime 
from pathlib import Path 
from loguru import logger 
from src.constants import PROJECT_ROOT_DIR


LOGS_DIR = PROJECT_ROOT_DIR / "logs"

__all__ = ["logger", "configure_logger"]


def configure_logger(log_level: str = "INFO",
                     logs_dir: Path | None = None) -> Path:
    """Configure loguru logger setup."""
    logs_root = logs_dir if logs_dir is not None else LOGS_DIR

    now = datetime.now()
    day_folder = logs_root / now.strftime("%Y-%m-%d")
    day_folder.mkdir(parents=True, exist_ok=True)
    log_file_path = day_folder / f"{now.strftime('%H-%M-%S')}.log"

    logger.remove()

    logger.add(
        sys.stdout,
        level=log_level,
        colorize=True,
        format=(
            "<green>{time:HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=True,
        diagnose=False,
    )

    logger.add(
        log_file_path,
        level="DEBUG",
        rotation="50 MB",
        retention="30 days",
        encoding="utf-8",
        format=(
            "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
            "{level: <8} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
        backtrace=True,
        diagnose=True,
        enqueue=True,
    )

    logger.info("Logger configured — log file: {}", log_file_path)
    return log_file_path

configure_logger()