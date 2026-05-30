"""
Structured logging for the LLM Alignment System.

Provides a single setup_logging() call that wires:
    - Console handler with colour-coded level output
    - Rotating file handler (10 MB per file, 5 rotations)
    - Per-library silencing so HuggingFace spam does not pollute logs
    - A JSONLineHandler for machine-readable log ingestion (optional)

Every training script calls setup_logging() as its first line.
get_logger(name) is used in every module instead of logging.getLogger()
directly so the name always appears in the structured output.

Design decisions:
    - Root logger level set to DEBUG so file handler captures everything.
      Console handler has its own level controlled by the `level` argument.
    - RotatingFileHandler instead of TimedRotatingFileHandler because
      training runs produce bursty log output (heavy during forward/backward,
      silent during data loading). Size-based rotation is more predictable.
    - JSONLineHandler is disabled by default. Enable it when you want to
      pipe logs into a log aggregator (Loki, Datadog, ELK).
    - Library silencing list is explicit, not a blanket WARNING override,
      so you can un-silence a specific library for debugging by removing
      it from the list.
"""
from __future__ import annotations

import json
import logging
import logging.handlers
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Format constants
# ─────────────────────────────────────────────────────────────────────────────

_CONSOLE_FMT = "%(asctime)s | %(levelname)-8s | %(name)-32s | %(message)s"
_FILE_FMT    = "%(asctime)s | %(levelname)-8s | %(name)-32s | %(process)d | %(message)s"
_DATE_FMT    = "%H:%M:%S"

# ANSI colour codes for console output
_LEVEL_COLOURS = {
    "DEBUG":    "\033[36m",   # Cyan
    "INFO":     "\033[32m",   # Green
    "WARNING":  "\033[33m",   # Yellow
    "ERROR":    "\033[31m",   # Red
    "CRITICAL": "\033[41m",   # Red background
}
_RESET = "\033[0m"

# Third-party libraries that produce excessive log output during training.
# Remove a library from this list to restore its logs for debugging.
_SILENCED_LIBRARIES = [
    "transformers",
    "transformers.tokenization_utils_base",
    "transformers.modeling_utils",
    "transformers.trainer",
    "transformers.trainer_utils",
    "datasets",
    "datasets.arrow_dataset",
    "datasets.builder",
    "peft",
    "peft.tuners",
    "trl",
    "accelerate",
    "accelerate.utils",
    "bitsandbytes",
    "urllib3",
    "urllib3.connectionpool",
    "filelock",
    "huggingface_hub",
    "huggingface_hub.utils",
    "huggingface_hub.file_download",
    "fsspec",
    "PIL",
    "matplotlib",
    "asyncio",
]


# ─────────────────────────────────────────────────────────────────────────────
# Colour formatter for console output
# ─────────────────────────────────────────────────────────────────────────────

class ColouredFormatter(logging.Formatter):
    """
    Logging formatter that adds ANSI colour codes to the level name.

    Only applies colour when the stream is a real TTY — disables
    automatically when output is piped or redirected, preventing
    escape codes from appearing in log files or CI output.
    """

    def __init__(self, fmt: str, datefmt: str, use_colour: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colour = use_colour

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        if not self.use_colour:
            return formatted
        colour = _LEVEL_COLOURS.get(record.levelname, "")
        if colour:
            # Only colour the level name portion, not the entire line
            formatted = formatted.replace(
                record.levelname,
                f"{colour}{record.levelname}{_RESET}",
                1,
            )
        return formatted


# ─────────────────────────────────────────────────────────────────────────────
# JSON line handler for structured log ingestion
# ─────────────────────────────────────────────────────────────────────────────

class JSONLineHandler(logging.FileHandler):
    """
    Writes one JSON object per log line to a .jsonl file.

    Useful for piping logs into Loki, Datadog, or any structured
    log aggregator. Disabled by default — pass json_log_path to
    setup_logging() to enable.

    Each line is a JSON object with keys:
        ts:       ISO 8601 timestamp
        level:    Log level string
        logger:   Logger name
        message:  Formatted message
        exc:      Exception traceback string (if present)
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            entry = {
                "ts":      datetime.utcnow().isoformat() + "Z",
                "level":   record.levelname,
                "logger":  record.name,
                "message": record.getMessage(),
                "exc":     None,
            }
            if record.exc_info:
                entry["exc"] = "".join(traceback.format_exception(*record.exc_info))
            self.stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self.flush()
        except Exception:
            self.handleError(record)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(
    level: str = "INFO",
    log_dir: str = "outputs/logs/training",
    run_name: str = "run",
    silence_libs: bool = True,
    use_colour: bool = True,
    json_log_path: Optional[str] = None,
) -> logging.Logger:
    """
    Initialise the logging system for one training or evaluation run.

    Call this as the very first line of every script entry point.
    Calling it multiple times is safe — handlers are cleared before
    re-adding so you never get duplicate log lines.

    Args:
        level:         Minimum log level for console output.
                       File handler always writes DEBUG regardless.
        log_dir:       Directory for rotating log files.
                       Created if it does not exist.
        run_name:      Prefix for the log filename.
                       Use the W&B run name for full traceability.
        silence_libs:  If True, suppress DEBUG/INFO from third-party libraries.
                       Set False only when debugging HuggingFace internals.
        use_colour:    Enable ANSI colour on console output.
                       Auto-disabled if stdout is not a TTY.
        json_log_path: Optional path for machine-readable JSON log.
                       If None, no JSON log is written.

    Returns:
        Logger named "llm_alignment" for the calling script to use.

    Example:
        logger = setup_logging(
            level="INFO",
            log_dir="outputs/logs/training",
            run_name="sft-qwen2.5-r16",
        )
        logger.info("Training started")
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"{run_name}_{timestamp}.log"

    # ── Console handler ───────────────────────────────────────────────────────
    is_tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
    console_formatter = ColouredFormatter(
        fmt=_CONSOLE_FMT,
        datefmt=_DATE_FMT,
        use_colour=use_colour and is_tty,
    )
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(getattr(logging, level.upper(), logging.INFO))

    # ── Rotating file handler ─────────────────────────────────────────────────
    file_formatter = logging.Formatter(fmt=_FILE_FMT, datefmt=_DATE_FMT)
    file_handler = logging.handlers.RotatingFileHandler(
        filename=log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.DEBUG)

    # ── Root logger ───────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    root.handlers.clear()  # Prevent duplicate handlers on repeated calls
    root.addHandler(console_handler)
    root.addHandler(file_handler)

    # ── Optional JSON log ─────────────────────────────────────────────────────
    if json_log_path is not None:
        Path(json_log_path).parent.mkdir(parents=True, exist_ok=True)
        json_handler = JSONLineHandler(json_log_path, mode="a", encoding="utf-8")
        json_handler.setLevel(logging.INFO)
        root.addHandler(json_handler)

    # ── Silence noisy third-party libraries ───────────────────────────────────
    if silence_libs:
        for lib in _SILENCED_LIBRARIES:
            logging.getLogger(lib).setLevel(logging.WARNING)

    # ── Return named logger ───────────────────────────────────────────────────
    logger = logging.getLogger("llm_alignment")
    logger.info(
        f"Logging initialised | "
        f"level={level} | "
        f"file={log_file} | "
        f"colour={'on' if (use_colour and is_tty) else 'off'}"
    )
    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Return a named child logger under the llm_alignment namespace.

    Usage in any module:
        from src.utils.logging import get_logger
        logger = get_logger(__name__)

    This ensures all module loggers appear under the llm_alignment
    hierarchy in log output, making it easy to filter by component.

    Example log line:
        10:42:31 | INFO     | llm_alignment.src.data.generator | ...
    """
    return logging.getLogger(name)


def log_section(logger: logging.Logger, title: str, width: int = 60) -> None:
    """
    Print a clearly visible section separator to the log.

    Use at the start of major pipeline phases to make the log scannable.

    Example:
        log_section(logger, "Phase 5: SFT Training")

    Output:
        10:42:31 | INFO | llm_alignment | ════════════════════════════
        10:42:31 | INFO | llm_alignment | Phase 5: SFT Training
        10:42:31 | INFO | llm_alignment | ════════════════════════════
    """
    bar = "═" * width
    logger.info(bar)
    logger.info(title)
    logger.info(bar)


def log_dict(
    logger: logging.Logger,
    data: dict,
    title: str = "",
    level: str = "INFO",
) -> None:
    """
    Log a dictionary with aligned key-value formatting.

    Use for logging hyperparameters, evaluation results, and
    system info at the start of a training run.

    Example:
        log_dict(logger, {"learning_rate": 2e-4, "lora_r": 16}, "Config")

    Output:
        10:42:31 | INFO | Config
        10:42:31 | INFO |   learning_rate   2e-04
        10:42:31 | INFO |   lora_r          16
    """
    log_fn = getattr(logger, level.lower(), logger.info)
    if title:
        log_fn(title)
    if not data:
        log_fn("  (empty)")
        return
    max_key_len = max(len(str(k)) for k in data.keys())
    for k, v in data.items():
        log_fn(f"  {str(k):<{max_key_len + 2}} {v}")