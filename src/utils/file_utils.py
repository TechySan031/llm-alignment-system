"""
File I/O utilities for the LLM Alignment System.

Provides consistent, safe, and typed wrappers around common file operations:
    - JSONL streaming read/write (primary format for large datasets)
    - JSON read/write with proper encoding and indentation
    - Safe directory creation
    - File size and directory size reporting
    - Atomic file writes (write to temp, rename — prevents partial writes)
    - Checkpoint file management
    - CSV read/write for metrics/benchmark_history.csv

All write functions create parent directories automatically.
All read functions raise FileNotFoundError with a clear message
rather than the default cryptic OS error.

Why JSONL over JSON for datasets:
    A 10K-example dataset as a single JSON array must be fully parsed
    into memory before any example can be read. A JSONL file can be
    streamed line by line — constant memory usage regardless of file size.
    JSONL also supports append operations without rewriting the entire file,
    which is used when incrementally saving evaluation results.
"""
from __future__ import annotations

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Generator, Iterator, Optional, Union


# ─────────────────────────────────────────────────────────────────────────────
# JSONL utilities
# ─────────────────────────────────────────────────────────────────────────────

def read_jsonl(path: Union[str, Path]) -> Generator[dict, None, None]:
    """
    Stream-read a JSONL file one record at a time.

    Memory usage is O(1) with respect to file size — only one line
    is held in memory at a time. Use this for reading large dataset
    files during preprocessing.

    Args:
        path: Path to .jsonl file.

    Yields:
        One parsed dict per non-empty line.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If any line contains invalid JSON
                              (includes line number in error message).

    Example:
        for example in read_jsonl("data/processed/train.jsonl"):
            print(example["task_type"])
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"JSONL file not found: {path}\n"
            f"Run scripts/generate_dataset.py first."
        )
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_num} of {path}: {e.msg}",
                    e.doc,
                    e.pos,
                )


def read_jsonl_all(path: Union[str, Path]) -> list[dict]:
    """
    Read all records from a JSONL file into a list.

    Use only when you need random access to the full dataset.
    For sequential processing, prefer the generator read_jsonl().

    Args:
        path: Path to .jsonl file.

    Returns:
        List of parsed dicts.
    """
    return list(read_jsonl(path))


def write_jsonl(
    records: list[dict],
    path: Union[str, Path],
    mode: str = "w",
    ensure_ascii: bool = False,
) -> None:
    """
    Write a list of dicts to a JSONL file.

    Uses an atomic write pattern (temp file + rename) to prevent
    partial writes from corrupting the file on crash or interrupt.

    Args:
        records:      List of JSON-serialisable dicts.
        path:         Output path. Parent directories created automatically.
        mode:         "w" to overwrite, "a" to append.
                      Atomic write is only used for "w" mode.
        ensure_ascii: If False, Unicode characters are written as-is.
                      Set True only if the downstream consumer requires ASCII.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if mode == "a":
        # Append mode: no atomic write needed
        with open(path, mode="a", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")
        return

    # Atomic write: write to temp file in the same directory, then rename
    dir_ = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp", prefix=".write_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=ensure_ascii) + "\n")
        # On Windows, destination must not exist before rename
        if path.exists():
            path.unlink()
        os.rename(tmp_path, path)
    except Exception:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def append_jsonl(record: dict, path: Union[str, Path]) -> None:
    """
    Append a single record to a JSONL file.

    Used for incrementally saving evaluation results — each prediction
    is written immediately so results are not lost if the process crashes.

    Args:
        record: Single JSON-serialisable dict.
        path:   Output path. Created if it does not exist.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, mode="a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# JSON utilities
# ─────────────────────────────────────────────────────────────────────────────

def read_json(path: Union[str, Path]) -> dict:
    """
    Read a single JSON file into a dict.

    Args:
        path: Path to .json file.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file contains invalid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"JSON file not found: {path}"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(
    data: Any,
    path: Union[str, Path],
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> None:
    """
    Write a JSON-serialisable object to a file.

    Uses atomic write (temp + rename) to prevent partial writes.

    Args:
        data:         Any JSON-serialisable object (dict, list, etc.).
        path:         Output path. Parent directories created automatically.
        indent:       JSON indentation spaces. Use 2 for human-readable output.
        ensure_ascii: See write_jsonl.
        sort_keys:    Sort dictionary keys alphabetically for stable diffs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    dir_ = path.parent
    fd, tmp_path = tempfile.mkstemp(dir=dir_, suffix=".tmp", prefix=".write_")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(
                data, f,
                indent=indent,
                ensure_ascii=ensure_ascii,
                sort_keys=sort_keys,
                default=_json_serialiser,
            )
            f.write("\n")  # POSIX: files end with a newline
        if path.exists():
            path.unlink()
        os.rename(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _json_serialiser(obj: Any) -> Any:
    """
    Custom JSON serialiser for types not handled by the default encoder.

    Handles:
        datetime    → ISO 8601 string
        Path        → POSIX path string
        set         → sorted list
        Enum        → .value
        Pydantic    → .model_dump()
        bytes       → base64 string
    """
    import datetime
    from enum import Enum

    if isinstance(obj, (datetime.datetime, datetime.date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, set):
        return sorted(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, bytes):
        import base64
        return base64.b64encode(obj).decode("ascii")
    # Pydantic v2
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serialisable")


# ─────────────────────────────────────────────────────────────────────────────
# CSV utilities
# ─────────────────────────────────────────────────────────────────────────────

def read_csv(path: Union[str, Path]) -> list[dict]:
    """
    Read a CSV file into a list of dicts (one dict per row).

    Uses csv.DictReader so the first row is treated as the header.

    Args:
        path: Path to .csv file.

    Returns:
        List of OrderedDicts with column names as keys.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def write_csv(
    records: list[dict],
    path: Union[str, Path],
    fieldnames: Optional[list[str]] = None,
    mode: str = "w",
) -> None:
    """
    Write a list of dicts to a CSV file.

    Args:
        records:    List of dicts. All dicts must have the same keys
                    unless fieldnames is explicitly provided.
        path:       Output path. Parent directories created automatically.
        fieldnames: Column order. If None, inferred from first record keys.
        mode:       "w" to overwrite, "a" to append.
                    When appending, header is not written.
    """
    if not records:
        return

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if fieldnames is None:
        fieldnames = list(records[0].keys())

    write_header = mode == "w"
    with open(path, mode=mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        writer.writerows(records)


def append_csv_row(row: dict, path: Union[str, Path], fieldnames: list[str]) -> None:
    """
    Append a single row to a CSV file.

    Creates the file with a header if it does not exist.
    Used for incrementally updating metrics/benchmark_history.csv.

    Args:
        row:        Dict with at least the keys in fieldnames.
        path:       Output path.
        fieldnames: Column names (used for header if creating new file).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with open(path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Directory and path utilities
# ─────────────────────────────────────────────────────────────────────────────

def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Create a directory and all parents if they do not exist.
    Returns the Path object for use in f-strings and path operations.

    Example:
        output_dir = ensure_dir("outputs/models/sft")
        model.save_pretrained(str(output_dir / "final"))
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_file_size_mb(path: Union[str, Path]) -> float:
    """Return the size of a file in megabytes."""
    p = Path(path)
    if not p.exists():
        return 0.0
    return round(p.stat().st_size / 1_000_000, 3)


def get_dir_size_mb(path: Union[str, Path]) -> float:
    """
    Return the total size of all files under a directory in megabytes.

    Used to report adapter checkpoint sizes after training.
    For a LoRA r=16 adapter on Qwen2.5-7B: expect ~80–120 MB.
    For the full merged model: expect ~14,000 MB (14 GB).
    """
    p = Path(path)
    if not p.exists():
        return 0.0
    total = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    return round(total / 1_000_000, 2)


def list_files(
    directory: Union[str, Path],
    pattern: str = "*",
    recursive: bool = False,
) -> list[Path]:
    """
    List files matching a glob pattern in a directory.

    Args:
        directory: Root directory to search.
        pattern:   Glob pattern (e.g. "*.jsonl", "checkpoint-*").
        recursive: If True, search recursively with rglob.

    Returns:
        Sorted list of matching Path objects.

    Example:
        checkpoints = list_files("experiments/sft_runs", "checkpoint-*")
    """
    p = Path(directory)
    if not p.exists():
        return []
    fn = p.rglob if recursive else p.glob
    return sorted(fn(pattern))


def safe_copy(
    src: Union[str, Path],
    dst: Union[str, Path],
    overwrite: bool = True,
) -> Path:
    """
    Copy a file to a destination, creating parent directories.

    Args:
        src:       Source file path.
        dst:       Destination file path.
        overwrite: If False and dst exists, raises FileExistsError.

    Returns:
        The destination Path.
    """
    src, dst = Path(src), Path(dst)
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    if dst.exists() and not overwrite:
        raise FileExistsError(f"Destination already exists: {dst}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def safe_move(
    src: Union[str, Path],
    dst: Union[str, Path],
) -> Path:
    """
    Move a file or directory to a destination.
    Creates parent directories of dst automatically.
    """
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return dst


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint utilities
# ─────────────────────────────────────────────────────────────────────────────

def find_latest_checkpoint(
    checkpoint_dir: Union[str, Path],
) -> Optional[Path]:
    """
    Find the most recent checkpoint directory by step number.

    HuggingFace Trainer saves checkpoints as:
        experiments/sft_runs/run_001/checkpoint-200
        experiments/sft_runs/run_001/checkpoint-400

    This function returns the checkpoint with the highest step number.

    Args:
        checkpoint_dir: Parent directory containing checkpoint-N folders.

    Returns:
        Path to the latest checkpoint, or None if none found.

    Example:
        ckpt = find_latest_checkpoint("experiments/sft_runs/run_001")
        if ckpt:
            trainer = SFTTrainer.from_pretrained(str(ckpt), ...)
    """
    p = Path(checkpoint_dir)
    if not p.exists():
        return None

    checkpoints = []
    for d in p.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                checkpoints.append((step, d))
            except (IndexError, ValueError):
                continue

    if not checkpoints:
        return None

    checkpoints.sort(key=lambda x: x[0])
    return checkpoints[-1][1]


def find_best_checkpoint(
    checkpoint_dir: Union[str, Path],
    metric_file: str = "trainer_state.json",
    metric_key: str = "best_model_checkpoint",
) -> Optional[Path]:
    """
    Find the best checkpoint using HuggingFace Trainer's state file.

    The trainer saves trainer_state.json in the output directory with
    a 'best_model_checkpoint' key pointing to the checkpoint with the
    best eval metric.

    Args:
        checkpoint_dir: Directory containing trainer_state.json.
        metric_file:    Name of the trainer state file.
        metric_key:     Key in the state file pointing to best checkpoint.

    Returns:
        Path to best checkpoint, or None if not found.
    """
    state_file = Path(checkpoint_dir) / metric_file
    if not state_file.exists():
        return None
    try:
        state = read_json(state_file)
        best_path = state.get(metric_key)
        if best_path and Path(best_path).exists():
            return Path(best_path)
    except Exception:
        pass
    return None


def cleanup_old_checkpoints(
    checkpoint_dir: Union[str, Path],
    keep_last_n: int = 3,
    keep_best: Optional[str] = None,
) -> list[Path]:
    """
    Remove old checkpoints keeping only the most recent N.

    Args:
        checkpoint_dir: Directory containing checkpoint-N folders.
        keep_last_n:    Number of most recent checkpoints to keep.
        keep_best:      If provided, this checkpoint path is never deleted
                        even if it falls outside the keep_last_n window.

    Returns:
        List of deleted checkpoint Paths.
    """
    p = Path(checkpoint_dir)
    checkpoints: list[tuple[int, Path]] = []

    for d in p.iterdir():
        if d.is_dir() and d.name.startswith("checkpoint-"):
            try:
                step = int(d.name.split("-")[1])
                checkpoints.append((step, d))
            except (IndexError, ValueError):
                continue

    if len(checkpoints) <= keep_last_n:
        return []

    checkpoints.sort(key=lambda x: x[0])
    to_delete = checkpoints[:-keep_last_n]
    deleted: list[Path] = []

    for _, ckpt_path in to_delete:
        if keep_best and Path(keep_best).resolve() == ckpt_path.resolve():
            continue  # Never delete the best checkpoint
        shutil.rmtree(ckpt_path, ignore_errors=True)
        deleted.append(ckpt_path)

    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# Metrics persistence
# ─────────────────────────────────────────────────────────────────────────────

def save_metrics(
    metrics: dict,
    path: Union[str, Path],
    run_name: str = "",
) -> None:
    """
    Save evaluation metrics to JSON with a timestamp and run identifier.

    The saved file includes run_name and saved_at for traceability.
    Used by evaluation scripts to write to metrics/*.json.

    Args:
        metrics:  Dict of metric name → value.
        path:     Output path (e.g. "metrics/sft_metrics.json").
        run_name: Identifier for the run that produced these metrics.
    """
    from datetime import datetime

    payload = {
        "run_name": run_name,
        "saved_at": datetime.utcnow().isoformat() + "Z",
        "metrics": metrics,
    }
    write_json(payload, path)


def load_metrics(path: Union[str, Path]) -> dict:
    """
    Load previously saved metrics from a JSON file.

    Returns the metrics dict directly (unwraps the run_name/saved_at wrapper).
    Returns an empty dict if the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        return {}
    data = read_json(path)
    return data.get("metrics", data)


def update_benchmark_history(
    row: dict,
    csv_path: Union[str, Path] = "metrics/benchmark_history.csv",
) -> None:
    """
    Append one benchmark result row to the history CSV.

    Creates the file with a header if it does not exist.
    Used after every evaluation run to maintain a running history
    of all model versions and their metrics.

    The CSV is the data source for the benchmark comparison charts.

    Args:
        row:      Dict with benchmark metrics. Must include at minimum:
                  run_name, model_stage, evaluated_at.
        csv_path: Path to the history CSV file.

    Example:
        update_benchmark_history({
            "run_name":            "sft-qwen2.5-r16",
            "model_stage":         "sft",
            "evaluated_at":        "2024-06-15T10:30:00Z",
            "format_valid":        0.91,
            "instruction_followed": 0.88,
            "hallucination_rate":  0.06,
            "avg_alignment_score": 0.82,
            "avg_latency_ms":      340.0,
            "n_examples":          500,
        })
    """
    expected_fields = [
        "run_name", "model_stage", "evaluated_at",
        "format_valid", "schema_compliant", "instruction_followed",
        "hallucination_rate", "avg_alignment_score",
        "avg_bleu", "avg_rouge_l", "avg_field_f1",
        "avg_latency_ms", "p95_latency_ms",
        "avg_tokens_per_second", "n_examples",
    ]
    append_csv_row(row, csv_path, fieldnames=expected_fields)