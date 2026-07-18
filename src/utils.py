"""Shared utilities for validation, logging, and output file handling."""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


RESULTS_DIR = Path("results")
LOGGER_NAME = "plga_simulation"


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
    """Return a project logger.

    Parameters
    ----------
    name : str
        Logger name.

    Returns
    -------
    logging.Logger
        Configured logger instance.

    Units
    -----
    Not applicable.
    """

    return logging.getLogger(name)


def configure_logging(level: int = logging.INFO) -> None:
    """Configure concise console logging for the simulation.

    Parameters
    ----------
    level : int
        Python logging level.

    Returns
    -------
    None

    Units
    -----
    Not applicable.
    """

    logging.basicConfig(
        level=level,
        format="%(message)s",
        force=True,
    )


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it is missing.

    Parameters
    ----------
    path : str or pathlib.Path
        Directory path to create.

    Returns
    -------
    pathlib.Path
        Normalized directory path.

    Units
    -----
    Not applicable.
    """

    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def ensure_parent_directory(path: str | Path) -> Path:
    """Create the parent directory for an output file.

    Parameters
    ----------
    path : str or pathlib.Path
        Output file path.

    Returns
    -------
    pathlib.Path
        Normalized output file path.

    Units
    -----
    Not applicable.
    """

    file_path = Path(path)
    ensure_directory(file_path.parent)
    return file_path


def validate_positive(value: float, name: str) -> None:
    """Validate that a scalar is positive and finite.

    Parameters
    ----------
    value : float
        Scalar value to validate.
    name : str
        Human-readable parameter name.

    Returns
    -------
    None

    Units
    -----
    Same as the validated value.
    """

    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite value; got {value!r}.")


def validate_nonnegative(value: float, name: str) -> None:
    """Validate that a scalar is non-negative and finite.

    Parameters
    ----------
    value : float
        Scalar value to validate.
    name : str
        Human-readable parameter name.

    Returns
    -------
    None

    Units
    -----
    Same as the validated value.
    """

    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a non-negative finite value; got {value!r}.")


def validate_array(
    array: np.ndarray,
    name: str,
    expected_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    """Validate array size, shape, and finite values.

    Parameters
    ----------
    array : numpy.ndarray
        Array to validate.
    name : str
        Human-readable array name.
    expected_shape : tuple of int, optional
        Required array shape.

    Returns
    -------
    numpy.ndarray
        Input converted with ``np.asarray``.

    Units
    -----
    Same as the validated array.
    """

    values = np.asarray(array)

    if values.size == 0:
        raise ValueError(f"{name} must not be empty.")

    if expected_shape is not None and values.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}; got {values.shape}."
        )

    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or Inf values.")

    return values


def validate_matching_shapes(*named_arrays: tuple[str, np.ndarray]) -> None:
    """Validate that multiple arrays share one shape.

    Parameters
    ----------
    *named_arrays : tuple of str and numpy.ndarray
        Name-array pairs to compare.

    Returns
    -------
    None

    Units
    -----
    Not applicable.
    """

    shapes = {name: np.asarray(array).shape for name, array in named_arrays}
    unique_shapes = set(shapes.values())

    if len(unique_shapes) > 1:
        raise ValueError(f"Array shapes do not match: {shapes}.")


def validate_no_nan_inf(values: Iterable[Any], name: str) -> None:
    """Validate that numeric values contain no NaN or Inf entries.

    Parameters
    ----------
    values : iterable
        Numeric values to inspect.
    name : str
        Human-readable collection name.

    Returns
    -------
    None

    Units
    -----
    Same as the validated values.
    """

    for value in values:
        if isinstance(value, (int, float, np.number)) and not np.isfinite(value):
            raise ValueError(f"{name} contains NaN or Inf values.")


def format_csv_value(value: Any) -> Any:
    """Format numeric CSV values with consistent precision.

    Parameters
    ----------
    value : object
        Value to format.

    Returns
    -------
    object
        Formatted string for floats; original value otherwise.

    Units
    -----
    Same as the input value.
    """

    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            raise ValueError("CSV output contains NaN or Inf.")
        return f"{float(value):.10g}"

    if isinstance(value, (int, np.integer)):
        return int(value)

    return value


def write_csv_rows(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write CSV rows with headers, UTF-8 encoding, and finite checks.

    Parameters
    ----------
    path : str or pathlib.Path
        Output CSV path.
    fieldnames : sequence of str
        CSV column headers.
    rows : sequence of mappings
        Row data keyed by column name.

    Returns
    -------
    None

    Units
    -----
    Not applicable.
    """

    output_path = ensure_parent_directory(path)

    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(
                {field: format_csv_value(row.get(field, "")) for field in fieldnames}
            )


def write_text(path: str | Path, content: str) -> None:
    """Write UTF-8 text after creating the parent directory.

    Parameters
    ----------
    path : str or pathlib.Path
        Output text path.
    content : str
        Text content to write.

    Returns
    -------
    None

    Units
    -----
    Not applicable.
    """

    output_path = ensure_parent_directory(path)
    output_path.write_text(content, encoding="utf-8")
