from dataclasses import dataclass
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from time import perf_counter
from typing import Callable, Optional, Tuple, Union
import resource
from contextlib import contextmanager

import numpy

__all__ = [
    'G_GRAVITY',
    'timer',
    'TimerContext',
    'get_min_unit_dtype',
    'ensure_dir_exists',
]

# Gravitational constant
G_GRAVITY: float = 4.3e-09     # Mpc (km/s)^2 / M_sun


@dataclass(frozen=True)
class AnsiColor:
    """ANSI escape codes for colored and styled terminal output."""
    HEADER: str = "\033[35m"
    OKBLUE: str = "\033[34m"
    OKCYAN: str = "\033[36m"
    OKGREEN: str = "\033[32m"
    WARNING: str = "\033[33m"
    FAIL: str = "\033[31m"
    ENDC: str = "\033[0m"
    BOLD: str = "\033[1m"
    UNDERLINE: str = "\033[4m"
    BULLET: str = "\u25CF"


def timer(
    func: Optional[Callable] = None,
    *,
    fancy: bool = True,
    enabled: bool = True,
    precision: int = 3,
) -> Callable:
    """
    Decorator that measures and prints a function's execution time.

    Parameters
    ----------
    func : Callable, optional
        Function to be timed.
    fancy : bool, optional
        Whether to use colored output, by default True.
    enabled : bool, optional
        Whether timing is enabled, by default True.
    precision : int, optional
        Decimal precision for time display, by default 3.

    Returns
    -------
    Callable
        Decorated function or decorator factory.

    Examples
    --------
    >>> @timer
    ... def slow_function():
    ...     time.sleep(1)

    >>> @timer(fancy=False, precision=2)
    ... def another_function():
    ...     return 42
    """
    if func is None:
        return partial(timer, fancy=fancy, enabled=enabled, precision=precision)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not enabled:
            return func(*args, **kwargs)

        fmt = '%Y-%m-%d %H:%M:%S'
        start_time = datetime.now()
        perf_start = perf_counter()

        try:
            result = func(*args, **kwargs)
        finally:
            finish_time = datetime.now()
            elapsed = perf_counter() - perf_start

            hours = int(elapsed // 3600)
            remaining_seconds = elapsed % 3600
            minutes = int(remaining_seconds // 60)
            seconds = remaining_seconds % 60
            miliseconds = int(10**precision * (seconds - int(seconds)))
            elapsed_str = f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{miliseconds:03d}"

            if fancy:
                print(
                    f"\t{AnsiColor.BOLD}Process:{AnsiColor.ENDC} "
                    f"{AnsiColor.FAIL}{func.__name__}{AnsiColor.ENDC}\n"
                    f"Start:  {AnsiColor.OKBLUE}{start_time.strftime(fmt)}{AnsiColor.ENDC}\n"
                    f"Finish: {AnsiColor.OKBLUE}{finish_time.strftime(fmt)}{AnsiColor.ENDC}\n"
                    f"{AnsiColor.BULLET}{AnsiColor.BOLD}{AnsiColor.OKGREEN} Elapsed: "
                    f"{AnsiColor.ENDC}{AnsiColor.WARNING}{elapsed_str}{AnsiColor.ENDC}"
                )
            else:
                print(
                    f"\tProcess: {func.__name__}\n"
                    f"Start:  {start_time.strftime(fmt)}\n"
                    f"Finish: {finish_time.strftime(fmt)}\n"
                    f"{AnsiColor.BULLET} Elapsed: {elapsed_str}"
                )

        return result

    return wrapper


class TimerContext:
    """
    Context manager for timing code blocks.

    Parameters
    ----------
    name : str, optional
        Name of the operation being timed, by default "operation".
    fancy : bool, optional
        Whether to use ANSI colors and bullet markers, by default True.
    precision : int, optional
        Decimal precision for the time display, by default 3.

    Examples
    --------
    >>> with TimerContext("data processing"):
    ...     time.sleep(1)
    """

    def __init__(self, name: str = "operation", fancy: bool = True, precision: int = 3):
        self.name = name
        self.fancy = fancy
        self.precision = precision
        self.start_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = perf_counter()
        if self.fancy:
            print(
                f"{AnsiColor.BULLET} {AnsiColor.BOLD}{AnsiColor.OKGREEN}"
                f"{self.name}... {AnsiColor.ENDC}"
            )
        else:
            print(f"{self.name}...")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.elapsed = perf_counter() - self.start_time
        
        hours = int(self.elapsed // 3600)
        remaining_seconds = self.elapsed % 3600
        minutes = int(remaining_seconds // 60)
        seconds = remaining_seconds % 60
        miliseconds = int(10**self.precision * (seconds - int(seconds)))
        elapsed_str = f"{hours:02d}:{minutes:02d}:{int(seconds):02d}.{miliseconds:03d}"

        if self.fancy:
            print(f"\t {AnsiColor.OKGREEN}completed in {AnsiColor.WARNING}{elapsed_str}{AnsiColor.ENDC}")
        else:
            print(f"\t completed in {elapsed_str}")


def get_min_uint_dtype(num: int) -> numpy.dtype:
    """
    Determine the minimum unsigned integer dtype to represent a number.

    Parameters
    ----------
    num : int
        Non-negative integer value.

    Returns
    -------
    numpy.dtype
        Minimum unsigned integer dtype that can represent the number.

    Raises
    ------
    TypeError
        If input is not a non-negative integer.
    OverflowError
        If number exceeds uint64 maximum value.

    Examples
    --------
    >>> get_min_uint_dtype(255)
    dtype('uint8')
    >>> get_min_uint_dtype(70000)
    dtype('uint32')
    """
    if not isinstance(num, (int, numpy.number)) or num < 0:
        raise TypeError("Input must be a non-negative integer")

    # Check in order of size
    for dtype in [numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64]:
        if num <= numpy.iinfo(dtype).max:
            return dtype

    raise OverflowError(f"Number {num} exceeds maximum uint64 value")


def ensure_dir_exists(
    path: Union[str, Path],
    verbose: bool = False,
    parents: bool = True,
) -> Path:
    """
    Ensure a directory exists, creating it if necessary.

    Parameters
    ----------
    path : str or Path
        Directory path to ensure exists.
    verbose : bool, optional
        Whether to print status messages, by default False.
    parents : bool, optional
        Whether to create parent directories, by default True.

    Returns
    -------
    Path
        Absolute path to the directory.

    Raises
    ------
    PermissionError
        If lacking permissions to create directory.
    OSError
        If directory creation fails for other reasons.

    Examples
    --------
    >>> ensure_dir_exists("./data/output")
    PosixPath('/absolute/path/to/data/output')
    """
    path_obj = Path(path).resolve()

    if path_obj.exists():
        if not path_obj.is_dir():
            raise OSError(f"Path exists but is not a directory: {path_obj}")
        if verbose:
            print(f"Directory already exists: {path_obj}")
        return path_obj

    try:
        path_obj.mkdir(parents=parents, exist_ok=True)
        if verbose:
            print(f"Created directory: {path_obj}")
        return path_obj
    except PermissionError as e:
        raise PermissionError(
            f"Permission denied creating directory {path_obj}") from e
    except OSError as e:
        raise OSError(f"Failed to create directory {path_obj}: {e}") from e


def _validate_positive_number(value: float | int, name: str) -> None:
    """
    Validate that a value is numeric and non-negative.

    Parameters
    ----------
    value : int or float
        The value to check.
    name : str
        Name of the variable, used in error messages.

    Raises
    ------
    TypeError
        If `value` is not numeric.
    ValueError
        If `value` is negative.
    """
    if not isinstance(value, (int, float, numpy.number)):
        raise TypeError(f"{name} must be numeric")
    if value < 0:
        raise ValueError(f"{name} must be zero or positive")


def _validate_positive_number_non_zero(value: float | int, name: str) -> None:
    """
    Validate that a value is numeric, positive, and non-zero.

    Parameters
    ----------
    value : int or float
        The value to check.
    name : str
        Name of the variable, used in error messages.

    Raises
    ------
    TypeError
        If `value` is not numeric.
    ValueError
        If `value` is negative or zero.
    """
    _validate_positive_number(value, name)
    if value == 0 or value == 0.:
        raise ValueError(f"{name} must be non-zero")


def _validate_coordinate_array(arr: numpy.ndarray, name: str) -> None:
    """
    Validate that a coordinate array has the correct shape and type.

    Parameters
    ----------
    arr : numpy.ndarray
        The coordinate array to validate. Must have shape `(3,)` or `(n_particles, 3)`.
    name : str
        Name of the variable, used in error messages.

    Raises
    ------
    TypeError
        If `arr` is not a NumPy array.
    ValueError
        If `arr` is empty or has an invalid shape.
    """
    if not isinstance(arr, numpy.ndarray):
        raise TypeError("Input must be a numpy array")

    if arr.shape[0] == 0:
        raise ValueError(f"Input {name} must contain at least one particle")

    if (arr.ndim == 1 and arr.shape[0] != 3) or (arr.ndim == 2 and arr.shape[1] != 3) or \
            len(arr.shape) > 2:
        raise ValueError(
            f"Input {name} must have shape (3,) or (n_particles, 3)")


def _validate_existing_path(path: str | Path) -> None:
    """
    Validate that a given path exists and is a directory.

    Parameters
    ----------
    path : str or Path
        Path to validate. Can be a string or a pathlib.Path object.

    Raises
    ------
    TypeError
        If `path` is not a string or Path instance.
    FileNotFoundError
        If the path does not exist.
    NotADirectoryError
        If the path exists but is not a directory.
    """
    if not isinstance(path, (str, Path)):
        raise TypeError("path must be a string or Path object")

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")


def _validate_boxsize_minisize(boxsize: float | int, minisize: float | int) -> None:
    """Validate that `boxsize` and `minisize` are positive, non-zero numbers
    and that `minisize` does not exceed `boxsize`."""
    # Validate both are positive numbers
    _validate_positive_number_non_zero(boxsize, "boxsize")
    _validate_positive_number_non_zero(minisize, "minisize")

    # Check that minisize is not larger than boxsize
    if minisize > boxsize:
        raise ValueError("minisize cannot be larger than boxsize")


def _validate_mini_box_id(mini_box_id: int | numpy.integer, cells_per_side: int | numpy.integer) -> None:
    """Validate that a mini-box ID is an integer within the valid range for a 
    cubic grid defined by the number of cells per side."""
    # Check mini_box_id is an integer
    if not isinstance(mini_box_id, (int, numpy.integer)):
        raise TypeError("mini_box_id must be an integer")

    # Check mini_box_id is a positive number
    _validate_positive_number(mini_box_id, 'mini_box_id')

    # Check cells_per_side is a positive integer
    if not isinstance(cells_per_side, (int, numpy.integer)):
        raise TypeError("cells_per_side must be a positive integer")

    if cells_per_side is None or cells_per_side <= 0:
        raise ValueError("cells_per_side must be a positive integer")

    # Validate mini_box_id is within valid range
    total_mini_boxes = cells_per_side**3
    if mini_box_id >= total_mini_boxes:
        raise ValueError(
            f"mini_box_id {mini_box_id} exceeds maximum valid ID {total_mini_boxes - 1} "
            f"for grid with {cells_per_side}³ = {total_mini_boxes} mini-boxes"
        )


def _validate_process_objects(
    data: Tuple[list | tuple, list | tuple, list | tuple], 
    n_items: int
) -> None:
    """Validate the structure and contents of process object data."""
    # Check number of items in data must be three
    if not isinstance(data, tuple) or len(data) != 3:
        raise ValueError("data must be a tuple of (arrays, labels, dtypes)")
    
    # Check all items in data are tuples or lists
    arrays, labels, dtypes = data
    if not (isinstance(arrays, (list, tuple)) and
            isinstance(labels, (list, tuple)) and
            isinstance(dtypes, (list, tuple))):
        raise ValueError("data must contain only lists or tuples")

    # Check all items in data are arrays and have the same number of rows
    for i, arr in enumerate(arrays):
        if not isinstance(arr, numpy.ndarray):
            raise ValueError(f"array {i} in data[0] must be a numpy array")
        if arr.shape[0] != n_items:
            raise ValueError(f"array {i} in data[0] must have {n_items} elements")


def _validate_inputs_load(
    mini_box_id: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    padding: float,
) -> None:
    """Validate inputs for load functions."""
    _validate_mini_box_id(mini_box_id, int(numpy.ceil(boxsize / minisize)))
    _validate_boxsize_minisize(boxsize, minisize)
    _validate_existing_path(load_path)
    _validate_positive_number_non_zero(padding, "padding")


def _validate_seed_data(
    seed_data: Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
) -> None:
    """Validate the structure and consistency of seed data inputs."""
    if len(seed_data) != 4:
        raise ValueError("seed_data must be a tuple of 4 elements")

    position, velocity, mass, radius = seed_data

    _validate_coordinate_array(position, "seed positions")
    _validate_coordinate_array(velocity, "seed velocities")

    # Check all inputs have the same number of elements
    n_items = len(position)
    if n_items == 1:
        mass = numpy.asarray(mass)
        radius = numpy.asarray(radius)

    if not (mass.size == radius.size == len(position) == len(velocity)):
        raise ValueError("All elements in seed_data must have the same length")

def peak_rss_mb() -> float:
    """Peak resident memory of this process so far, in MB (linux reports kB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class StageTimer:
    """Records wall time and peak memory for named stages.
    
    >>> timer = StageTimer()
    >>> with timer('load'):
    ...     ...
    >>> timer.stages        # {'load': {'seconds': 1.2, 'peak_rss_mb': 830.0}}

    Repeated stages accumulate their time. `peak_rss_mb` is the process's
    memory high-water mark at the end of the stage (memory peaks never go
    down, so it is "the peak so far", not the stage's own usage).
    
    """
    def __init__(self):
        self.stages = {}

    @contextmanager
    def __call__(self, name):
        start = perf_counter()
        try:
            yield
        finally:
            entry = self.stages.setdefault(name, {'seconds': 0.0, 'peak_rss_mb': 0.0})
            entry['seconds'] += perf_counter() - start
            entry['peak_rss_mb'] = max(entry['peak_rss_mb'], peak_rss_mb())