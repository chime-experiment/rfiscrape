"""Utilities functions."""

import base64
import datetime

import dateutil.parser
import numpy as np


def naive_to_utc(dt: datetime.datetime) -> datetime.datetime:
    """Ensure a datetime has a valid timezone by setting naive datetimes to UTC."""
    if dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None:
        return dt

    return dt.replace(tzinfo=datetime.timezone.utc)


def convert_unix(t: int | float | str) -> float:
    """Checks whether a date is a Unix timestamp or ISO8601 compliant

    Parameters
    ----------
    t
        The time to convert.
        if a string, will be evaluated it ISO8601 compliant

    Raises
    ------
    ValueError
        s is invalid type for date
    """
    if isinstance(t, str):
        try:
            dt = dateutil.parser.parse(t)
        except Exception as e:
            raise RuntimeError("Could not parse the passed datetime string.") from e

        return naive_to_utc(dt).timestamp()
    else:
        return float(t)


def numpy_to_json(arr: np.ndarray) -> dict:
    """Take an array and return as a dict that can be json serialised."""

    return {
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr).decode("utf8"),
    }


def json_to_numpy(jdict: dict) -> np.ndarray:
    """Take a JSON dict and decode into a numpy array."""
    try:
        dtype = np.dtype(jdict["dtype"])
        shape = jdict["shape"]
        data_str = jdict["data"]
    except KeyError:
        raise RuntimeError("JSON does not represent a numpy array.")

    try:
        arr = np.frombuffer(base64.b64decode(data_str.encode("utf8")), dtype=dtype)
    except Exception as e:
        raise RuntimeError(
            f"Could not decode base64 string into a valid array of dtype {dtype}"
        ) from e

    try:
        arr = arr.reshape(shape)
    except Exception as e:
        raise RuntimeError(
            f"Could not transform into array of specified shape {shape}"
        ) from e

    return arr
