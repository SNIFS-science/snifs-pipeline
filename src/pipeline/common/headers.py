from typing import Any, TypeVar

import numpy as np
from astropy.io.fits import Header

from pipeline.common.log import get_logger

VALID_TYPES = TypeVar("VALID_TYPES", str, bool, int, float, list[str], list[int], list[float])


class Headers:
    def __init__(self, **kwargs):
        self._headers = kwargs
        self._metrics: set[str] = set()
        if "_METRICS" in self._headers:
            self._metrics = set(self._headers["_METRICS"])

    def __getitem__(self, key: str) -> VALID_TYPES:
        return self._headers[key]  # type: ignore

    def __setitem__(self, key: str, value: VALID_TYPES) -> None:
        self._headers[key] = value

    def __contains__(self, key: str) -> bool:
        return key in self._headers

    def __len__(self) -> int:
        return len(self._headers)

    def __or__(self, other: "Headers | dict") -> "Headers":
        if isinstance(other, dict):
            other = Headers(**other)
        return Headers.merge_all(self, other)

    def __delitem__(self, name: str) -> None:
        if name in self._headers:
            del self._headers[name]
        else:
            raise AttributeError(f"{name} not found in headers")

    def set(
        self,
        key: str,
        value: VALID_TYPES,
        metric: bool = False,
    ) -> None:
        """
        Set a header key to a value.
        """
        self[key] = value
        if metric:
            self._metrics.add(key)
        self._headers["_METRICS"] = list(self._metrics)

    def get(self, key: str, default: Any = None) -> Any:
        return self._headers.get(key, default)

    def items(self):
        return self._headers.items()

    def get_optional_int(self, key: str, default: int | None = None) -> int | None:
        value = self.get(key)
        if isinstance(value, (int, float)):
            return int(value)
        elif value is None:
            return default
        raise ValueError(f"Key {key} is not an int: {value} has type {type(value)}")

    def get_int(self, key: str, default: int = None) -> int:  # type: ignore
        value = self.get_optional_int(key, default=default)
        assert value is not None, f"Key {key} is not not available in the header"
        return value

    def get_optional_float(self, key: str, default: float | None = None) -> float | None:
        value = self.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            try:
                return float(value)
            except ValueError as err:
                raise ValueError(f"Key {key} is not a float: {value} has type {type(value)}") from err
        elif value is None:
            return default
        raise ValueError(f"Key {key} is not a float: {value} has type {type(value)}")

    def get_float(self, key: str, default: float = None) -> float:  # type: ignore
        value = self.get_optional_float(key, default=default)
        assert value is not None, f"Key {key} is not not available in the header"
        return value

    def get_float_list(self, key: str) -> list[float]:
        value = self.get(key)
        if isinstance(value, list) and all(isinstance(v, (int, float)) for v in value):
            return [float(v) for v in value]
        raise ValueError(f"Key {key} is not a list of floats: {value} has type {type(value)}")

    def get_optional_bool(self, key: str, default: bool | None = None) -> bool | None:
        value = self.get(key)
        if isinstance(value, bool) or (isinstance(value, int) and value in (0, 1)):
            return bool(value)
        elif isinstance(value, str):
            if value.lower() in ("true", "1", "yes"):
                return True
            elif value.lower() in ("false", "0", "no"):
                return False
            raise ValueError(f"Key {key} is not a bool: {value} has type {type(value)}")
        elif isinstance(value, list):
            return len(value) > 0
        elif value is None:
            return default
        raise ValueError(f"Key {key} is not a bool: {value} has type {type(value)}")

    def get_bool(self, key: str, default: str = None) -> bool:  # type: ignore
        value = self.get_optional_bool(key)
        assert value is not None, f"Key {key} is not not available in the header"
        return value

    def get_optional_str(self, key: str, default: str | None = None) -> str | None:
        value = self.get(key)
        if isinstance(value, str):
            return value
        elif value is None:
            return default
        raise ValueError(f"Key {key} is not a str: {value} has type {type(value)}")

    def get_str(self, key: str) -> str:  # type: ignore
        value = self.get_optional_str(key)
        assert value is not None, f"Key {key} is not not available in the header"
        return value

    def set_default(self, defaults: dict[str, VALID_TYPES]) -> None:
        """
        Set default values in the header.
        """
        logger = get_logger()
        for key, value in defaults.items():
            if key not in self._headers:
                self.set(key, value)
                logger.debug(f"Header {key} was not set. Setting to default: {value}")

    def get_metric_keys(self) -> "set[str]":
        """
        Get the keys that are metrics.
        """
        return self._metrics

    def get_metrics(self, lowercase: bool = True) -> dict[str, VALID_TYPES]:
        """
        Get the metrics in the header.
        """
        metrics = {k: v for k, v in self.items() if k in self._metrics}
        if lowercase:
            metrics = {k.lower(): v for k, v in metrics.items()}
        return metrics

    @classmethod
    def merge_all(cls, *headers: "Headers") -> "Headers":
        """
        Merge all headers into one.
        """
        logger = get_logger()
        result = Headers()
        for header in headers:
            for key, value in header.items():
                if key not in result._headers:
                    result[key] = value
                else:
                    logger.debug(
                        f"Header {key} already exists. Overwriting old value {result[key]} with new value {value}"
                    )
                    result[key] = value
            result._metrics = result._metrics.union(header._metrics)
        return result

    def merge(self, other: "Headers") -> "Headers":
        """
        Merge another header into this one.
        """
        original = self._headers.copy()
        for key, value in other.items():
            original[key] = value
        return Headers(**original)

    def copy(self) -> "Headers":
        return Headers(**self._headers)

    def to_dict(self) -> dict[str, VALID_TYPES]:
        """
        Convert the header to a dictionary.
        """
        return {k: v for k, v in self.items() if v is not None}

    def to_astropy_header(self) -> Header:
        """
        Convert the header to an Astropy Header.
        """
        header = Header()
        for key, value in self.items():
            # FITS headers cannot be lists. So we append 1,2,3, etc to the end of the key
            # ALSO fits files cannot have keys longer than 8 characters, so we truncate the key if necessary
            if isinstance(value, list | tuple | set):
                num_digits = int(np.ceil(np.log10(len(value))))
                for i, v in enumerate(value):
                    key = f"{key[: 8 - num_digits]}{i:0{num_digits}d}"
                    header[key] = v
            else:
                header[key[:8]] = value
        return header

    @classmethod
    def from_dict(cls, data: dict[str, str | bool | int | float | list[str] | list[int] | list[float]]) -> "Headers":
        """
        Create a Headers object from a dictionary.
        """
        return Headers(**data)

    @classmethod
    def from_astropy_header(cls, header: Header) -> "Headers":
        return Headers(**{k: v for k, v in sorted(header.items()) if v is not None})
