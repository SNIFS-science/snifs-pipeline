import numpy as np
from astropy.io.fits import Header

from pipeline.common.log import get_logger


class Headers(dict[str, str | bool | int | float | list[str] | list[int] | list[float]]):
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

    def set_default(self, defaults: dict[str, str | bool | int | float]) -> None:
        """
        Set default values in the header.
        """
        logger = get_logger()
        for key, value in defaults.items():
            if key not in self:
                self[key] = value
                logger.debug(f"Header {key} was not set. Setting to default: {value}")

    @classmethod
    def merge_all(cls, *headers: "Headers") -> "Headers":
        """
        Merge all headers into one.
        """
        logger = get_logger()
        result = Headers()
        for header in headers:
            for key, value in header.items():
                if key not in result:
                    result[key] = value
                else:
                    logger.debug(
                        f"Header {key} already exists. Overwriting old value {result[key]} with new value {value}"
                    )
                    result[key] = value
        return result

    def merge(self, other: "Headers") -> "Headers":
        """
        Merge another header into this one.
        """
        original = self.copy()
        for key, value in other.items():
            original[key] = value
        return Headers(**original)

    def copy(self) -> "Headers":
        return Headers(**self)

    def to_dict(self) -> dict[str, str | bool | int | float | list[str] | list[int] | list[float]]:
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
