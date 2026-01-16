from collections.abc import Callable
from functools import wraps
from typing import Concatenate

from pipeline.common.image import Image
from pipeline.common.log import get_logger


def listify[**P](func: Callable[Concatenate[Image, P], Image]) -> Callable[Concatenate[list[Image], P], list[Image]]:
    """Convert a function that operates on a single Image to one that operates on a list of Images."""

    def inner(images: list[Image], *args, **kwargs) -> list[Image]:
        return [func(image, *args, **kwargs) for image in images]

    inner.__name__ = func.__name__.replace("_image", "")
    return inner


def flag_skip(key: str):
    """Skips a pipeline task (to just return a copy) if the given key is set to True in the Image header."""

    def decorator[**P](func: Callable[P, Image]) -> Callable[P, Image]:
        # The first argument should be an Image instance from which we look at the header.
        @wraps(func)
        def inner(*args: P.args, **kwargs: P.kwargs) -> Image:
            image: Image = args[0]  # type: ignore
            assert isinstance(image, Image), f"First argument must be an Image instance, got {type(image)}"
            flag = image.header.get_optional_bool(key)
            if flag:
                logger = get_logger()
                logger.info(f"Skipping {func.__name__} as the flag {key} is set in the header.")
                return image.copy()

            result = func(*args, **kwargs)  # type: ignore
            if not isinstance(result, Image):
                raise TypeError(f"Function {func.__name__} must return an Image instance, got {type(result)}")
            if not result.header.get_optional_bool(key, default=False):
                result.header[key] = True
            return result

        return inner

    return decorator
