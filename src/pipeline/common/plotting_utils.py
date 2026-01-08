from typing import TypedDict

import numpy as np


def find_closest_index(array: np.ndarray, value: float) -> int:
    """
    Args:
        array : The numpy array in which to find the closest index.
        value : The value to which the closest index in the array is to be found.
    Returns:
        int: The index of the element in the array that is closest to the given value.
    """
    idx = np.argmin(np.abs(array - value))
    return int(idx)


def get_all_peaks():
    """
    Returns:
        list: A list of all the peaks in the blue channel spectrum
              used for wavelength calibration of the arcs.
    """
    return [
        5769.6,
        5460.735,
        5085.822,
        4916,
        4358.328,
        4198.317,
        4158.59,
        4077.837,
        4046.563,
        3906.371,
        3663.279,
        3650.153,
        3610.5077,
        3466.1996,
        3261.0548,
        3131.7,
    ]


class WavelengthSearch(TypedDict):
    line_source: str
    first_fit: bool
    doublet: bool
    pixel_start_search: int | None
    pixel_end_search: int | None


def get_wavelengths_to_fit() -> dict[float, WavelengthSearch]:
    wavelengths_to_fit: dict[float, WavelengthSearch] = {
        5769.6: WavelengthSearch(
            line_source="HgI",
            first_fit=True,
            doublet=True,
            pixel_start_search=300,
            pixel_end_search=390,
        ),
        5460.735: WavelengthSearch(
            line_source="HgI",
            first_fit=True,
            doublet=False,
            pixel_start_search=400,
            pixel_end_search=550,
        ),
        5085.822: WavelengthSearch(
            line_source="CdI",
            first_fit=True,
            doublet=False,
            pixel_start_search=580,
            pixel_end_search=640,
        ),
        4916: WavelengthSearch(
            line_source="HgI",
            first_fit=True,
            doublet=True,
            pixel_start_search=641,
            pixel_end_search=705,
        ),
        4799.912: WavelengthSearch(
            line_source="CdI",
            first_fit=True,
            doublet=False,
            pixel_start_search=705,
            pixel_end_search=770,
        ),
        4358.1: WavelengthSearch(
            line_source="HgI",
            first_fit=True,
            doublet=False,
            pixel_start_search=880,
            pixel_end_search=920,
        ),
        4198.317: WavelengthSearch(
            line_source="ArI",
            first_fit=False,
            doublet=False,
            pixel_start_search=None,
            pixel_end_search=None,
        ),
        4158.59: WavelengthSearch(
            line_source="ArI",
            first_fit=False,
            doublet=False,
            pixel_start_search=None,
            pixel_end_search=None,
        ),
        4077.837: WavelengthSearch(
            line_source="HgI",
            first_fit=False,
            doublet=False,
            pixel_start_search=None,
            pixel_end_search=None,
        ),
        4046.56: WavelengthSearch(
            line_source="HgI",
            first_fit=True,
            doublet=False,
            pixel_start_search=1000,
            pixel_end_search=1100,
        ),
        3906.371: WavelengthSearch(
            line_source="ArI",
            first_fit=False,
            doublet=False,
            pixel_start_search=None,
            pixel_end_search=None,
        ),
        3663.279: WavelengthSearch(
            line_source="ArI",
            first_fit=False,
            doublet=False,
            pixel_start_search=None,
            pixel_end_search=None,
        ),
        3651.3: WavelengthSearch(
            line_source="HgI",
            first_fit=True,
            doublet=False,
            pixel_start_search=1150,
            pixel_end_search=1250,
        ),
        3446.1996: WavelengthSearch(
            line_source="ArI",
            first_fit=False,
            doublet=False,
            pixel_start_search=None,
            pixel_end_search=None,
        ),
        3261.0548: WavelengthSearch(
            line_source="ArI",
            first_fit=False,
            doublet=False,
            pixel_start_search=None,
            pixel_end_search=None,
        ),
        3131.55: WavelengthSearch(
            line_source="HgI",
            first_fit=True,
            doublet=False,
            pixel_start_search=1400,
            pixel_end_search=1448,
        ),
    }
    return wavelengths_to_fit
