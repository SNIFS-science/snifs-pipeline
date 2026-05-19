from pathlib import Path

import numpy as np
from numpy.polynomial import Polynomial
from scipy.optimize import curve_fit

from pipeline.common.fitting_math__utils import shifted_cosh
from pipeline.common.plotting_utils import plot_new_to_old_comparison
from pipeline.resolver.resolver import get_run_id


def find_ideal_shift(spaxel_ID: int, is_translational_shift: bool) -> tuple[np.ndarray, int, int]:
    flow_run_id = get_run_id()

    if is_translational_shift:
        shift_type = "translational"
    else:
        shift_type = "width"
    data = np.load(f"{shift_type}_shift_{spaxel_ID}_{flow_run_id}.npy")

    xPoints = data[:, 0]

    # TODO: come up with a more robust way to set the bounds
    lower = (16 * (spaxel_ID % 15 - 5) + 150) // 2
    upper = (16 * (spaxel_ID % 15 - 5) + 265) // 2

    minima = []
    for y in range(lower, upper):
        yPoints = data[:, y]

        xVals = list(np.linspace(-1.5, 1.51, 51))
        yVals = []

        # p = np.polyfit(xPoints[:],yPoints[:],2)
        try:
            p, _ = curve_fit(shifted_cosh, xPoints, yPoints)

            for x in xVals:
                yVals.append(shifted_cosh(x, *p))
            if p[0] < 0:
                minima.append(np.nan)
            else:
                minima.append(p[2])
        except RuntimeError:
            try:
                p = np.polyfit(xPoints[:], yPoints[:], 2)
            except np.exceptions.RankWarning:
                p = [-1, -1, -1]

            for x in xVals:
                # why do they do this the reverse order compared to np.Polynomial?
                yVals.append(p[2] + p[1] * x + p[0] * x**2)
            if p[0] < 0:
                minima.append(np.nan)
            else:
                minima.append(-p[1] / (2 * p[0]))
    return np.array(minima), upper, lower


def shift_and_save(spaxel_list: list[int], is_translational_shift: bool, polynomial_order: int = 5) -> None:
    oldShift = np.zeros((225, 6))
    flow_run_id = get_run_id()

    for spaxel_ID in spaxel_list:
        minima, upper, lower = find_ideal_shift(spaxel_ID, is_translational_shift)
        xRange = np.array(list(np.arange(lower, upper)))
        scaling_factor = xRange.size / 1400  # to make sure coeffients appropriately scale
        xRange = xRange * scaling_factor
        med = np.nanmedian(minima)
        mask = (~np.isnan(minima)) & (np.abs(minima - med) <= 0.5 * np.abs(med))
        poly_object = Polynomial.basis(polynomial_order)
        params, _ = curve_fit(poly_object, xRange[mask], np.array(minima)[mask])
        oldShift[spaxel_ID] = params
    if is_translational_shift:
        shift_type = "translational"
    else:
        shift_type = "width"
    np.save(f"{shift_type}_shifts_{flow_run_id}.npy", oldShift)


if __name__ == "__main__":
    # import os

    from astropy.io import fits

    from pipeline.common import Headers, Image

    # directory = "/Users/anousha/Desktop/preprocessed"
    directory = "/Users/anousha/Desktop/SNIFS"

    image_dict = {}
    for file in [
        # "generated_image_no_oversample.fits",
        # "model_generated_image.fits",
        # "model_generated_image_no_oversample.fits",
        # "adjustment_point_05.fits",
        # "point_four_minus05.fits",
        # "point_four_minus1.fits",
        # "fit_113_for_offset_-01_chi_3313727291872.fits",
        # "fit_113_for_offset_01_chi_3322373886127.fits",
        # "fit_113_for_offset_03_chi_3333953513280.fits",
        # "fit_113_for_offset_05_chi_3348638484797.fits",
        # "112_chi2_21554478462.fits",
        # "112_c-stat_21551997431.fits",
        # "L1_oversampled.fits",
        # "L1_not_oversampled.fits",
        "final_model_12.fits",
        "final_model_113.fits",
        # "112_removed_53642519180.fits",
        # "112_L1_21553366393.fits",
        # "112_NCC_21601163078.fits",
        # "112_SSIM_21553439105.fits",
        # "112_fRMS_21735928431.fits",
    ]:
        try:
            key = file[:-5]
            images = []
            with fits.open(directory + "/fromTitan/" + file) as hdul:
                data = Image(
                    data=hdul[0].data,  # type: ignore
                    header=Headers.from_astropy_header(hdul[0].header),  # type: ignore
                    variance=np.zeros_like(hdul[0].data),  # hdul[1].data.T,  # type: ignore
                )
                images.append(data)
            with fits.open(directory + "/model/" + "refs/deep_skyflat_coadd.fits") as hdul:
                data = Image(
                    data=hdul[0].data,  # type: ignore
                    header=Headers.from_astropy_header(hdul[0].header),  # type: ignore
                    variance=np.zeros_like(hdul[0].data),  # hdul[1].data,  # type: ignore
                )
                images.append(data)
            image_dict[key] = images
            print("successfully loaded ", key)
        except FileNotFoundError:
            print("could not find ", key)
            continue

    """
    for file in os.listdir(directory+"/newPipeline"):
        if ".DS" not in file:
            try:
                key = file[1:15]
                images = []
                with fits.open(directory+"/newPipeline/"+file) as hdul:
                    data = Image(
                        data=hdul[0].data.T,  # type: ignore
                    header=Headers.from_astropy_header(hdul[0].header),  # type: ignore
                    variance=hdul[1].data.T,  # type: ignore
                    )
                    images.append(data)
                with fits.open(directory+"/oldPipeline/P"+file[1:]) as hdul:
                    data = Image(
                        data=hdul[0].data,  # type: ignore
                    header=Headers.from_astropy_header(hdul[0].header),  # type: ignore
                    variance=hdul[1].data,  # type: ignore
                    )
                    images.append(data)
                image_dict[key] = images
                print("successfully loaded ", key)
            except FileNotFoundError:
                print("could not find ", key)
                continue """
    plot_new_to_old_comparison(image_dict, Path("./comparison_plots"))
    # spaxel_numbers = list(range(0,225))
    # shift_and_save(spaxel_numbers,is_translational_shift=True)
    # shift_and_save(spaxel_numbers,is_translational_shift=False)
