import pickle
import time

import numpy as np
from astropy.io import fits
from astropy.io.fits import PrimaryHDU
from numpy.polynomial import Polynomial
from scipy import sparse

from pipeline.common.fitting_math__utils import pseudo_voigt
from pipeline.common.log import get_logger
from pipeline.common.model_params import (
    A0_PARAMS,
    A1_PARAMS,
    B0_PARAMS,
    B1_PARAMS,
    ELL_A,
    ELL_B,
    ELL_C0,
    ELL_C1,
    LIN_CROSS,
    LIN_SPEC,
    QUAD_SPEC,
    QUARTIC_LINEAR_PARAMS,
    Z_1ST,
    Z_2ND,
    Z_4TH,
    default_shift_offsets,
    default_width_offsets,
)
from pipeline.resolver.resolver import PUBLIC_PATH_MAP, get_run_id
from pipeline.tasks.plotting.wavelength_arc_calibration_plots import plot_fitting_check

# load the parameterization of the properties of the optics system


# global offsets to the spectrum taken from the cross-correlation of the arc with a reference arc.
xoff = 0
yoff = 0

# make a grid as basis of the per spaxel and per wavelength element
x = np.arange(-50, 50, 1)
xv, yv = np.meshgrid(x, x)

n_cross = (np.log(0.0032) - np.log(6.90e-4)) / (np.log(4.917) - np.log(26.939))
n_spec = (np.log(0.0040) - np.log(8.447e-4)) / (np.log(4.52) - np.log(21.026))

start_time = time.time()

rowindex, columnindex = np.meshgrid(np.arange(0, 2048, 1), np.arange(0, 4096, 1))


def psf_calculation(sum_along_cross_disp: np.ndarray, spaxel_ID: int) -> np.ndarray:
    if len(sum_along_cross_disp) != 100:
        toAppend = np.zeros(100)
        if spaxel_ID > 1400:
            toAppend[100 - len(sum_along_cross_disp) :] = sum_along_cross_disp
        else:
            toAppend[: len(sum_along_cross_disp)] = sum_along_cross_disp
        return toAppend
    return sum_along_cross_disp


# TODO: typehint and add docstring
def make_matrix(
    spaxel: int,
    offsets: np.ndarray = default_shift_offsets,
    widths: np.ndarray = default_width_offsets,
    partial: bool = False,
    oversample_factor: int = 1,
) -> sparse.csr_matrix:
    """
    Args:
        spaxel (int): spaxel number that will be adjusted (shifted or widened)
        offsets (np.ndarray): array of extra offsets in cross-dispersion direction for all spaxels (225,)
            defaults to zeros
        widths (np.ndarray): array of width scaling factors for all spaxels (225,)
            defaults to a multiplier of 1
        partial (bool, optional): whether to compute only a subset of the matrix. Defaults to False.
        oversample_factor (int, optional): oversampling factor for the model. Defaults to 1.
    Returns:
        sparse.csr_matrix: sparse matrix representing the shifted model
    """
    list_huge_matrix = []
    line_profile = []

    if partial:
        grouping = spaxel // 15
        spaxel_range = range(grouping * 15, (grouping + 1) * 15)
    else:
        spaxel_range = range(225)

    for spaxel_ID in spaxel_range:
        print(spaxel_ID, time.time() - start_time)
        # find the place in the image where to put the spectrum per spaxel
        a0 = int(A0_PARAMS[spaxel_ID] + yoff)
        a1 = int(A1_PARAMS[spaxel_ID] + yoff) + 1
        b0 = int(B0_PARAMS[spaxel_ID] + xoff - 50)
        off = 50
        if b0 < 0:
            off += b0
            b0 = 0

        b1 = int(B1_PARAMS[spaxel_ID] + xoff + 50) + 1

        xsub = np.linspace(0, a1 - a0 - 1, (a1 - a0) * oversample_factor)
        ysub = np.linspace(0, b1 - b0 - 1, (b1 - b0) * oversample_factor)

        xv_sub, yv_sub = np.meshgrid(ysub, xsub)

        ##########################################################################
        # the spectrum for spaxel with number spaxel_ID can be found in this box.
        # image[a0:a1,b0:b1]
        ##########################################################################

        # we will model 1400 spectral elements for each spaxel in the blue cube
        x0 = np.arange(0, 1400, 1)

        p = Polynomial([QUARTIC_LINEAR_PARAMS[spaxel_ID], Z_1ST[spaxel_ID], Z_2ND[spaxel_ID], 0, Z_4TH[spaxel_ID]])
        adjustmentP = Polynomial(offsets[spaxel_ID])

        # TODO: make sure the parameters are saved so they cover 0,1400 rather than 50 something
        curve = p(x0) + adjustmentP(x0) + off
        width_polynomial = Polynomial(widths[spaxel_ID])

        for spec_element in range(0, 1400):
            # the per spaxel per wavelength model will be in this box:
            c0 = int(spec_element - 50) * oversample_factor
            if c0 < 0:
                c0 = 0
            c1 = int(spec_element + 50) * oversample_factor  # (c0+100*factor-(factor-1))
            if c1 >= 1400 * oversample_factor:
                c1 = int(1400 * oversample_factor)

            ##########################################################################
            # the monochromatic image for spaxel with number spaxel_ID can be found in this box.
            # image[a0:a1,b0:b1][c0:c1,int(curve[spec_element]-50):int(curve[spec_element]+50)]

            d0 = int(round((curve[spec_element] - 50))) * oversample_factor
            if d0 < 0:
                d0 = 0
            d1 = int(round((curve[spec_element] + 50))) * oversample_factor
            if d1 >= 1400 * oversample_factor:
                d1 = int(1400 * oversample_factor) - 1

            xv_sub_mono = xv_sub[c0:c1, d0:d1]
            yv_sub_mono = yv_sub[c0:c1, d0:d1]

            #################  make the model ###############
            popt = [0, 0]  ## a way to allow for a shift.
            # redefine x and y
            y = yv_sub_mono.T[0] - spec_element
            # print(curve[spec_element],18/1400*spec_element)
            x = xv_sub_mono[0] - curve[spec_element]

            spec_trace = QUAD_SPEC[spaxel_ID] * y**2 + LIN_SPEC[spaxel_ID] * y + popt[0]
            cross_trace = LIN_CROSS[spaxel_ID] * x + popt[1]

            xiii = (xv_sub_mono - curve[spec_element]).T - spec_trace
            yiii = (yv_sub_mono - spec_element) - cross_trace

            mask_footprint = (
                np.sqrt(
                    (yv_sub_mono - spec_element - ELL_C1[spaxel_ID] - popt[1]) ** 2 / ELL_B[spaxel_ID] ** 2
                    + (xv_sub_mono - curve[spec_element] - ELL_C0[spaxel_ID] - popt[0]) ** 2 / ELL_A[spaxel_ID] ** 2
                )
                < 1
            )

            # adjusting the width
            wavelength_dep_width = float(width_polynomial(spec_element))
            spectral = 0.8 * pseudo_voigt(
                np.abs(xiii), 0, 0.6 * wavelength_dep_width, 1.3 * wavelength_dep_width, 5.4, 0.6
            ) + pseudo_voigt(np.abs(xiii), 0, 1.2, 0.2, -n_spec, 0.1, beta=0)
            crossdis = 0.99 * pseudo_voigt(
                np.abs(yiii), 0, 0.6 * wavelength_dep_width, 1.4 * wavelength_dep_width, 5.2, 0.6
            ) + pseudo_voigt(np.abs(yiii), 0, 1.2, 0.1, -n_cross, 0.1, beta=0, l_off=10)

            model = spectral * crossdis.T * (mask_footprint.T * 1.0)
            model = model / np.max(model)

            testModel = (
                model.reshape((model.shape[0] // oversample_factor, oversample_factor, -1, oversample_factor))
                .sum(axis=3)
                .sum(axis=1)
            )

            # image[a0:a1,b0:b1][c0:c1,d0:d1] = model.T
            sum_cross_disp_axis = np.sum(testModel, axis=1)
            line_profile.append(psf_calculation(sum_cross_disp_axis, spaxel_ID))

            c0 = c0 // oversample_factor
            c1 = c1 // oversample_factor
            d0 = d0 // oversample_factor
            d1 = d1 // oversample_factor

            # get the indicies for the sparse matrix
            mask_val = testModel.T > 1e-4

            rowind = rowindex[a0:a1, b0:b1][c0:c1, d0:d1]

            row = rowind[mask_val[: len(rowind), : len(rowind.T)]]
            colind = columnindex[a0:a1, b0:b1][c0:c1, d0:d1]
            col = colind[mask_val[: len(rowind), : len(rowind.T)]]

            data = testModel.T[: len(rowind), : len(rowind.T)][mask_val[: len(rowind), : len(rowind.T)]]
            s_image = sparse.csr_matrix((data, (col, row)), shape=(4096, 2048))

            s_image = s_image.reshape((1, int(2048 * 4096)))
            list_huge_matrix.append(s_image)

    # np.save(f"{fileCalib}crossSumFitArc",np.vstack(line_profile))

    huge_matrix = sparse.vstack(list_huge_matrix)
    # sparse.save_npz(f'{fileCalib}fit_arc_matrix.npz',huge_matrix)

    return huge_matrix  # type: ignore


# TODO: typehint and add docstring
def calculate_residuals(fitModel, imagea, heights) -> np.ndarray:
    norms = []
    notbadmodel = fitModel + 9  # to account for readout noise
    difference = imagea - fitModel
    chi2 = np.square(difference) / notbadmodel

    for i in range(len(heights) - 1):  # for every height bin
        chi2Sub = chi2[int(heights[i]) : int(heights[i + 1]), :]
        numpix = chi2Sub.shape[0] * chi2Sub.shape[1]
        norms.append(np.sum(chi2Sub) / (numpix + 1))
    return np.array(norms)


# TODO: typehint and add docstring
# TODO: make this accept a path rather than a dataImage directly
def fit(matrix: sparse.csr_matrix, dataImage, spectra, spaxel: int, num_height_bins: int = 256) -> np.ndarray:
    assert 4096 % num_height_bins == 0, "num_height_bins must be a factor of 4096"
    heights = np.linspace(0, 4095, num_height_bins)  # must be a factor of 4096

    start = time.time()

    matrix = matrix.transpose()  # type: ignore
    spectra = sparse.csr_matrix(spectra).transpose()

    # calculate the product of matrix and vector, then reshape it to the CCD size of 4096 x 2048 pixels
    shifted_s_image = matrix.dot(spectra)
    shifted_s_image = shifted_s_image.reshape((4096, 2048))
    shifted_image = shifted_s_image.todense()

    # load an example SNIFS file, the file should be preprocessed
    # make a mask for all pixels containing signal from the model
    hdul = fits.open(dataImage)

    hdu: PrimaryHDU = hdul[0]  # type: ignore
    image: np.ndarray = hdu.data  # type: ignore

    flag = (shifted_image > 0.0) & np.isfinite(image)
    flag = np.array(flag.astype(float))
    imagea = np.where(flag, image, 0.0)

    # bring the image into the right shape for fitting
    flat_image = image.flatten()
    fl = np.array(flat_image.transpose().flat)

    assert np.all(np.isfinite(fl)), "b contains NaN or Inf!"

    # do the final fit using scipy
    from scipy.sparse.linalg import lsqr

    x, istop, itn, normr = lsqr(matrix, fl)[:4]
    # np.save(f"{fileCalib}fit_arc_vector",x)
    stop = time.time()

    fitModel = matrix.dot(x)
    fitModel = fitModel.reshape((4096, 2048))

    plot_fitting_check(fitModel, imagea)
    norms = calculate_residuals(fitModel, imagea, heights)

    stop = time.time()
    print(stop - start)

    return np.array(norms)


def save_results(data: np.ndarray, spaxel: int, isTranslationalShift: bool) -> None:
    flow_run_id = get_run_id()
    logger = get_logger()

    if isTranslationalShift:
        shift_type = "translational"
    else:
        shift_type = "width"

    output_location = (PUBLIC_PATH_MAP[flow_run_id] / f"{shift_type}_shift_{spaxel}_{flow_run_id}.npy").resolve()
    output_location.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving {shift_type} shift data to {output_location}")

    np.save(output_location, data)


def shifting_spaxel(
    spaxel: int,
    shift: float,
    isTranslationalShift: bool,
    translational_params: np.ndarray = default_shift_offsets,
    width_params: np.ndarray = default_width_offsets,
    oversample_factor: int = 1,
) -> sparse.csr_matrix:
    os = translational_params
    ws = width_params
    if isTranslationalShift:
        os[spaxel][0] = shift
    else:
        ws[spaxel][0] = shift
    return make_matrix(spaxel, os, widths=ws, partial=True, oversample_factor=oversample_factor)


def repeat_shift_fit(
    spaxels: list[int],
    shifts: list[float],
    isTranslationalShift: bool,
    translational_params: np.ndarray = default_shift_offsets,
    width_params: np.ndarray = default_width_offsets,
    oversample_factor: int = 1,
) -> None:
    assert all((s < 225 and s > 0) for s in spaxels), "spaxel numbers must be between 0 and 224"
    if not isTranslationalShift:
        assert all(shift > 0 for shift in shifts), "all width shifts must be positive"

    for spaxel in spaxels:
        errors = []
        for shift in shifts:
            shifted_matrix = shifting_spaxel(
                spaxel,
                shift,
                isTranslationalShift,
                translational_params=translational_params,
                width_params=width_params,
                oversample_factor=oversample_factor,
            )
            errs = fit(shifted_matrix, "refs/deep_skyflat_coadd.fits", spectra, spaxel)
            errors.append(errs)
        errors = np.array(errors)
        offsets = np.array(shifts)
        offsets = np.reshape(offsets, (-1, 1))
        data = np.concatenate((offsets, errors), axis=1)
        save_results(data, spaxel, isTranslationalShift)


if __name__ == "__main__":
    offsets = list(np.arange(-1.7, 1.7, 0.2125, dtype=float))  # type: ignore
    # TODO: check which file to use and make this a path argument? (idk if that works with pkls)
    spec = pickle.load(open("/home/anousha/snifs_model/science_spectra.pkl", "rb"))

    spectra = np.concatenate(spec[15 * 10 : 15 * 11])  # CHANGE THIS
    # spectra = np.concatenate(spec[:])
    ws = list(np.linspace(0.9, 1.3, 1))

    repeat_shift_fit(list(range(150, 165)), offsets, isTranslationalShift=True)
