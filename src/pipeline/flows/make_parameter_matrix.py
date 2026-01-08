import pickle
import time

import matplotlib.pyplot as plt  # TODO: remove matplotlib dependency
import numpy as np
from astropy.io import fits
from astropy.io.fits import PrimaryHDU
from scipy import sparse

from pipeline.common.fitting_math__utils import func_4th, pseudo_voigt, quartic

# load the parameterization of the properties of the optis system

PARA = pickle.load(open("refs/poly_smooth_bananas.pkl", "rb"))
MASK = pickle.load(open("refs/mask_properties_4th_deg_polynomial.pkl", "rb"))
DIFF = pickle.load(open("refs/diffraction_properties_4th_deg_polynomial.pkl", "rb"))
ELLI = pickle.load(open("refs/ellipse_properties_raw.pkl", "rb"))
ELL = pickle.load(open("refs/ellipse_properties.pkl", "rb"))

CENTER_CROSS = func_4th((PARA[0], PARA[1]), *MASK[0])
CENTERR_SPEC = func_4th((PARA[0], PARA[1]), *MASK[1])
RADIUS = func_4th((PARA[0], PARA[1]), *MASK[2])
QUAD_SPEC = func_4th((PARA[0], PARA[1]), *DIFF[0])
LIN_SPEC = func_4th((PARA[0], PARA[1]), *DIFF[1])
OFFSET_SPEC = func_4th((PARA[0], PARA[1]), *DIFF[2])
LIN_CROSS = func_4th((PARA[0], PARA[1]), *DIFF[3])
OFFSET_CROSS = func_4th((PARA[0], PARA[1]), *DIFF[4])

ELL_C0 = func_4th((PARA[0], PARA[1]), *ELL[0])
ELL_C1 = func_4th((PARA[0], PARA[1]), *ELL[1])
ELL_A = func_4th((PARA[0], PARA[1]), *ELL[2])
ELL_B = func_4th((PARA[0], PARA[1]), *ELL[3])


# global offsets to the spectrum taken from the cross-correlation of the arc with a reference arc.
xoff = 0
yoff = 0
offsets_cross_disp = np.zeros(225)
offsets_cross_disp[~np.isfinite(offsets_cross_disp)] = 0.0  #### fix for nans in the array

# make a grid as basis of the per spaxel and per wavelength element
x = np.arange(-50, 50, 1)
xv, yv = np.meshgrid(x, x)


z_1st = func_4th((PARA[0], PARA[1]), *PARA[8])
z_2nd = func_4th((PARA[0], PARA[1]), *PARA[9])
z_4th = func_4th((PARA[0], PARA[1]), *PARA[10])

n_cross = (np.log(0.0032) - np.log(6.90e-4)) / (np.log(4.917) - np.log(26.939))
n_spec = (np.log(0.0040) - np.log(8.447e-4)) / (np.log(4.52) - np.log(21.026))

start_time = time.time()

rowindex, columnindex = np.meshgrid(np.arange(0, 2048, 1), np.arange(0, 4096, 1))

fileCalib = "P25_196_027"
print(fileCalib)

# parameters = np.load(f'{fileCalib}/{fileCalib}_translationalShifts.npy')
# wParameters = np.load(f'{fileCalib}/{fileCalib}_widthShifts.npy')

fileCalib = fileCalib + "_"

"""except:
    parameters = np.load('translationalShifts.npy')
    wParameters = np.load('widthShifts.npy') """

# parameters = np.load('translationalShifts.npy')


# TODO: typehint and add docstring
def makeShiftedMat(spaxel: int, offsets: np.ndarray, width: float = 1, oversample_factor: int = 1):
    # offsests is a list of 225 offsets in the cross-dispersion direction

    offsets_cross_disp = offsets
    # offsets_cross_disp[112] = offsets_cross_disp[112]-0.3#0.4661760883435381

    list_huge_matrix = []
    list_crossSums = []

    for spaxel_ID in range(15 * 10, 15 * 11):
        print(spaxel_ID, time.time() - start_time)
        # find the place in the image where to put the spectrum per spaxel
        a0 = int(PARA[4][spaxel_ID] + yoff)
        a1 = int(PARA[5][spaxel_ID] + yoff) + 1
        b0 = int(PARA[2][spaxel_ID] + xoff - 50)
        off = 50
        if b0 < 0:
            off += b0
            b0 = 0

        b1 = int(PARA[3][spaxel_ID] + xoff + 50) + 1

        ### try:
        xsub = np.linspace(0, a1 - a0 - 1, (a1 - a0) * oversample_factor)
        ysub = np.linspace(0, b1 - b0 - 1, (b1 - b0) * oversample_factor)

        xv_sub, yv_sub = np.meshgrid(ysub, xsub)

        # the model should be placed to
        ##########################################################################
        # the spectrum for spaxel with number spaxel_ID can be found in this box.
        # image[a0:a1,b0:b1]
        ##########################################################################

        # we will model 1400 spectral elements for each spaxel in the blue cube
        x0 = np.arange(0, 1400, 1)

        # we need to know the position of the peak with spectral element (wavelength)
        # in the crossdispersion direction on the CCD

        # TODO: check why  the extra 0.5 - 1/1400*x0 is there?
        # curve = 0.5 - 1/1400*x0+poly_4(x0,z_2nd[spaxel_ID], z_1st[spaxel_ID], para[6][spaxel_ID], z_4th[spaxel_ID])
        # +off + offsets_cross_disp[spaxel_ID]
        curve = (
            quartic(x0, z_4th[spaxel_ID], 0, z_2nd[spaxel_ID], z_1st[spaxel_ID], PARA[6][spaxel_ID])
            + off
            + offsets_cross_disp[spaxel_ID]
        )

        widthVals = np.ones(1400)
        # doing the quartic shift thing
        # print(wParameters)
        """
        if spaxel_ID in list(range(0,225)):
            index = spaxel_ID
            print(parameters[index])
            quarticArea = fifth(x0/25, parameters[index][0], parameters[index][1], parameters[index][2]
                            ,parameters[index][3], parameters[index][4],parameters[index][5])
            curve[:] = curve[:] + quarticArea
            #plt.plot(quarticArea,label=spaxel_ID)
            #plt.show()
        """
        """
        if spaxel_ID in list(range(0,225)):
            index = spaxel_ID
            onesCount = 0
            for i in range(len(wParameters[index])):
                #print(wParameters[index])
                if wParameters[index][i] == 1.:
                    onesCount += 1
            if onesCount != 5:
                print("not skipping", spaxel_ID)
                width = quartic(x0/25,wParameters[index][0], wParameters[index][1],
                    wParameters[index][2],wParameters[index][3], wParameters[index][4])
                widthVals = widthVals*width
                print(widthVals)
            #print(x0/17)
            #print("width vals",widthVals)"""

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

            spectral = 0.8 * pseudo_voigt(
                np.abs(xiii), 0, 0.6 * widthVals[spec_element], 1.3 * widthVals[spec_element], 5.4, 0.6
            ) + pseudo_voigt(np.abs(xiii), 0, 1.2, 0.2, -n_spec, 0.1, beta=0)
            crossdis = 0.99 * pseudo_voigt(
                np.abs(yiii), 0, 0.6 * widthVals[spec_element], 1.4 * widthVals[spec_element], 5.2, 0.6
            ) + pseudo_voigt(np.abs(yiii), 0, 1.2, 0.1, -n_cross, 0.1, beta=0, l_off=10)

            model = spectral * crossdis.T * (mask_footprint.T * 1.0)
            model = model / np.max(model)

            testModel = (
                model.reshape((model.shape[0] // oversample_factor, oversample_factor, -1, oversample_factor))
                .sum(axis=3)
                .sum(axis=1)
            )

            # image[a0:a1,b0:b1][c0:c1,d0:d1] = model.T
            crossSum = np.sum(testModel, axis=1)
            if len(crossSum) != 100:
                toAppend = np.zeros(100)
                if spaxel_ID > 1400:
                    toAppend[100 - len(crossSum) :] = crossSum
                else:
                    toAppend[: len(crossSum)] = crossSum
                list_crossSums.append(toAppend)
            else:
                list_crossSums.append(crossSum)

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

            # flat_image = np.concatenate(image)
            s_image = s_image.reshape((1, int(2048 * 4096)))
            # s_image = sparse.csr_matrix(s_image)
            list_huge_matrix.append(s_image)

    # np.save(f"{fileCalib}crossSumFitArc",np.vstack(list_crossSums))

    huge_matrix = sparse.vstack(list_huge_matrix)
    # sparse.save_npz(f'{fileCalib}fit_arc_matrix.npz',huge_matrix)

    return huge_matrix


# TODO: typehint and add docstring
def fit(matrix, dataImage, spectra, spaxel, offset, width=1, dynamic=False):
    start = time.time()

    matrix = matrix.transpose()
    spectra = sparse.csr_matrix(spectra).transpose()

    # calculate the product of matrix and vector, then reshape it to the CCD size of 4096 x 2048 pixels

    shifted_s_image = matrix.dot(spectra)
    shifted_s_image = shifted_s_image.reshape((4096, 2048))
    shifted_image = shifted_s_image.todense()

    # load an example SNIFS file, the file should be preprocessed
    # make a mask for all pixels containing signal from the model
    hdul = fits.open(dataImage)

    # TODO: ask Sam if this is okay to do
    hdu: PrimaryHDU = hdul[0]  # type: ignore
    image: np.ndarray = hdu.data  # type: ignore

    flag = (shifted_image > 0.0) & np.isfinite(image)
    flag = np.array(flag.astype(float))
    imagea = np.where(flag, image, 0.0)

    # bring the image into the right shape for fitting
    flat_image = image.flatten()
    fl = np.array(flat_image.transpose().flat)

    # do the final fit using scipy
    from scipy.sparse.linalg import lsqr

    x, istop, itn, normr = lsqr(matrix, fl)[:4]
    # np.save(f"{fileCalib}fit_arc_vector",x)
    stop = time.time()

    fitModel = matrix.dot(x)
    fitModel = fitModel.reshape((4096, 2048))

    notbadmodel = fitModel + 9  # to account for readout noise
    difference = imagea - fitModel

    heights = np.linspace(0, 4095, 256)  # must be a factor of 4096

    chi2 = np.square(difference) / notbadmodel
    print("Chi^2 = ", np.sum(chi2))

    if dynamic:
        pass

    # calculating the reduced chi^2
    norms = []

    stop = time.time()
    print(stop - start)

    for i in range(len(heights) - 1):  # for every height bin
        # numerator = np.square(difference[int(heights[i]):int(heights[i+1]),:])
        # denominator = notbadmodel

        chi2Sub = chi2[int(heights[i]) : int(heights[i + 1]), :]

        numpix = chi2Sub.shape[0] * chi2Sub.shape[1]  # 133,120 px w/ cutting into 16 I think?
        norms.append(np.sum(chi2Sub) / (numpix + 1))

    fitModel = sparse.csr_matrix(fitModel)

    return norms, chi2, fitModel


offs = 13
# TODO: should maybe switch to arange with spacing 0.2125 for consistency w/-1.7-1.7 w/ 17
offsets = list(np.linspace(-1.7, 1.7, offs))
# 1.9125,3.1875,8

spec = pickle.load(open("/home/anousha/snifs_model/science_spectra.pkl", "rb"))
mono = np.ones((225, 1400))


spectra = np.concatenate(spec[15 * 10 : 15 * 11])  # CHANGE THIS
# spectra = np.concatenate(spec[:])

fig, ax = plt.subplots(3)

os = np.zeros(225)
ws = list(np.linspace(0.9, 1.3, 1))

# TODO: automate this
for spax in [150, 151, 152, 153, 154, 155]:
    errors = []

    for o in offsets:
        for w in [1]:
            os[spax] = o
            testMat = makeShiftedMat(spax, os, width=w, oversample_factor=4)
            # testMat = sparse.load_npz(f'{fileCalib}fit_arc_matrix.npz')

            # errs = fit(testMat,f'refs/EG131_2025/{fileCalib}003_17_B.fits',spectra,spax,o,width=w)
            errs, chi2, fitModel = fit(
                testMat, "refs/deep_skyflat_coadd.fits", spectra, spax, o, width=w, dynamic=False
            )
            # errs, chi2, fitModel = fit(testMat,f'refs/EG131_2025/{fileCalib}004_03_B.fits',
            # spectra,spax,o,width=w,dynamic=False) #arc
            fitModel = sparse.csr_matrix.todense(fitModel) + 9

            """xoff = -1.8
            yoff = 0.35
            testMat = makeShiftedMat(spax,os,width=w,factor=4)
            errs, chi22, fitModel2 = fit(testMat,f'refs/EG131_2025/{fileCalib}004_03_B.fits',
            spectra,spax,o,width=w,dynamic=False) #arc
            fitModel2 = sparse.csr_matrix.todense(fitModel2) + 9


            hdu = fits.open(f'refs/EG131_2025/{fileCalib}004_03_B.fits')
            image = hdu[0].data

            flag = (fitModel > 0.0) & np.isfinite(image)
            flag = np.array(flag.astype(float))
            imagea = np.where(flag, image, 0.0)  # safer than image * flag

            ax[0].imshow(imagea-fitModel)
            ax[0].set_title(f"difference image with no shift: chi^2: {chi2}")

            flag = (fitModel2 > 0.0) & np.isfinite(image)
            flag = np.array(flag.astype(float))
            imagea = np.where(flag, image, 0.0)  # safer than image * flag

            ax[1].imshow(imagea-fitModel2)
            ax[1].set_title(f"difference image with no shift: chi^2: {chi22}")
            ax[2].imshow(fitModel-fitModel2)
            ax[2].set_title("difference between two fit models")

            plt.savefig(f"{fileCalib}FFT_shift_results.png",dpi=300,bbox_inches="tight")"""

            # errs = fit(testMat,f'/Users/anousha/Desktop/SNIFS/model/refs/EG131_2025/{fileCalib}003_17_B.fits',
            # spectra,spax,o,width=w,dynamic=True)
            # errs = fit(testMat,'/Users/anousha/Desktop/SNIFS/model/refs/P22_170_083_003_17_B.fits',
            # spectra,spax,o,width=w,dynamic=True)

            # errs = fit(testMat,'/home/anousha/snifs_model/refs/deep_skyflat_coadd.fits',
            # spectra,spax,o,width=w,dynamic=False)
            print(o, w)
            errors.append(errs)

    print(f"done with fittings for spaxel {spax}")
    # what we want for different offsets

    errors = np.array(errors)
    offsets = np.array(offsets)
    offsets = np.reshape(offsets, (offs, 1))
    data = np.concatenate((offsets, errors), axis=1)

    """
    ## TO APPEND
    try:
        oldData = np.load(f'{fileCalib}shiftError{spax}DanielMethod.npy')
        fullData = np.concatenate((oldData, data))
        np.save(f"{fileCalib}shiftError{spax}DanielMethod",fullData)
    except:
        np.save(f"{fileCalib}shiftErrorNew{spax}DanielMethod",data)
    """

    # np.save("shiftErrorDanielMethod",data)
    np.save(f"twilightShiftError{spax}DanielMethod", data)

    '''

    #what we want for different widths
    errors = np.array(errors)
    offsets = np.array(ws)
    offsets = np.reshape(offsets,(len(ws),1))
    data=np.concatenate((offsets,errors),axis=1)



    """
    try:
        oldData = np.load(f'widthErrorNew{spax}DanielMethod.npy')
        fullData = np.concatenate((oldData, data))
        np.save(f"widthErrorNew{spax}DanielMethod",fullData)
    except:
        np.save(f"widthErrorNew{spax}DanielMethod",data)"""

    #np.save("shiftErrorDanielMethod",data)
    np.save(f"{fileCalib}widthError{spax}DanielMethod",data)
    '''
