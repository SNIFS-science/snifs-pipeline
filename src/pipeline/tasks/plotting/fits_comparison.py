from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from scipy import sparse

from pipeline import settings
from pipeline.common import Image, get_logger, pipeline_task
from pipeline.common.model_params import A0_PARAMS, A1_PARAMS, B0_PARAMS, B1_PARAMS
from pipeline.common.plotting_utils import get_all_peaks
from pipeline.resolver.resolver import PUBLIC_PATH_MAP, get_run_id
from pipeline.tasks.plotting.plots import plot_standalone

from astropy.io import fits

outer_key = 146

a0 = int(A0_PARAMS[int(outer_key)])
a1 = int(A1_PARAMS[int(outer_key)]) + 1
b0 = int(B0_PARAMS[int(outer_key)]) - 50
off = 50
if b0 < 0:
    off += b0
    b0 = 0
b1 = int(B1_PARAMS[int(outer_key)]) + 50 + 1

row_lo, row_hi = a0, a1  # row_range
col_lo, col_hi = (b0 + b1) // 2 - 6, (b0 + b1) // 2 + 6  # col_range
col_indices = np.arange(col_lo, col_hi)


with fits.open('/Users/anousha/Downloads/sky_deep_coadd_group_9_corrected.fits') as hdul:
    alex_corrected_data = hdul[0].data

with fits.open('/Users/anousha/Desktop/SNIFS/model/refs/deep_skyflat_coadd.fits') as hdul:
    science_data = hdul[0].data

with fits.open('/Users/anousha/Desktop/Homework/snifs-pipeline/tester_spaxel_146_iteration_4.fits') as hdul:
    anousha_data = hdul[0].data

alex_cutout = science_data[row_lo:row_hi, col_lo:col_hi] - alex_corrected_data[row_lo:row_hi, col_lo:col_hi]
anousha_cutout = science_data[row_lo:row_hi, col_lo:col_hi] - anousha_data[row_lo:row_hi, col_lo:col_hi]

fig, ax = plt.subplots(3)

vlo, vhi = np.nanpercentile(alex_cutout, [1, 99])
im0 = ax[0].imshow(alex_cutout,norm="symlog",cmap="RdBu_r", aspect="auto", vmin=-vhi,vmax=vhi,origin="lower",)
plt.colorbar(im0,ax=ax[0])

vlo, vhi = np.nanpercentile(anousha_cutout, [1, 99])
im1 = ax[1].imshow(anousha_cutout,norm="symlog",cmap="RdBu_r", aspect="auto",vmin=-vhi,vmax=vhi,origin="lower",)
plt.colorbar(im1,ax=ax[1])

im2 = ax[2].imshow(alex_cutout-anousha_cutout,norm="symlog",cmap="RdBu_r", aspect="auto",origin="lower",)
plt.colorbar(im2,ax=ax[2])

plt.show()
