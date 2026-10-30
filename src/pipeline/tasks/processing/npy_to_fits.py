import numpy as np
from astropy.io import fits
#CHANGE HERE TODO
fits_file_name = "24_028_001_001_03_B"
IS_CONTINUUM = False

#URDONE TODO

savename = f"{fits_file_name}_reconstructed"
if IS_CONTINUUM:
    savename += "_cont.npy"
else:
    savename += "_arc.npy"

downloads_dir = r"C:/Users/gibis/URAP/downloads from nersc"
folder_dir = f"C:/Users/gibis/URAP/downloads from nersc/{fits_file_name}/spaxel_fits"
fits_path = f"C:/Users/gibis/URAP/downloads from nersc/{fits_file_name}/{fits_file_name}.fits"



savepath = f"C:/Users/gibis/URAP/downloads from nersc/{fits_file_name}/{savename}"

newsavepath = f"C:/Users/gibis/URAP/downloads from nersc/{fits_file_name}/{fits_file_name}_reconstructed"
if IS_CONTINUUM:
    newsavepath += "_cont.fits"
else:
    newsavepath += "_arc.fits"
data = np.load(savepath)

hdu = fits.PrimaryHDU(data=data)

hdu.writeto(newsavepath, overwrite=True)