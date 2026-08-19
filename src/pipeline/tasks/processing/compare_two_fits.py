import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from astropy.io import fits
from astropy.visualization import ZScaleInterval, AsinhStretch, ImageNormalize


# FITS_FILE_1 = "C:/Users/gibis/URAP/snifs-pipeline/output/level=preprocessed/deep_skyflat_coadd.fits"
FITS_FILE_1 = "./sky_deep_coadd_uncorrected.fits"
FITS_FILE_2 = "C:/Users/gibis/URAP/snifs-pipeline/src/pipeline/tasks/processing/sky_deep_coadd_group_9_corrected.fits"


def get_user_inputs():
    print("--- FITS Image Difference Viewer ---")
    

    valid_scales = ['linear', 'log', 'sinh']
    scale_choice = input("Select scale (linear, log, sinh) [default: linear]: ").strip().lower()
    if scale_choice not in valid_scales:
        print("Invalid or blank input. Defaulting to 'linear'.")
        scale_choice = 'linear'
        

    zscale_choice = input("Apply ZScale? (Y/N) [default: Y]: ").strip().upper()
    use_zscale = False if zscale_choice == 'N' else True
    
    return scale_choice, use_zscale

def main():
    scale_choice, use_zscale = get_user_inputs()


    data1 = fits.getdata(FITS_FILE_1)
    data2 = fits.getdata(FITS_FILE_2)



    diff_data = data1 - data2
    # print(diff_data.shape)
    crop_choice = input("Apply Crop? (Y/N) [default: N]: ").strip().upper()
    crop = True if crop_choice == 'Y' else False
    if crop:
        diff_data = diff_data[:2500]

    if use_zscale:
        z_vmin, z_vmax = ZScaleInterval().get_limits(diff_data)
    else:
        z_vmin, z_vmax = np.nanmin(diff_data), np.nanmax(diff_data)


    vmax = max(abs(z_vmin), abs(z_vmax))
    vmin = -vmax


    if scale_choice == 'linear':
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        
    elif scale_choice == 'log':
        linthresh = vmax / 100.0 if vmax != 0 else 1e-3
        norm = colors.SymLogNorm(linthresh=linthresh, vmin=vmin, vmax=vmax, base=10)
        
    elif scale_choice == 'sinh':
        norm = ImageNormalize(vmin=vmin, vmax=vmax, stretch=AsinhStretch())


    fig, ax = plt.subplots(figsize=(6, 6))
    
    im = ax.imshow(diff_data, cmap='RdBu', origin='lower', norm=norm)
    

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Difference (Raw Image - Fit group 10)')
    

    title_str = (f"Difference Map\n"
                 f"Scale: {scale_choice.capitalize()} | ZScale: {'Yes' if use_zscale else 'No'}\n"
                 f"vmin: {vmin:.2f} | vmax: {vmax:.2f}")
    ax.set_title(title_str)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()