from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy import sparse

from pipeline import settings
from pipeline.common import get_logger
from pipeline.common.plotting_utils import get_all_peaks
from pipeline.resolver.resolver import get_run_id

ALL_PEAKS = get_all_peaks()

TEST_ID = "01"


def plot_refined_spectrum(spec: np.ndarray, other_new_centers: np.ndarray) -> None:
    """Args:
    spec: numpy array of spectrum data
    other_new_centers: numpy array of new center points for the spectrum
    """
    logger = get_logger()

    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return
    plt.plot(spec)
    plt.vlines(other_new_centers, 0, 10e5, color="r", alpha=0.5)
    plt.ylim(10, np.max(spec) * 1.1)
    plt.yscale("log")

    flow_run_id = get_run_id()
    output_location = Path(
        f"C:/Users/gibis/URAP/snifs-pipeline/output/wavelength_arc_fit_{TEST_ID}.webp"
    )  # PUBLIC_PATH_MAP[flow_run_id] / f"wavelength_arc_fit_{flow_run_id}.webp").resolve()
    output_location.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plot to {output_location}")
    plt.savefig(output_location, dpi=600, bbox_inches="tight")
    plt.close()
    output_location.chmod(0o644)  # Make the file readable by everyone

    return


def plot_params(params: np.ndarray) -> None:
    """Args:
        params: numpy array of parameters for the cubic wavelength fit for each spaxel
    Returns:
        None
    """
    logger = get_logger()

    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return
    fig, ax = plt.subplots(2, 2)
    coeff_names = ["x³ coefficient", "x² coefficient", "x¹ coefficient", "constant term"]
    labels = np.arange(1, 226).reshape(15, 15)

    x, y = np.meshgrid(np.arange(15), np.arange(15))
    for i in range(4):
        grid = params[:, i].reshape(15, 15)
        row, col = divmod(i, 2)
        ax_i = ax[row, col]
        sc = ax_i.scatter(x, y, c=grid, s=50)
        # Add labels like in the figure for Yannick's habilitation
        for xi in range(15):
            for yi in range(15):
                ax_i.text(x[xi, yi] + 0.05, y[xi, yi] + 0.05, str(labels[xi, yi]), fontsize=8, rotation=35)
        fig.colorbar(sc, ax=ax_i, label=coeff_names[i])
        ax_i.set_title(coeff_names[i])
        ax_i.set_aspect("equal")

    flow_run_id = get_run_id()
    output_location = Path(
        f"C:/Users/gibis/URAP/snifs-pipeline/output/fit_correction_grid_{TEST_ID}.webp"
    )  # output_location = (PUBLIC_PATH_MAP[flow_run_id] / f"fit_coefficients_grid_{flow_run_id}.webp").resolve()
    output_location.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plot to {output_location}")
    plt.tight_layout()
    plt.savefig(output_location, dpi=600, bbox_inches="tight")
    plt.close()
    output_location.chmod(0o644)  # Make the file readable by everyone

    return


def plot_spectrum(wavelengths: np.ndarray, fluxes: np.ndarray) -> None:
    """Args:
        wavelengths: numpy array of wavelengths
        fluxes: numpy array of flux values
    Returns:
        None
    """
    logger = get_logger()

    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return

    flow_run_id = get_run_id()

    # Flatten the data for plotting
    X = wavelengths.flatten()  # Wavelengths
    Y = np.repeat(np.arange(225), 1499)  # Object indices was something else before not 140000 vs 1499
    C = fluxes.flatten()  # Flux values

    # Avoid log(0) issues — filter out or replace nonpositive values
    mask = C > 1
    X, Y, C = X[mask], Y[mask], C[mask]

    plt.figure(figsize=(12, 6))
    sc = plt.scatter(X, Y, c=C, cmap="viridis", s=2, norm=LogNorm())
    plt.colorbar(sc, label="Flux (log scale)")
    plt.xlabel(r"Wavelength ($\mathrm{\AA}$)")
    plt.ylabel("Spaxel")
    plt.vlines(ALL_PEAKS, -2, 230, color="k", linestyle="--", alpha=0.5)

    plt.title(f"{flow_run_id} Arc: Flux vs Wavelength")
    output_location = Path(
        f"C:/Users/gibis/URAP/snifs-pipeline/output/fit_wavelengths_{TEST_ID}.webp"
    )  # (PUBLIC_PATH_MAP[flow_run_id] / f"fit_wavelengths_{flow_run_id}.webp").resolve()
    output_location.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plot to {output_location}")
    plt.tight_layout()
    plt.savefig(output_location, dpi=600, bbox_inches="tight")
    plt.close()
    output_location.chmod(0o644)  # Make the file readable by everyone

    return


def plot_fitting_check(fitModel: sparse.csr_matrix, imagea: np.ndarray) -> None:
    """Args:
        fitModel: sparse matrix of fitted model data
        imagea: numpy array of actual image data
    Returns:
        None
    """
    logger = get_logger()

    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return

    flow_run_id = get_run_id()

    notbadmodel = fitModel + 9  # to account for readout noise
    difference = imagea - fitModel

    chi2 = np.square(difference) / notbadmodel
    logger.info("Chi^2 = ", np.sum(chi2))

    for i in range(1000, 3000, 600):
        plt.plot(imagea[i, :], label="data")
        plt.plot(fitModel[i, :], label="model")
        plt.plot(difference[i, :], label="data-model")
        plt.fill_between(
            np.arange(imagea[i, :].shape[0]),
            np.sqrt(imagea[i, :] + 9),
            -np.sqrt(imagea[i, :] + 9),
            alpha=0.3,
            color="C2",
        )
        plt.title(f"row {i} Both in first PV")
        plt.legend()
        output_location = Path(
            f"C:/Users/gibis/URAP/snifs-pipeline/output/spaxel_{i}_cut_through_{TEST_ID}.webp"
        )  # (PUBLIC_PATH_MAP[flow_run_id] / f"spaxel_{i}_cut_through_{flow_run_id}.webp").resolve()
        output_location.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving plot to {output_location}")
        plt.tight_layout()
        plt.savefig(output_location, dpi=600, bbox_inches="tight")
        plt.close()
        output_location.chmod(0o644)

    plt.clf()

    fig, ax = plt.subplots(2, 2)
    im1 = ax[0][0].imshow(fitModel, cmap="plasma", aspect="auto")
    fig.colorbar(im1, ax=ax[0][0])
    ax[0][0].set_xlim(900, 1200)
    ax[0][0].set_ylim(3500, 0)
    ax[0][0].set_title("Model with 0-ed spaxel but including the read noise")
    # plt.show()

    im2 = ax[0][1].imshow(imagea, cmap="plasma", aspect="auto")
    fig.colorbar(im2, ax=ax[0][1])
    ax[0][1].set_xlim(900, 1200)
    ax[0][1].set_ylim(3500, 0)
    ax[0][1].set_title("Data")

    im3 = ax[1][0].imshow(difference, cmap="plasma", aspect="auto", norm="symlog")
    fig.colorbar(im3, ax=ax[1][0])

    ax[1][0].set_xlim(900, 1200)
    ax[1][0].set_ylim(3500, 0)
    ax[1][0].set_title("Data -  Model")

    im4 = ax[1][1].imshow(chi2, cmap="plasma", aspect="auto", norm="symlog")
    fig.colorbar(im4, ax=ax[1][1])
    ax[1][1].set_xlim(900, 1200)
    ax[1][1].set_ylim(3500, 0)
    ax[1][1].set_title("Chi2")

    flow_run_id = "01"

    output_location = Path(
        f"C:/Users/gibis/URAP/snifs-pipeline/output/fitter_checking_{TEST_ID}.webp"
    )  # (PUBLIC_PATH_MAP[flow_run_id] / f"fitter_checking_{flow_run_id}.webp").resolve()
    output_location.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving plot to {output_location}")
    plt.tight_layout()
    plt.savefig(output_location, dpi=600, bbox_inches="tight")
    plt.close()
    output_location.chmod(0o644)  # Make the file readable by everyone
    return
