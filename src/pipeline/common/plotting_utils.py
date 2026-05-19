import contextlib
from pathlib import Path
from typing import TypedDict

import cmasher as cmr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from pipeline import settings
from pipeline.common import get_logger
from pipeline.common.prefect_utils import create_image_artifact
from pipeline.tasks.plotting.plots import (
    add_callout_rectangle,
    add_colorbar,
    add_midlines,
    add_ticks,
    convert_path_to_url,
    extract_zoom,
)


def find_closest_index(array: np.ndarray, value: float) -> int:
    """Finding the closest index to the given points.

    Args:
        array : The numpy array in which to find the closest index.
        value : The value to which the closest index in the array is to be found.

    Returns:
        int: The index of the element in the array that is closest to the given value.
    """
    idx = np.argmin(np.abs(array - value))
    return int(idx)


def get_all_peaks():
    """Get all the peaks for the blue emission lines.

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


# for spax 07: 59 184
ZOOM_START = (1220, 0)  # (1000, 509 - 25)
ZOOM_SIZE = (200, 200)
ZOOM_END = (ZOOM_START[0] + ZOOM_SIZE[0], ZOOM_START[1] + ZOOM_SIZE[1])
MIDLINE_X_COORD = ZOOM_START[0] + ZOOM_SIZE[0] // 2
MIDLINE_Y_COORD = ZOOM_START[1] + ZOOM_SIZE[1] // 2
MIDLINE_HORIZONTAL_COLOUR = "#146727"
MIDLINE_VERTICAL_COLOUR = "#755028"

# I love the tailwind colours, and you can get them from the tailwind site
# or this fun tool: https://tailscan.com/colors
LINES_X: dict[int, str] = {
    7: "#e4e23b",
    1022: "#9f1239",
    1032: "#d97706",
}
LINES_Y: dict[int, str] = {
    128: "#601d37",
    1536: "#0e7490",
    2048: "#fb7185",
    4090: "#ffafbb",
}
CMAP_DATA = cmr.get_sub_cmap("cmr.torch", 0, 0.95)
CMAP_ZOOM = cmr.rainforest
CMAP_DIFF = cmr.prinsenvlag


def plot_new_to_old_comparison(image_dict, output_path: Path, start: str | None = None) -> None:  # noqa: C901
    """Plot the images in the data store."""
    logger = get_logger()
    if not settings.plot:
        logger.info("Plotting is disabled. Skipping plot generation.")
        return

    title_prefix = "Comparison of New and Old Pipeline Outputs"
    for _, (key, images) in enumerate(image_dict.items()):
        if len(images) < 2:
            logger.warning(f"Skipping {key}: need at least 2 images for comparison")
            continue

        title = f"{title_prefix} - {key} 112 w region"
        num_cols = 3  # 3 image sets (new, old, delta) × 2 columns each (data, variance)
        aspect_ratio = images[0].data.shape[1] / images[0].data.shape[0]

        fig, axes = plt.subplots(
            4,
            num_cols,
            figsize=(num_cols * 1.5 + 7, 16),
            gridspec_kw={"hspace": 0.2, "wspace": 0.2},
            height_ratios=[1.5 * aspect_ratio, 1, 1, 1],
        )
        axes[0, 0].annotate(title, xy=(0, 1.01), xycoords="axes fraction", ha="left", va="bottom", fontsize=10)

        # Process new image (k=0), old image (k=1), and their delta
        image_labels = ["New", "Old", "Delta"]
        for idx, label in enumerate(image_labels):
            k = idx
            ax_data = axes[0, 1 * k]  # , ax_variance = axes[0, 1*k], axes[0, 3+1*k]
            ax_data_zoom = axes[1, 1 * k]  # , ax_variance_zoom = axes[1, 1 * k], axes[1, 3+1*k]
            axxdl = axes[3, 1 * k]  # , axxvl = axes[2, 1 * k], axes[2, 3+1*k]
            axxc = axes[2, 1 * k]  # , axyc = axes[3, 1 * k], axes[3, 3+1*k]

            # Get data and variance
            if idx == 2:  # Delta
                new_data = images[0].data.astype(np.float64)
                new_data[~np.isfinite(new_data)] = np.nan
                old_data = images[1].data.astype(np.float64)
                old_data[~np.isfinite(old_data)] = np.nan
                data = np.round(new_data - old_data, 6)

                new_var = images[0].variance.astype(np.float64)
                new_var[~np.isfinite(new_var)] = np.nan
                old_var = images[1].variance.astype(np.float64)
                old_var[~np.isfinite(old_var)] = np.nan
                variance = np.round(new_var - old_var, 6)

                cmap_data = CMAP_DIFF
            else:
                image = images[idx]
                data = image.data.astype(np.float64)
                data[~np.isfinite(data)] = np.nan
                variance = image.variance.astype(np.float64)
                variance[~np.isfinite(variance)] = np.nan
                cmap_data = CMAP_DATA

            im_kw = {"origin": "lower", "interpolation": "none"}

            vmin, vmax = np.nanpercentile(data, [1, 99])
            imd = ax_data.imshow(data, cmap=cmap_data, aspect="equal", vmin=vmin, vmax=vmax, **im_kw)
            add_colorbar(f"{label} Data", fig, ax_data, imd)

            vmin, vmax = np.nanpercentile(variance, [1, 99])
            # imv = ax_variance.imshow(variance, cmap=cmap_data, aspect="equal", vmin=vmin, vmax=vmax, **im_kw)
            # add_colorbar(f"{label} Variance", fig, ax_variance, imv)

            imdz = ax_data_zoom.imshow(
                extract_zoom(data, zoom_start=ZOOM_START, zoom_size=ZOOM_SIZE), cmap=CMAP_ZOOM, aspect="auto", **im_kw
            )
            add_colorbar(f"Zoomed {label} Data", fig, ax_data_zoom, imdz, height=0.04)

            # imvz = ax_variance_zoom.imshow(extract_zoom(variance), cmap=CMAP_ZOOM, aspect="auto", **im_kw)
            # add_colorbar(f"Zoomed {label} Var", fig, ax_variance_zoom, imvz, height=0.04)
            # Line plots
            kwargs = {"lw": 0.3, "alpha": 0.7}
            y_lim_percentages = [1, 99]
            for ax, sec in [(axxdl, data)]:
                for location, colour in LINES_X.items():
                    ax.plot(sec[location, :], color=colour, **kwargs)
                dx = sec[list(LINES_X.keys()), :]
                ymin, ymax = np.nanpercentile(dx, y_lim_percentages)
                if abs(ymin) < 1e-4:
                    ymin -= 0.02 * (ymax - ymin)
                if abs(ymax) < 1e-4:
                    ymax += 0.02 * (ymax - ymin)
                ax.set_ylim(ymin, ymax)

            # Midlines
            for ax, sec in [(axxc, data)]:  # , (axyc, variance)]:
                with contextlib.suppress(Exception):
                    row = sec[MIDLINE_X_COORD, ZOOM_START[1] : ZOOM_END[1]]
                    ax.step(np.arange(row.size), row, where="post", color=MIDLINE_HORIZONTAL_COLOUR, **kwargs)
                    column = sec[ZOOM_START[0] : ZOOM_END[0], MIDLINE_Y_COORD]
                    ax.step(np.arange(column.size), column, where="post", color=MIDLINE_VERTICAL_COLOUR, **kwargs)
                    combined = np.concatenate((row.flatten(), column.flatten()))
                    ax.set_ylim(*np.nanpercentile(combined, y_lim_percentages))

            # for ax in (ax_data, ax_variance):
            ax = ax_data  # for each image in the first row
            if ax:
                add_callout_rectangle(ax, zoom_start=ZOOM_START, zoom_size=ZOOM_SIZE)
                add_ticks(ax, LINES_X, axis="x", reach=0.02)
                add_ticks(ax, LINES_Y, axis="y", reach=0.05)
                add_midlines(ax, zoom_end=ZOOM_END, zoom_start=ZOOM_START)

            for ax in (ax_data, ax_data_zoom):  # (ax_data, ax_variance, ax_data_zoom, ax_variance_zoom):
                ax.set_xticks([])
                ax.set_yticks([])
            for ax in (axxdl, axxc):  # (axxdl, axxvl, axxc, axyc):
                ax.set_yticks([])
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=3))
                ax.tick_params(axis="y", labelsize=6, labelrotation=90)
                ax.tick_params(axis="both", labelsize=6)
                ax.set_xmargin(0)

            axxc.set_xlabel(f"{label} Data callout", fontsize=8)
            # axyc.set_xlabel(f"{label} Variance callout", fontsize=8)
            axxdl.set_xlabel(f"{label} Data columns", fontsize=8)
            # axxvl.set_xlabel(f"{label} Variance columns", fontsize=8)

        output_location = (output_path / f"detailed_{key}_w_zone.webp").resolve()
        output_location.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving plot to {output_location}")
        fig.savefig(output_location, dpi=600, bbox_inches="tight")
        plt.close(fig)
        output_location.chmod(0o644)  # Make the file readable by everyone
        create_image_artifact(
            image_url=convert_path_to_url(output_location),
            description=title,
            key="detailed-" + key.replace("_", "-"),
        )
