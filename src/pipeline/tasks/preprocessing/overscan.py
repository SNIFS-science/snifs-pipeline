import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import linregress

from pipeline.common import Image, flag_skip, listify, pipeline_task
from pipeline.tasks.plotting import plot, plot_bias
from pipeline.tasks.plotting.plots import plot_standalone


@plot_bias()
@plot()
@pipeline_task()
@plot_standalone("correct_even_odd")
@listify
@flag_skip("OEPARAM")
def correct_even_odd(image: Image) -> Image:
    """The odd-even effect is touched on in Emmanual Gangler's thesis, section 3.3.2
    which you can find in the docs/pdfs folder in this repository."""
    image = image.copy()
    # TODO: It would be good to actually test this in case the "S->XFirst()" is a 0 or 1
    # TODO: to make sure we're not applying the odd-even the wrong way around.
    # TODO: this needs to be done on the bias section
    _, data, _ = image.get_bias_section()
    data = data.astype(np.float64)

    # NaN out points that are more than 3 sigma away from the global median and 10-90th std
    # TODO: I feel like we'd want to pull this back in because there are outliers that bias the subtraction
    # median = np.median(data)
    # minv, maxv = np.percentile(data, [10, 90])
    # std_dev = np.std(data[(data > minv) & (data < maxv)], axis=0)
    # mask = np.abs(data - median) < 3 * std_dev
    # data[~mask] = np.nan

    odd_differences = data[:-1:2, :] - data[1::2, :]
    odd_means = np.nanmean(odd_differences, axis=0).flatten()
    # fill NaNs with the mean of the odd differences
    odd_means = np.nan_to_num(odd_means, nan=np.nanmean(odd_means))

    # Perform a linear regression on the mean odd differences
    result = linregress(np.arange(odd_means.size), odd_means)
    slope, intercept = result.slope, result.intercept  # type: ignore

    # At this point we have a linear fit to the odd differences in the bias section.
    # TODO: This part is super confusing to read to. Conceptually I think its just subtract it out
    y_values = np.arange(image.data.shape[1], dtype=np.float64)
    correction = 0.5 * (intercept + slope * y_values)  # type: ignore
    # The original code (overscan.cxx:431,436) subtracts the correction if even
    # and adds it if odd
    image.data[:-1:2, :] -= correction
    image.data[1::2, :] += correction

    # We create the params to put in headers for historical purposes
    image.header["OEPARAM"] = [intercept, slope]
    image.add_function_lineage(f"Applied even-odd correction: slope={slope:0.5g}, intercept={intercept:0.4f} to image.")
    return image


@plot_bias()
@pipeline_task()
@listify
@flag_skip("OVSCNOIS")
def add_overscan_variance(image: Image) -> Image:
    image = image.copy()
    _, var, _ = image.get_bias_section()
    variance = np.mean(np.var(var[:, 1:-1], ddof=1, axis=0))
    rdnoise = np.sqrt(variance)
    image.header.set("RDNOISE", rdnoise, metric=True)
    image.variance += variance
    image.add_function_lineage(f"Added overscan variance: {variance:0.3f} to image (aka RDNOISE={rdnoise:0.3f} ADU).")
    return image


@plot_bias()
@plot()
@pipeline_task()
@plot_standalone("subtract_offset")
@listify
@flag_skip("overscan_done")
def subtract_offset(image: Image) -> Image:
    image = image.copy()
    # ComputeLinesMean from overscan.cxx:202 iterates over every Y value
    # in the bias section and sums across X axis to compute the mean
    bias_section, bias_data, _ = image.get_bias_section()
    mean = np.mean(bias_data, axis=0)

    # Now turning to ImproveLinesMedian in overscan.cxx:296, we have an array of means
    # and variances, and the algo uses a median filter on those means.
    window_size = 5 * 2 + 1  # 5 pixels on either side, plus the pixel itself
    offset_centre = median_filter(mean, size=window_size, mode="reflect")

    # Now that we have the medians and the variance, overscan.cxx:83. SubstractRamp is called
    # This algorithm makes a ramp between left and right overscans
    line_length = image.data.shape[0]
    line_zero = 0.5 * (bias_section.x_min + bias_section.x_max + 1)  # Middle of the bias section
    offset_edge = np.insert(offset_centre[1:], offset_centre.size - 1, offset_centre[-1])

    # If interpolating, we're really interpolating from one bias section midpoint to the next.
    # Ie if the midpoint was 80% of the way through, [xxxxxxxMxx] then out lerp (uncaring about points in the
    # the biassec after the midpoint) would be: [0.2 ... 1.0, 1.0, 1.0]
    # This is because the offset_centre is defined from the middle of the bias section, not the edge of the CCD
    # So if the middle of the bias section represents 90% of the way through the row, we'd want want to use
    start = (line_length - line_zero + 1) / line_length
    end = line_length / line_zero
    lerp = np.clip(np.linspace(start, end, line_length), 0, 1)
    index = np.repeat(lerp[:, None], offset_centre.size, axis=1)
    correction = offset_edge * (1 - index) + offset_centre * index

    if correction.shape[1] < image.data.shape[1]:
        # We've got extra rows, fun fun fun.
        start_rows = bias_section.y_min
        end_rows = image.data.shape[1] - bias_section.y_max
        if start_rows > 0:
            correction = np.concatenate([np.repeat(correction[:, 0:1], start_rows, axis=1), correction], axis=1)
        if end_rows > 0:
            correction = np.concatenate([correction, np.repeat(correction[:, -1:], end_rows, axis=1)], axis=1)

    image.data -= correction

    # There are some header value shenanigans in overscan.cxx:585 that I replicate
    # with minimal understanding.
    if image.header.get_optional_str("file_type") != "BIAS":  #! TODO: this negation confuses me
        image.header["bias_frame"] = 1

    # Save out the median medians to the header for posterity
    overscan_median = float(np.median(offset_centre))
    max_overscan = float(np.max(offset_centre))
    # ^ Don't ask me why it also compares to double the first difference.
    image.header.set("overscan_median", overscan_median, metric=True)
    image.header.set("overscan_max", max_overscan, metric=True)
    image.header.set("overscan_done", True)
    image.add_function_lineage(f"Applied overscan correction: median={overscan_median:0.4f}, max={max_overscan:0.4f}")

    return image
