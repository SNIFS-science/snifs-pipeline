import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import linregress

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task
from pipeline.tasks.common import Image, flag_skip, listify
from pipeline.tasks.preprocessing.plots import plot_bias


@plot_bias()
# @plot()
@pipeline_task()
@listify
@flag_skip("OEPARAM")
def correct_even_odd(image: Image) -> Image:
    """The odd-even effect is touched on in Emmanual Gangler's thesis, section 3.3.2
    which you can find in the docs/pdfs folder in this repository."""
    image = image.copy()
    logger = get_logger()
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

    # slope, intercept = -6.072720355e-06, -1.4087406134
    # slope, intercept = 2.5839584208e-05, 1.0279772404
    # logger.warning("You're still using hardcoded values bruh")

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
    logger.info(f"Applied even-odd correction: slope={slope:0.5g}, intercept={intercept:0.4f} to image.")  # type: ignore
    return image


@plot_bias()
# @plot()
@pipeline_task()
@listify
@flag_skip("OVSCNOIS")
def add_overscan_variance(image: Image) -> Image:
    logger = get_logger()
    image = image.copy()
    _, var, _ = image.get_bias_section()
    variance = np.mean(np.var(var[:, 1:-1], ddof=1, axis=0))
    rdnoise = np.sqrt(variance)
    image.header["RDNOISE"] = rdnoise
    image.variance += variance
    logger.info(f"Added overscan variance: {variance:0.3f} to image (aka RDNOISE={rdnoise:0.3f} ADU).")
    return image


@plot_bias()
# @plot()
@pipeline_task()
@listify
@flag_skip("OVSCDONE")
def subtract_offset(image: Image) -> Image:
    image = image.copy()
    # ComputeLinesMean from overscan.cxx:202 iterates over every Y value
    # in the bias section and sums across X axis to compute the mean
    bias_section, bias_data, _ = image.get_bias_section()
    mean = np.mean(bias_data, axis=0)

    # To compute the variance, the RMS is loaded from the RDNOISE header value
    # and then divided by the X-length of the array (so one value per Y)
    # NOTE: this is a red herring - the value does not come from the instrument,
    # but RDNOISE is set in the prior AddOverscanVariance function. It's just the average
    # RMS noise of the bias section
    # assert "RDNOISE" in image.header, "RDNOISE header value not set. It should be set in Add add_overscan_variance."
    # column_variance = image.header.get_float("RDNOISE", 0.0) ** 2 / bias_data.shape[1]

    # Now turning to ImproveLinesMedian in overscan.cxx:296, we have an array of means
    # and variances, and it seems the algorithm uses a median filter on those arrays
    # to estimate things. From OverscanBase, the default window is 5 pixels to either side.
    # There may be some subtlety in the default of scipy's boundary condition.
    window_size = 5 * 2 + 1  # 5 pixels on either side, plus the pixel itself
    medians = median_filter(mean, size=window_size, mode="reflect")

    # Now, the variance is trickier. The math is computed as if our means
    # are a square distribution, and as the lines are full correlated, the
    # original algorithm adds 'a small something' to the variance. Apparently
    # this should not worry us too much, because the variance added here is negligible
    # when compared to the readout error. This comes to the original variance / 3 / (N+2)
    # where N are the number of pixels in the window.
    #! TODO: Check in with Greg about the spaxel locations. If we dont have to worry about the
    #! first few pixels, great
    # column_variance /= 3 * (window_size + 2)

    # Now that we have the medians and the variance, overscan.cxx:83. SubstractRamp is called
    # This algorithm makes a ramp between left and right overscans
    # I note that the first pixel of the medians (due to window effects) is trash and should not be used
    line_length = image.data.shape[0]
    line_zero = 0.5 * (bias_section.x_min + bias_section.x_max + 1)
    # Correction for chip edges
    offset_edge = np.insert(medians[1:], medians.size - 1, medians[-1])
    offset_centre = medians
    # If interpolating, we're really interpolating from one bias section midpoint to the next.
    # Ie if the midpoint was 80% of the way through, [xxxxxxxMxx] then out lerp (uncaring about points in the
    # the biassec after the midpoint) would be: [0.2 ... 1.0, 1.0, 1.0]
    # This is because the offset_centre is defined from the middle of the bias section, not the edge of the CCD
    # So if the middle of the bias section represents 90% of the way through the row, we'd want want to use
    start = (line_length - line_zero + 1) / line_length
    end = line_length / line_zero
    lerp = np.clip(np.linspace(start, end, line_length), 0, 1)
    index = np.repeat(lerp[:, None], medians.size, axis=1)
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

    # The variance is just a constant value (apart from at the window boundary technically)
    # and so that means that all the differences used to compute the slope are 0.

    # There are some header value shenanigans in overscan.cxx:585 that I replicate
    # with minimal understanding.
    if image.header.get_optional_str("OBSTYPE") != "BIAS":  #! TODO: this negation confuses me
        image.header["BIASFRAM"] = 1

    # Save out the median medians to the header for posterity
    overscan_median = float(np.median(medians))
    max_overscan = float(max(np.max(medians), (2 * medians[0]) - medians[1]))
    # ^ Don't ask me why it also compares to double the first difference.
    image.header["OVSCMED"] = overscan_median
    image.header["OVSCMAX"] = max_overscan
    get_logger().info(f"Applied overscan correction: median={overscan_median:0.4f}, max={max_overscan:0.4f} to image.")

    return image
