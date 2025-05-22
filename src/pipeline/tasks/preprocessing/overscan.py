import numpy as np
from scipy.ndimage import median_filter
from scipy.stats import linregress

from pipeline.common.log import get_logger
from pipeline.tasks.common import Image, listify
from pipeline.tasks.preprocessing.plots import plotted_task


def correct_even_odd_image(image: Image) -> Image:
    image = image.copy()
    header_key = "OEPARAM"
    logger = get_logger()
    if header_key in image.header:
        logger.info("Image has already had even-odd correction applied.")
        return image
    # TODO: It would be good to actually test this in case the "S->XFirst()" is a 0 or 1
    # TODO: to make sure we're not applying the odd-even the wrong way around.
    # TODO: this needs to be done on the bias section
    data = image.get_bias_section()
    odd_differences = data[:-1:2, :] - data[1::2, :]
    odd_means = np.mean(odd_differences, axis=0).flatten()  # TODO: check axis. This should be y-length (4096 or so)

    # Perform a linear regression on the mean odd differences
    result = linregress(np.arange(odd_means.size), odd_means)

    # At this point we have a linear fit to the odd differences in the bias section.
    # TODO: This part is super confusing to read to. Conceptually I think its just subtract it out
    # TODO: But the indexation is very confusing. I should create a wrapper for plotting
    x_values = np.arange(image.data.shape[1])
    correction = 0.5 * (result.intercept + result.slope * x_values)  # type: ignore
    # The original code (overscan.cxx:431,436) subtracts the correction if even
    # and adds it if odd
    image.data[::2, :] -= correction
    image.data[1::2, :] += correction

    # We create the params to put in headers for historical purposes
    image.header[header_key] = [result.slope, result.intercept]  # type: ignore
    logger.info(f"Applied even-odd correction: slope={result.slope:0.3f}, intercept={result.intercept:0.3f} to image.")  # type: ignore
    return image


correct_even_odd = plotted_task()(listify(correct_even_odd_image))


def add_overscan_variance_image(image: Image) -> Image:
    header_key = "OVSCNOIS"
    logger = get_logger()
    image = image.copy()
    # OVSCNOIS indicates that this has been done
    if header_key in image.header and image.header[header_key]:
        logger.info("Image has already had overscan variance added.")
        return image

    variance = np.var(image.get_bias_section())
    image.header["RDNOISE"] = np.sqrt(variance)
    image.variance += variance
    image.header[header_key] = 1
    logger.info(f"Added overscan variance: {variance:0.3f} to image.")
    return image


add_overscan_variance = plotted_task()(listify(add_overscan_variance_image))


def subtract_offset_image(image: Image) -> Image:
    header_key = "OVSCDONE"
    logger = get_logger()
    image = image.copy()
    if header_key in image.header and image.header[header_key]:
        logger.info("Image has already had overscan offset subtracted.")
        return image

    # There are some header value shenanigans in overscan.cxx:585 that I replicate
    # with minimal understanding.
    if image.header.get_optional_str("OBSTYPE") != "BIAS":  #! TODO: this negation confuses me
        image.header["BIASFRAM"] = 1

    # ComputeLinesMean from overscan.cxx:202 iterates over every Y value
    # in the bias section and sums across X axis to compute the mean
    bias_data = image.get_bias_section()
    mean = np.mean(bias_data, axis=0)

    # To compute the variance, the RMS is loaded from the RDNOISE header value
    # and then divided by the X-length of the array (so one value per Y)
    # NOTE: this is a red herring - the value does not come from the instrument,
    # but RDNOISE is set in the prior AddOverscanVariance function. It's just the average
    # RMS noise of the bias section
    assert "RDNOISE" in image.header, "RDNOISE header value not set. It should be set in Add add_overscan_variance."
    column_variance = image.header.get_float("RDNOISE", 0.0) ** 2 / bias_data.shape[1]

    # Now turning to ImproveLinesMedian in overscan.cxx:296, we have an array of means
    # and variances, and it seems the algorithm uses a median filter on those arrays
    # to estimate things. From OverscanBase, the default window is 5 pixels to either side.
    # There may be some subtlety in the default of scipy's boundary condition.
    window_size = 5 * 2 + 1  # 5 pixels on either side, plus the pixel itself
    medians = median_filter(mean, size=window_size, mode="reflect")
    # Save out the median medians to the header for posterity
    image.header["OVSCMED"] = float(np.median(medians))

    # Now, the variance is trickier. The math is computed as if our means
    # are a square distribution, and as the lines are full correlated, the
    # original algorithm adds 'a small something' to the variance. Apparently
    # this should not worry us too much, because the variance added here is negligible
    # when compared to the readout error. This comes to the original variance / 3 / (N+2)
    # where N are the number of pixels in the window.
    #! TODO: Check in with Greg about the spaxel locations. If we dont have to worry about the
    #! first few pixels, great
    column_variance /= 3 * (window_size + 2)

    # TODO: implement the subtract ramp function
    # Now that we have the medians and the variance, overscan.cxx:83. SubstractRamp is called
    # This algorithm makes a ramp between left and right overscans
    # I note that the first pixel of the medians (due to window effects) is trash and should not be used
    # There's a "line zero" which according to a comment is 1024/2 + 1 + 1024 = 1537.
    # I made a plot of this median data and line zero (docs/plots/overscan_ramp.png) and its a lot
    # more obvious why there's an inflection point when you look at the data.
    # ? Im confused as to where how theres any actual ramp though... I thought it was going to
    # ? make two ramps, one for the LHS and one for the RHS. Instead, on the index1 column,
    # ? the slope is set to the delta from a 1pixel lookahead. Each step forward in the median array
    # ? then sets the slope/intercept based on the prior value... but like... where is the consistent ramp?
    x_length = image.data.shape[0]
    line_zero = int(x_length // 2 + 1 + x_length)
    # below is overscan.cxx:128 and 131
    multiplier = (medians[1:] - medians[:-1]) / x_length
    offset = ((medians[1:] - medians[:-1]) * (x_length - line_zero) / x_length) + medians[:-1]

    # For the first pixel, we want add = value and multipler = 0 (overscan.cxx:108)
    multiplier = np.insert(multiplier, 0, 0)
    offset = np.insert(offset, 0, medians[0])

    # Now the bias section is actually smaller than the full data (its 4096 columns, the full data is 4128)
    # To handle this, we zero pad the multiplier at the end, and duplicate the final offset value
    multiplier = np.pad(multiplier, (0, image.data.shape[1] - len(multiplier)), mode="constant", constant_values=0)
    offset = np.pad(offset, (0, image.data.shape[1] - len(offset)), mode="edge")

    correction = np.arange(x_length)[:, None] @ multiplier[None, :] + offset[None, :]
    image.data -= correction

    # The variance is just a constant value (apart from at the window boundary technically)
    # and so that means that all the differences used to compute the slope are 0.
    #! TODO: Check in with Greg again bout this.

    return image


subtract_offset = plotted_task()(listify(subtract_offset_image))
