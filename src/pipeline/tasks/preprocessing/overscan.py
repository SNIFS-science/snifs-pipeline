import numpy as np
from scipy.stats import linregress

from pipeline.tasks.common import Image
from pipeline.tasks.preprocessing.plots import plotted_task


def correct_even_odd_image(image: Image) -> Image:
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
    correction = result.intercept + result.slope * x_values  # type: ignore
    # The original code (overscan.cxx:431,436) subtracts the correction if even
    # and adds it if odd.
    image.data[::2, :] -= correction
    image.data[1::2, :] += correction

    # We create the params to put in headers for historical purposes
    image.header["OEPARAM"] = [result.slope, result.intercept]  # type: ignore
    return image


@plotted_task()
def correct_even_odd(datas: list[Image]) -> list[Image]:
    """Correct the even-odd pattern in the data.

    This is a port of the overscan.cxx OddEvenCorrect function.

    Note that this does not modify the variance, as its effect was 'negligible'."""
    header_key = "OEPARAM"
    output = []
    for data in datas:
        # OEPARAM indicates that this has been done
        if header_key in data.header:
            output.append(data.copy())
            continue
        output.append(correct_even_odd_image(data.copy()))
    return output
