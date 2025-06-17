import numpy as np

from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_task
from pipeline.tasks.common import Headers, Image, Section
from pipeline.tasks.preprocessing.plots import plot

# Reminder that sections are not end-inclusive
BAD_PIXELS = {
    "R": [
        Section(x_min=1016, x_max=1018, y_min=0, y_max=4079),
        Section(x_min=1110, x_max=1111, y_min=0, y_max=3888),
        Section(x_min=1028, x_max=1029, y_min=0, y_max=2032),
        Section(x_min=1031, x_max=1032, y_min=0, y_max=1000),
        Section(x_min=1061, x_max=1062, y_min=0, y_max=620),
        Section(x_min=1576, x_max=1580, y_min=2938, y_max=2944),
    ],
    "Phot": [
        Section(x_min=637, x_max=638, y_min=56, y_max=57),
        Section(x_min=636, x_max=639, y_min=57, y_max=59),
        Section(x_min=639, x_max=640, y_min=58, y_max=59),
        Section(x_min=637, x_max=640, y_min=59, y_max=4096),
    ],
}

# Hot columns bleeding constants from snifs_const.h
SPECIAL_RED = 12
SPECIAL_CORR = [61.3, 12.1, 10.3, 7.1, 5.1, 3.8, 3.4, 2.6, 2.1, 1.8, 1.2, 0.5]
SPECIAL_VAR = [14.6, 5.4, 5.4, 5.4, 1.7, 0.8, 0.8, 0.8, 0.6, 0.5, 0.5, 0.5]

SPECIAL_CONSERVATIVE = 2
SPECIAL_ERROR_FAST = 0.18


@plot()
@pipeline_task()
def handle_special_red_cosmetics(image: Image, primary_headers: Headers) -> Image:  # noqa: C901
    """Please don't ever ask me why anything in this function is the way it is."""
    logger = get_logger()
    ccd_section, ccd_data, ccd_var = image.get_ccd_section()
    # Please don't ask me why this one has a 1. See imagesnifs.cxx:860 and weep with me
    saturate = primary_headers.get_float("SATURAT1", 0.0)

    bad_xs = [1574, 1579]
    for i, bad_x_initial in enumerate(bad_xs):
        # In imagesnifs.cxx:866 they subtract the ccd section. So the bad_xs correspond
        # to the index of the entire data array, not the ccd section.
        # And they then translate into the ccd section index.
        ix = bad_x_initial - ccd_section.x_min
        if ix < 0 or ix >= ccd_data.shape[0]:
            logger.warning("Unable to apply special red cosmetics to image. Bad x index out of bounds.")
            return image
        # Same thing with the y section
        y_beg = min(2937 - ccd_section.y_min, ccd_data.shape[1])  # imagesnifs.cxx:874 and 896
        y_end = 2941 + i - ccd_section.y_min  # imagesnifs.cxx:887

        # You could vectorise this... but it only happens for two columns
        for index, value in enumerate(ccd_data[ix, :]):
            if value > saturate:
                y_beg = min(y_beg, index)
                y_end = max(y_end, index + 1)

        # Apparently start one row earlier if needed
        if y_beg > 0 and ccd_data[ix, y_beg] > saturate:
            y_beg -= 1

        # Again Im not going to vectorise this right away because I'm worried
        # that I don't quite understand what's going on here.
        for y in range(y_beg, y_end):
            prop, bound = 1, False  # I dont know what prop and bound are meant to be
            # So I'm going to mostly copy the original code without understanding it.
            if ccd_data[ix, y] < saturate:
                bound = True
                if ix > 0:
                    guess = 0.5 * (ccd_data[ix - 1, y] + ccd_data[ix + 1, y])
                    prop = (ccd_data[ix, y] - guess) / (saturate - guess)
                else:
                    prop = ccd_data[ix, y] / saturate
            if ccd_data[ix, y] > saturate and (
                (y > 1 and ccd_data[ix, y - 1] < saturate)
                or (y + 1 < ccd_data.shape[1] and ccd_data[ix, y + 1] < saturate)
            ):
                bound = True

            for dx in range(0, SPECIAL_RED):  # No I dont get this.
                value = ccd_data[ix - dx - 1, y] - SPECIAL_CORR[dx] * prop
                ccd_data[ix - dx - 1, y] = value

                # Now adjust the variance
                variance = ccd_var[ix - dx - 1, y]
                extra = (SPECIAL_CORR[dx] if bound else SPECIAL_VAR[dx]) * prop * SPECIAL_CONSERVATIVE
                ccd_var[ix - dx - 1, y] = variance + extra**2

        # Now for the line itself? (imagesnifs.cxx:943)
        # 1st, the quick-clocked part. That is, the beginning of the line = high y as it is flipped
        # SAM: So this is applying to y values LARGER than y_end, not the y_beginning -> y_end range
        k_last, k_height = 4087, 41
        ground_data, _ = image.get_section(Section(x_min=ix - 4, x_max=ix + 4, y_min=k_last - k_height, y_max=k_last))
        line_data, _ = image.get_section(Section(x_min=ix, x_max=ix, y_min=k_last - k_height, y_max=k_last))
        correction = np.median(ground_data) - np.median(line_data)
        # The correction actually goes from y_end to the end of the data, not y_beginning to y_end
        ccd_data[ix, y_end + 1 :] += correction
        # From imgagesnifs.cxx:960:
        #   we neglect the scaling of +0.04% effect on the variance.
        #   but there is a systematic uncertainty we take into account.
        #   called kSpecialErrorFast [the computation is however NOT accurate for rasters]
        ramp = np.linspace((y_end + 1) / ccd_data.shape[1], 1, ccd_data.shape[1] - (y_end + 1), endpoint=False)
        ccd_var[ix, y_end:] += (ramp * correction * SPECIAL_ERROR_FAST * SPECIAL_CONSERVATIVE) ** 2
        image.header["CORRLOW"] = True  # In the original code this the correction... but also overwrites itself

        # imagesnifs.cxx:974 -> The medium part : assume we lost all the information
        # SAM: This is the correction to the y_beginning to y_end range.
        # From what I can tell, we really just assume that x-value (column) is gone
        # and fill it with the average of the lines to either side.
        # TODO: I should breakpoint here and validate that these lines are actually bad.
        # TODO: Im worried about a one-off index meaning instead of fixing a problem, I create a new one.
        ccd_data[ix, y_beg:y_end] = 0.5 * (ccd_data[ix - 1, y_beg:y_end] + ccd_data[ix + 1, y_beg:y_end])
        ccd_var[ix, y_beg:y_end] = np.inf

        # From imagesnifs.cxx:987:
        #   the hot part of the line is difficult to handle.
        #   the function to fit is p0+p1/(x-p2), where p2 is very close to ybeg.
        #   but the best estimate we can get for p0 comes from the "end" of the line
        #   that is, the beginning once it is flipped.
        #   we know that an estimate using the mean of left and right makes a
        #   systematic error, and we apply it :
        #   1.0*the mean (measurment on the data from continnum lamp)
        #
        # SAM: This is the correction from 0 to y_beg. It seems to be just taking a
        # median on the difference to the left+right column average.
        means = 0.5 * (ccd_data[ix - 1, y_beg:y_end] + ccd_data[ix + 1, y_beg:y_end])
        differences = ccd_data[ix, y_beg:y_end] - means
        correction = np.median(differences)
        # Before applying this correction, there are some checks that the old code runs (line 1014)
        # Its actually just checking to see if the correction is less than 1sigma, with sigma being the
        # ...
        # ...
        # wait.
        # Its doing all this logic to set the value correctly...
        # But then on line 1022 it sets the variance to 1e31 anyway
        # SO WHAT IS THE POINT OF ALL THIS?
        # Fine. I'm simplifying. No fancy checks using the variance only to blow it up. Use path on 1019
        ccd_data[ix, :y_beg] = means - correction
    return image


# @plot()
@pipeline_task()
def cheat_cosmetics(image: Image, channel: str) -> Image:
    image = image.copy()
    # Turns out the bad sections are relative to the ccd section
    # (notice the ccdSec.XFirst() subtraction rather than sec.XFist() in imagesnifs.cxx:762)
    ccd_section, _, _ = image.get_ccd_section()

    for bad_section in BAD_PIXELS.get(channel, []):
        sec = bad_section - ccd_section

        # This masking is whats in FlagCosmetics in imagesnifs.cxx:762
        image.mask_bad_section(sec)

        # But they also replace the data in CheatCosmetics in imagesnifs.cxx:769
        # with a linear interpolation in the x direction for each y value
        for y in range(sec.y_min, sec.y_max):
            intercept = 0
            if sec.x_min > 0:
                intercept = image.data[sec.x_min - 1, y]
            slope = 0
            if sec.x_max < image.data.shape[0] - 1:
                slope = (image.data[sec.x_max, y] - intercept) / (sec.x_max - sec.x_min + 1)
            fill_values = intercept + slope * np.arange(sec.x_min, sec.x_max)
            image.data[sec.x_min : sec.x_max, y] = fill_values

    return image
