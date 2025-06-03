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


def handle_special_red_cosmetics(image: Image, primary_headers: Headers) -> Image:
    """Please don't ever ask me why anything in this function is the way it is."""

    _, ccd_data, ccd_var = image.get_ccd_section()
    # Please don't ask me why this one has a 1. See imagesnifs.cxx:860 and weep with me
    saturate = primary_headers.get_int("SATURAT1")

    bad_xs = [1574, 1579]
    for i, bad_x_initial in enumerate(bad_xs):
        bad_x_index = bad_x_initial
        y_beginning = min(2937, ccd_data.shape[1])  # imagesnifs.cxx:874 and 896
        y_end = 2941 + i  # imagesnifs.cxx:887
        # You could vectorise this... but it only happens for two columns
        for index, value in enumerate(ccd_data[bad_x_index, :]):
            if value > saturate:
                y_beginning = min(y_beginning, index)
                y_end = max(y_end, index + 1)

        # Apparently start one row earlier if needed
        if y_beginning > 0 and ccd_data[bad_x_index, y_beginning] > saturate:
            y_beginning -= 1

        # Again Im not going to vectorise this right away because I'm worried
        # that I don't quite understand what's going on here.
        for y in range(y_beginning, y_end):
            prop, bound = 1, False  # I dont know what prop and bound are meant to be
            # So I'm going to mostly copy the original code without understanding it.
            if ccd_data[bad_x_index, y] < saturate:
                bound = True
                if bad_x_index > 0:
                    guess = 0.5 * (ccd_data[bad_x_index - 1, y] + ccd_data[bad_x_index + 1, y])
                    prop = (ccd_data[bad_x_index, y] - guess) / (saturate - guess)
                else:
                    prop = ccd_data[bad_x_index, y] / saturate
            if ccd_data[bad_x_index, y] > saturate and (
                (y > 1 and ccd_data[bad_x_index, y - 1] < saturate)
                or (y + 1 < ccd_data.shape[1] and ccd_data[bad_x_index, y + 1] < saturate)
            ):
                bound = True

            for dx in range(0, SPECIAL_RED):  # No I dont get this.
                value = ccd_data[bad_x_index - dx - 1, y] - SPECIAL_CORR[dx] * prop
                ccd_data[bad_x_index - dx - 1, y] = value

                # Now adjust the variance
                variance = ccd_var[bad_x_index - dx - 1, y]
                extra = (SPECIAL_CORR[dx] if bound else SPECIAL_VAR[dx]) * prop * SPECIAL_CONSERVATIVE
                ccd_var[bad_x_index - dx - 1, y] = variance + extra**2

        # Now for the line itself?
        # 1st, the quick-clocked part. That is, the beginning of the line = high y as it is flipped

    return image


@plot()
@pipeline_task()
def handle_cosmetics(image: Image, primary_headers: Headers) -> Image:
    """Sets variance to infinity for known bad pixel regions"""
    image = image.copy()
    channel = primary_headers.get_str("CHANNEL")

    # TODO: repair hot zone via SpecialRedCosmtics
    if channel == "R":
        image = handle_special_red_cosmetics(image, primary_headers)

    for bad_section in BAD_PIXELS.get(channel, []):
        image.mask_bad_section(bad_section)

    # TODO: It seems like the CheatCosmetics actually doesnt just mask it out
    # it checks the CCDSEC and makes some form of linear interpolation?

    # TODO flag cosmetics imagesnifs.cxx:719
    # TODO: hot columns bleeding (snifs_const.h)
    # TODO: HFFF Lines
    return image
