from pipeline.common import Image, Section, pipeline_task
from pipeline.tasks.preprocessing.plots import plot

RED_HFFF_LINES = [
    {
        509: 0.9808,
        1020: 0.9751,
        1021: 1.0138,
        2041: 0.9827,
        2042: 0.9281,
        2043: 0.8741,
        2044: 0.9717,
        2045: 0.9678,
        2046: 0.9925,
        3068: 1.0368,
        3069: 0.9918,
    },
    {
        2041: 0.9852,
        2042: 0.9292,
        2043: 0.8764,
        2044: 0.9695,
        2045: 0.9681,
        2556: 1.0117,
        3068: 1.0267,
        3069: 0.9936,
        3580: 1.0541,
    },
]

RED_HFFFF_SIGMA = 0.004


@plot()
@pipeline_task()
def apply_custom_red_flat(image: Image) -> Image:
    # Here we are referencing the CustomFlat function in preprocessor.cxx:1042
    # It's interesting that this function is applied at the end, because there's
    # logic in this function (line 1066) which applies one amplifier at a time and
    # and needs to use the fact that the amplifiers are reconstructed from different
    # directions. Still, I won't move things around and let's just implement the per-amp
    # logic here like normal.
    image = image.copy()

    for i, lines in enumerate(RED_HFFF_LINES):
        amp_section = Section(x_min=i * 1024, x_max=(i + 1) * 1024, y_min=0, y_max=image.data.shape[1])
        amp_data, amp_var = image.get_section(amp_section)

        for y, factor in lines.items():
            amp_var[:, y - 1] = amp_var[:, y - 1] * factor**2 + amp_data[:, y - 1] ** 2 * RED_HFFFF_SIGMA**2
            amp_data[:, y - 1] *= factor

    return image
