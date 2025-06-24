from pipeline.common.prefect_utils import pipeline_task
from pipeline.tasks.common import Image
from pipeline.tasks.preprocessing.plots import plot


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
    return image
