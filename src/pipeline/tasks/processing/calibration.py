from pipeline.common.image import Image
from pipeline.common.prefect_utils import pipeline_task
from pipeline.tasks.plotting.plots import plot_standalone


@pipeline_task()
@plot_standalone("calibrate_continuum")
def calibrate_continuum(primary: Image, continuum: Image) -> Image:
    return primary.subtract(continuum)


@pipeline_task()
@plot_standalone("calibrate_wavelengths")
def calibrate_wavelengths(primary: Image, arc: Image) -> Image:
    return primary.subtract(arc)
