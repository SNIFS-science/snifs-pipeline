import PyCosmic

from pipeline.common.image import Image
from pipeline.common.prefect_utils import pipeline_task
from pipeline.tasks.plotting.plots import plot, plot_standalone


@plot()
@pipeline_task()
@plot_standalone("remove_cosmic_rays")
def remove_cosmic_rays(image: Image) -> Image:
    image = image.copy()
    out = PyCosmic.det_cosmics(
        image.data,
        gain=1,
        rdnoise=image.header.get_float("RDNOISE"),
        rlim=0.9,
        sigma_det=6,
        fwhm_gauss=[1.5, 1.5],
        iterations=5,
        replace_box=[5, 5],
        replace_error=100,
        verbose=False,  # Have to turn this to false because they're using print statements instead of logging.
    )
    image.data = out._data  # type: ignore
    if out.mask is not None:
        image.add_function_lineage(f"Masked out {out.mask.sum()} pixels as cosmic rays")
        image.header.set("NUM_COSMIC", int(out.mask.sum()), metric=True)
    else:
        image.add_function_lineage("No cosmic rays detected")
    return image
