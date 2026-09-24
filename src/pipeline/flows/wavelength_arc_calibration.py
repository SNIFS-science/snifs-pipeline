from functools import cached_property
from pathlib import Path

from pydantic import Field

from pipeline.common.image import Image
from pipeline.common.log import get_logger
from pipeline.common.prefect_utils import pipeline_flow
from pipeline.resolver.resolver import FlowConfig, get_run_id
from pipeline.tasks.processing.make_parameter_matrix import extract_arc_vector_and_linespread
from pipeline.tasks.processing.wavelength_arc_calibration import calibrate_wavelength_arc


class WavelengthArcCalibrationConfig(FlowConfig):
    arc_exposure_path: Path = Field(description="Preprocessed ARC exposure file (.asdf).")
    shift_coeff_path: Path = Field(description="Combined shift-coefficient JSON from run_make_parameter_matrix.")
    width_coeff_path: Path = Field(description="Combined width-coefficient JSON from run_make_parameter_matrix.")
    refresh_filestore: bool = Field(default=True)
    spaxels_to_process: list[int] | None = Field(default=None, description="Spaxels to process; None means all 225.")

    @cached_property
    def output_folder(self) -> Path:
        return self.resolver.output_path / f"level=processed/wavelength_arc/flow_run_id={get_run_id()}"

    @cached_property
    def public_folder(self) -> Path:
        return self.resolver.public_path / f"level=processed/wavelength_arc/flow_run_id={get_run_id()}"


@pipeline_flow()
def wavelength_arc_calibration(conf: WavelengthArcCalibrationConfig):
    logger = get_logger()
    conf.initialise_and_log()

    arc_image = Image.from_asdf(conf.arc_exposure_path)
    spaxels = conf.spaxels_to_process if conf.spaxels_to_process is not None else list(range(225))

    arc_vector_path, linespread_path = extract_arc_vector_and_linespread(
        arc_image.data.T,
        spaxels,
        conf.shift_coeff_path,
        conf.width_coeff_path,
        conf.output_folder,
    )
    logger.info(f"Built arc_vector at {arc_vector_path} and linespread at {linespread_path}")

    return calibrate_wavelength_arc(arc_vector_path, linespread_path)


if __name__ == "__main__":
    raw_dir = Path(__file__).parents[3] / "data/level=raw"
    config = WavelengthArcCalibrationConfig(
        arc_exposure_path=raw_dir / "runs/run_id=25_199_028/25_199_028_004_03_B.fits",
        shift_coeff_path=Path("tester_loop_shifts_editable.json"),
        width_coeff_path=Path("tester_loop_widths_editable.json"),
        spaxels_to_process=[117],
    )
    wavelength_arc_calibration(config)
