from pipeline.tasks.augment import augment_science_file
from pipeline.tasks.calibrate import calibrate_with_flats
from pipeline.tasks.cfht_weather import update_cfht_weather
from pipeline.tasks.correct_dichroic import correct_dichoric
from pipeline.tasks.extract_snifs_run_logs import extract_snifs_run_logs
from pipeline.tasks.preprocess import preprocess_exposure
from pipeline.tasks.remove_continuum import remove_continuum

__all__ = [
    "augment_science_file",
    "calibrate_with_flats",
    "preprocess_exposure",
    "correct_dichoric",
    "remove_continuum",
    "update_cfht_weather",
    "extract_snifs_run_logs",
]
