from conftest import dict_parametrize

from pipeline.tasks.plotting.plots import convert_path_to_url


@dict_parametrize(
    {
        "path with NERSC": {
            "path": "/global/cfs/cdirs/some_project/www/some_file.txt",
            "expected": "https://portal.nersc.gov/cfs/some_project/some_file.txt",
        },
    }
)
def test_convert_path_to_url(path: str, expected: str):
    result = convert_path_to_url(path)
    assert result == expected
