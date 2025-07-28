import pytest


def dict_parametrize(data, **kwargs):
    args = list(next(iter(data.values())).keys())
    formatted_data = [[item[a] for a in args] for item in data.values()]
    ids = list(data.keys())
    return pytest.mark.parametrize(args, formatted_data, ids=ids, **kwargs)
