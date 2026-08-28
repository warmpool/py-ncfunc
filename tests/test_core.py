import os
import warnings
from pathlib import Path
from typing import Any

import datenum as dn
import netCDF4 as nc
import numpy as np
import pytest

import ncfunc as ncf
from ncfunc import core
from tests import nctools as nct

TEST_FILE = "tests/ersst_2022-2024.nc"
TEST_FILE_LINK = "../ncplot/tests/data/file1.nc"


@pytest.fixture
def tmp_nc(tmp_path: Path) -> Path:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    """Create a temporary NetCDF file with sample data."""
    path = tmp_path / "test.nc"
    with nc.Dataset(path, "w") as h:
        h.createDimension("x", 5)
        h.createDimension("y", 3)
        h.createDimension("t", 4)

        h.createVariable("lon", np.float64, ("x",))
        h["lon"][:] = np.arange(5)
        h.createVariable("lat", np.float64, ("y",))
        h["lat"][:] = np.arange(3)
        h.createVariable("time", np.float64, ("t",))
        h["time"][:] = np.arange(4)

        var = h.createVariable("data", np.float32, ("t", "y", "x"))
        var[:] = np.random.rand(4, 3, 5)

        h["time"].units = "days since 2020-01-01"
        h["time"].calendar = "standard"
        h.setncattr("title", "test file")
    return path


# ---- helpers ---
def test_abort_missing_file() -> None:
    with pytest.raises(FileNotFoundError):
        core.DatasetMeta("no-such-path")


def test_meta_data() -> None:
    ds = core.DatasetMeta(TEST_FILE)
    assert tuple(v for v in ds.variables) == ("time", "lon", "lat", "sst", "ssta")

    assert ds.mtime == 1785817424.361744

    assert ds.attributes["creator_name"] == "Boyin Huang"
    assert ds.attributes["creator_email"] == "boyin.huang@noaa.gov"
    assert ds.attributes["creator_type"] == "group"

    assert ds.variables["time"].dimensions == ("time",)
    assert ds.variables["lat"].dimensions == ("lat",)
    assert ds.variables["lon"].dimensions == ("lon",)
    assert ds.variables["sst"].dimensions == ("time", "lat", "lon")
    assert ds.variables["ssta"].dimensions == ("time", "lat", "lon")

    assert ds.variables["lon"].shape == (240,)
    assert ds.variables["sst"].shape == (36, 121, 240)

    assert ds.variables["sst"].attributes == {
        "standard_name": "sea_surface_temperature",
        "long_name": "Extended reconstructed sea surface temperature",
        "units": "degree_C",
        "_FillValue": np.float32(-999.0),
        "missing_value": np.float32(-999.0),
        "cell_methods": "lev: mean",
    }
    assert ds.variables["time"].attributes == {
        "standard_name": "time",
        "long_name": "Time",
        "units": "days since 1850-01-15 00:00",
        "calendar": "gregorian",
        "axis": "T",
    }


def test_cache() -> None:
    ds1 = core.DatasetMeta.try_from_cache(TEST_FILE)
    assert len(core.DatasetMeta.cached_registry) == 1
    ds2 = core.DatasetMeta.try_from_cache(TEST_FILE, cache_only=True)
    assert len(core.DatasetMeta.cached_registry) == 1
    assert ds1 == ds2
    core.DatasetMeta.try_from_cache(TEST_FILE)
    assert len(core.DatasetMeta.cached_registry) == 1
    core.DatasetMeta.try_from_cache(TEST_FILE_LINK)
    assert len(core.DatasetMeta.cached_registry) == 1


# ---- meta data queries ----
def test_var_names() -> None:
    assert ncf.var_names(TEST_FILE) == ("time", "lon", "lat", "sst", "ssta")


def test_dim_names() -> None:
    assert ncf.dim_names(TEST_FILE, "ssta") == ("time", "lat", "lon")
    assert ncf.dim_names(TEST_FILE, "sst") == ("time", "lat", "lon")
    assert ncf.dim_names(TEST_FILE, "lon") == ("lon",)

    with pytest.raises(KeyError):
        ncf.dim_names(TEST_FILE, "no-such-variable")


def test_shape() -> None:
    assert ncf.shape(TEST_FILE, "lon") == (240,)
    assert ncf.shape(TEST_FILE, "lat") == (121,)
    assert ncf.shape(TEST_FILE, "time") == (36,)
    assert ncf.shape(TEST_FILE, "sst") == (36, 121, 240)
    assert ncf.shape(TEST_FILE, "ssta") == (36, 121, 240)


def test_ndim() -> None:
    assert ncf.ndim(TEST_FILE, "lon") == 1
    assert ncf.ndim(TEST_FILE, "lat") == 1
    assert ncf.ndim(TEST_FILE, "time") == 1
    assert ncf.ndim(TEST_FILE, "sst") == 3
    assert ncf.ndim(TEST_FILE, "ssta") == 3


def test_attr_names() -> None:
    assert ncf.attr_names(TEST_FILE, "sst") == (
        "standard_name",
        "long_name",
        "units",
        "_FillValue",
        "missing_value",
        "cell_methods",
    )

    root_attrs = ncf.attr_names(TEST_FILE, "/")

    for some_attr in ("creator_name", "creator_email", "creator_type"):
        assert some_attr in root_attrs


def test_attr_val() -> None:
    def sub_test(path: str, var_name: str, gold: dict[str, Any]) -> None:
        for attr_name, attr_val in gold.items():
            assert ncf.attr_val(path, var_name, attr_name) == attr_val

    sub_test(
        TEST_FILE,
        "time",
        gold={
            "standard_name": "time",
            "long_name": "Time",
            "units": "days since 1850-01-15 00:00",
            "calendar": "gregorian",
            "axis": "T",
        },
    )

    sub_test(
        TEST_FILE,
        "/",
        gold={
            "creator_name": "Boyin Huang",
            "creator_email": "boyin.huang@noaa.gov",
            "creator_type": "group",
        },
    )


def test_var_name_includes() -> None:
    assert ncf.core.var_names_include(TEST_FILE, ("time", "date"), (1,)) == ("time",)

    assert ncf.core.var_names_include(TEST_FILE, ("sst",), (3,), (1, 2)) == (
        "sst",
        "ssta",
    )

    with pytest.raises(ValueError):
        ncf.core.var_names_include(TEST_FILE, ("sst",), (1,))

    with pytest.raises(ValueError):
        ncf.core.var_names_include(TEST_FILE, ("sst",), (3,))


def test_attr_name_includes() -> None:
    assert ncf.core.attr_names_include(TEST_FILE, "time", ("unit"), (1,)) == ("units",)
    assert ncf.core.attr_names_include(TEST_FILE, "time", ("name"), (2,)) == (
        "standard_name",
        "long_name",
    )


# ---- read test ----
def test_read() -> None:
    with pytest.raises(ValueError):
        ncf.read(TEST_FILE, "sst", (slice(0), slice(0)))

    with pytest.raises(ValueError):
        ncf.read(TEST_FILE, "lon", (slice(0), slice(0)))

    for var_name in ncf.var_names(TEST_FILE):
        assert np.array_equal(
            ncf.read(TEST_FILE, var_name), nct.read(TEST_FILE, var_name)
        )

    slices = (slice(5, 10), slice(10, 30), slice(-5))
    assert np.array_equal(
        ncf.read(TEST_FILE, "sst", slices),
        nct.ncread(TEST_FILE, "sst", list(slices)),
    )


def test_read_time() -> None:
    time = ncf.read_time(TEST_FILE)
    ymd = dn.to_ymd(time)[:3]
    assert ymd[0][0] == 2022
    assert ymd[0][-1] == 2024
    assert np.array_equal(ncf.read_time(TEST_FILE), nct.ncreadtime(TEST_FILE))


def test_read_within() -> None:
    sample, dims = ncf.read_within(TEST_FILE, "lon", ((120, 150),))
    assert sample.shape == (21,)
    assert len(dims) == 1
    assert dims[0].shape == (21,)

    sample, __ = ncf.read_within(TEST_FILE, "time", ((None, None),))
    assert sample.shape == (36,)
    __, dims = ncf.read_within(
        TEST_FILE, "time", ((dn.from_ymd(2024, 1, 15), dn.from_ymd(2024, 12, 16)),)
    )
    assert dims[0][0] == dn.from_ymd(2024, 1, 15)
    assert dims[0][-1] == dn.from_ymd(2024, 12, 15)


# ---- write tests ----
def test_write_attr(tmp_nc: Path) -> None:
    core.write_attr(tmp_nc, "data", "long_name", "test data")
    assert ncf.attr_val(tmp_nc, "data", "long_name") == "test data"

    with nc.Dataset(tmp_nc, "a") as h:
        h.setncattr("author", "tester")
    assert ncf.attr_val(tmp_nc, "/", "author") == "tester"

    with pytest.raises((KeyError, FileNotFoundError)):
        core.write_attr(tmp_nc, "nonexistent", "attr", "val")


def test_write(tmp_nc: Path) -> None:
    original = ncf.read(tmp_nc, "data")
    core.write(tmp_nc, "data", original)
    assert np.array_equal(ncf.read(tmp_nc, "data"), original)

    core.write(tmp_nc, "data", original.tolist())  # array-like input
    assert np.array_equal(ncf.read(tmp_nc, "data"), original)

    subset_data = original[:2, :3, :5]
    core.write(tmp_nc, "data", subset_data, (slice(0, 2), slice(0, 3), slice(0, 5)))
    assert np.array_equal(
        ncf.read(tmp_nc, "data", (slice(0, 2), slice(0, 3), slice(0, 5))), subset_data
    )

    with pytest.raises(ValueError):
        core.write(tmp_nc, "data", np.array([]))

    with pytest.raises(ValueError):
        core.write(tmp_nc, "data", np.zeros((2, 2)))

    with pytest.raises(ValueError):
        core.write(tmp_nc, "data", np.zeros((2, 3, 5)), (slice(0),))


def test_create(tmp_nc: Path) -> None:
    core.create(tmp_nc, "data", {"t": 4, "y": 3, "x": 5})
    assert "data" in ncf.var_names(tmp_nc)

    new_path = tmp_nc.parent / "new.nc"
    with nc.Dataset(new_path, "w") as h:
        h.createDimension("time", 5)
        h.createVariable("time", np.float32, {"time": 5})
    core.create(new_path, "time", {"time": 5})
    core.create(new_path, "var", {"time": 5, "y": 2, "x": 3})
    assert "time" in ncf.var_names(new_path)
    assert ncf.shape(new_path, "var") == (5, 2, 3)
    assert ncf.dim_names(new_path, "var") == ("time", "y", "x")

    with pytest.raises(ValueError):
        core.create(tmp_nc, "data", {"t": 99, "y": 99, "x": 99})


def test_create_chunksizes(tmp_path: Path) -> None:
    path = tmp_path / "chunked.nc"

    core.create(path, "var", {"y": 6, "x": 4}, chunksizes=(2, 2))
    with nc.Dataset(path) as h:
        assert h["var"].chunking() == [2, 2]

    path2 = tmp_path / "default.nc"
    core.create(path2, "var", {"y": 6, "x": 4})
    with nc.Dataset(path2) as h:
        assert h["var"].chunking() == [6, 4]  # contiguous by default


def test_set_my_attrs(tmp_nc: Path) -> None:
    with nc.Dataset(tmp_nc, "a") as h:
        core._set_my_attrs(h, "lon")
        assert h["lon"].axis == "X"
        assert h["lon"].units == "degrees_east"

        core._set_my_attrs(h, "lat")
        assert h["lat"].axis == "Y"
        assert h["lat"].units == "degrees_north"

        core._set_my_attrs(h, "time")
        assert h["time"].axis == "T"
        assert h["time"].units == "days since 2000-01-01 00:00:00"


def test_read_nonexistent_var(tmp_nc: Path) -> None:
    with pytest.raises(KeyError):
        ncf.read(tmp_nc, "nonexistent")


def test_read_within_mismatched_ndim(tmp_nc: Path) -> None:
    with pytest.raises(ValueError):
        ncf.read_within(tmp_nc, "data", ((1, 2),))


def test_read_within_idim_time_with_decode_false(tmp_nc: Path) -> None:
    with pytest.raises(ValueError):
        ncf.read_within(
            tmp_nc, "data", ((None, None),) * 3, idim_time=0, decode_time=False
        )


def test_read_within_idim_time_out_of_range(tmp_nc: Path) -> None:
    with pytest.raises(IndexError):
        ncf.read_within(tmp_nc, "data", ((None, None),) * 3, idim_time=99)


def test_attr_val_nonexistent(tmp_nc: Path) -> None:
    with pytest.raises(KeyError):
        ncf.attr_val(tmp_nc, "data", "nonexistent_attr")


# ---- save tests ----
def _fresh_file(path: Path) -> Path:
    """create an empty netcdf file"""
    with nc.Dataset(path, "w"):
        pass
    return path


def test_save_new_variable(tmp_path: Path) -> None:
    path = _fresh_file(tmp_path / "new.nc")

    data = {
        "sst": np.random.rand(4, 3).astype(np.float32),
        "time": np.arange(4),
        "lat": np.arange(3),
    }
    core.save(path, data)

    # note: create() makes dimension variables first and the variable last
    assert ncf.var_names(path) == ("time", "lat", "sst")
    assert ncf.dim_names(path, "sst") == ("time", "lat")
    assert ncf.shape(path, "sst") == (4, 3)

    for name, value in data.items():
        assert np.array_equal(ncf.read(path, name), value)

    assert ncf.attr_val(path, "time", "axis") == "T"
    assert ncf.attr_val(path, "time", "units") == "days since 2000-01-01 00:00:00"
    assert ncf.attr_val(path, "lat", "axis") == "Y"
    assert ncf.attr_val(path, "lat", "units") == "degrees_north"


def test_save_single_entry_var_is_dim(tmp_path: Path) -> None:
    path = _fresh_file(tmp_path / "new.nc")

    lon = [1.0, 2.0, 3.0]
    data = {"lon": lon}
    core.save(path, data)

    assert ncf.var_names(path) == ("lon",)
    assert ncf.shape(path, "lon") == (3,)
    assert np.array_equal(ncf.read(path, "lon"), [1.0, 2.0, 3.0])

    assert data["lon"] is lon  # the input dict must not be mutated


def test_save_dtype_option(tmp_path: Path) -> None:
    path = _fresh_file(tmp_path / "new.nc")

    core.save(path, {"v": np.arange(5)}, dtype=np.float64)

    assert ncf.read(path, "v").dtype == np.float64


def test_save_creates_missing_file(tmp_path: Path) -> None:
    path = tmp_path / "absent.nc"
    data = {"v": [1.0, 2.0]}

    core.save(path, data)

    assert path.exists()
    assert ncf.var_names(path) == ("v",)
    assert np.array_equal(ncf.read(path, "v"), [1.0, 2.0])


def test_save_overwrite_confirmed(
    tmp_nc: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    core.save(tmp_nc, {"vec": [1.0, 2.0, 3.0]})

    monkeypatch.setattr("builtins.input", lambda *_: "y")
    core.save(tmp_nc, {"vec": [9.0, 8.0, 7.0]})

    assert np.array_equal(ncf.read(tmp_nc, "vec"), [9.0, 8.0, 7.0])


def test_save_overwrite_declined(tmp_nc: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    core.save(tmp_nc, {"vec": [1.0, 2.0, 3.0]})

    answers = iter(["maybe", "", "n"])  # garbage first to exercise the retry loop
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    with pytest.raises(FileExistsError):
        core.save(tmp_nc, {"vec": [9.0, 8.0, 7.0]})

    assert np.array_equal(ncf.read(tmp_nc, "vec"), [1.0, 2.0, 3.0])


def test_save_overwrite_forced(tmp_nc: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    core.save(tmp_nc, {"vec": [1.0, 2.0, 3.0]})

    def no_input(*_: Any) -> str:
        raise AssertionError("input() must not be called")

    monkeypatch.setattr("builtins.input", no_input)
    core.save(tmp_nc, {"vec": [9.0, 8.0, 7.0]}, overwrite_var=True, overwrite_dim=True)

    assert np.array_equal(ncf.read(tmp_nc, "vec"), [9.0, 8.0, 7.0])


def test_save_rejects_uncastable_data(tmp_path: Path) -> None:
    path = _fresh_file(tmp_path / "new.nc")

    with pytest.raises(ValueError, match="cannot cast"):
        core.save(path, {"v": [[1, 2], [3]]})


def test_save_rejects_non_1d_dimension(tmp_path: Path) -> None:
    path = _fresh_file(tmp_path / "new.nc")

    data = {"v": np.zeros(2), "bad": np.zeros((2, 2))}
    with pytest.raises(ValueError, match="ndim==1"):
        core.save(path, data)


def test_save_rejects_empty_data(tmp_path: Path) -> None:
    path = _fresh_file(tmp_path / "new.nc")

    with pytest.raises(ValueError, match="empty"):
        core.save(path, {"v": np.zeros(0)})

    with pytest.raises(ValueError, match="empty"):
        core.save(path, {"v": np.zeros((2, 2)), "d": np.array([])})


def test_try_from_cache_refreshes_on_mtime_change(tmp_nc: Path) -> None:
    ds1 = core.DatasetMeta.try_from_cache(tmp_nc)
    mtime = tmp_nc.stat().st_mtime

    with nc.Dataset(tmp_nc, "a") as h:
        h["lon"].long_name = "changed longitude"

    os.utime(tmp_nc, (mtime + 10, mtime + 10))  # defeat timestamp granularity

    ds2 = core.DatasetMeta.try_from_cache(tmp_nc)
    assert ds2 is ds1  # same cached instance, refreshed in place
    assert ds2.variables["lon"].attributes["long_name"] == "changed longitude"


# ---- error quality ----
def test_errors_are_chained_and_contextual(tmp_nc: Path) -> None:
    with pytest.raises(ValueError) as info:
        ncf.read(tmp_nc, "data", (slice(None),) * 99)
    assert info.value.__cause__ is not None
    assert "path=" in str(info.value)

    with pytest.raises(KeyError, match="'data'") as info:  # available vars listed
        ncf.read(tmp_nc, "nonexistent")
    assert str(tmp_nc) in str(info.value)

    with pytest.raises(KeyError, match="cache"):
        core.DatasetMeta.try_from_cache(tmp_nc.parent / "nope.nc", cache_only=True)


def test_write_subset_failure_is_chained(tmp_nc: Path) -> None:
    too_big = np.zeros((2, 2, 2))
    with pytest.raises(ValueError) as info:
        core.write(tmp_nc, "data", too_big, (slice(1), slice(1), slice(1)))
    assert info.value.__cause__ is not None


def test_read_within_non_monotonic_dim(tmp_path: Path) -> None:
    path = _fresh_file(tmp_path / "new.nc")
    core.save(path, {"d": [0.0, 1.0, 3.0, 2.0]})  # scrambled ordering

    with pytest.raises(ValueError):
        ncf.read_within(path, "d", ((None, None),), decode_time=False)


def test_read_within_descending_dim(tmp_path: Path) -> None:
    path = _fresh_file(tmp_path / "new.nc")
    core.save(path, {"lon": [3.0, 2.0, 1.0, 0.0, -1.0]})

    sample, dims = ncf.read_within(path, "lon", ((0.5, 2.5),), decode_time=False)
    assert np.array_equal(sample, [1.0, 2.0])  # ascending despite desc storage
    assert np.array_equal(dims[0], sample)


def test_read_time_fixed_calendar_leap(tmp_path: Path) -> None:
    path = tmp_path / "fixed.nc"
    with nc.Dataset(path, "w") as h:
        h.createDimension("t", 366)
        v = h.createVariable("time", np.float64, ("t",))
        v[:] = np.arange(366)
        v.units = "days since 2024-01-01"
        v.calendar = "365_day"

    time = ncf.read_time(path)

    # a noleap year has no Feb29: day 59 must land on Mar 1 (not Mar 2)
    assert time[58] == dn.from_ymd(2024, 2, 28)
    assert time[59] == dn.from_ymd(2024, 3, 1)
    assert time[364] == dn.from_ymd(2024, 12, 31)
    assert time[365] == dn.from_ymd(2025, 1, 1)
