from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import datenum as dn
import netCDF4 as nc
import numpy as np
from numpy.typing import ArrayLike

PathLike = str | Path
StrOrStrs = str | tuple[str, ...]
IntOrInts = int | tuple[int, ...]
DTypeSpec = (
    np.dtype[Any]
    | str
    | type[int | float | complex | bool | str | np.bool_ | np.number | np.str_]
)
MAX_DATASET_CACHE: int = 128


# --------------metadata assets-----------------------
@dataclass
class VariableMeta:
    """static description of a single variable inside a NetCDF file"""

    shape: tuple[int, ...]
    dimensions: tuple[str, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    ndim: int = field(init=False)

    def __post_init__(self) -> None:
        self.ndim = len(self.shape)


class DatasetMeta:
    """
    static description of a NetCDF file: its variables and root attributes.

    instances are cheap snapshots built without reading any data values.
    a module-level registry keeps the most recently used entries (up to
    MAX_CACHE) keyed by resolved path; use try_from_cache() to reuse or
    refresh them.
    """

    variables: dict[str, VariableMeta]
    attributes: dict[str, Any]
    cached_registry: ClassVar[dict[str, DatasetMeta]] = {}

    def __init__(self, path: PathLike) -> None:
        """
        read the file's structure into this instance and register it in the cache

        raises FileNotFoundError if the path does not exist,
        RuntimeError if the netcdf4 library cannot open it
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(str(path))

        self.mtime = path.stat().st_mtime
        self._read(path)

        # -- uncaching the old entries
        while len(self.cached_registry) >= MAX_DATASET_CACHE:
            self.cached_registry.pop(next(iter(self.cached_registry)))

        # -- caching
        self.cached_registry[DatasetMeta._to_cache_key(path)] = self

    def _read(self, path: Path) -> None:
        try:
            with nc.Dataset(path, "r") as h:
                self.variables = {
                    name: VariableMeta(
                        variable.shape,
                        variable.dimensions,
                        {
                            aname: variable.getncattr(aname)
                            for aname in variable.ncattrs()
                        },
                    )
                    for name, variable in h.variables.items()
                }
                self.attributes = {name: h.getncattr(name) for name in h.ncattrs()}

        except Exception as e:
            msg = f"the netcdf4 library failed to read {path}"
            raise RuntimeError(msg) from e

    @staticmethod
    def _to_cache_key(path: PathLike) -> str:
        return str(Path(path).resolve())

    @staticmethod
    def try_from_cache(path: PathLike, cache_only: bool = False) -> DatasetMeta:
        """
        return the metadata of the file, reusing the cache when possible

        a fresh DatasetMeta is built when nothing is cached yet; a cached
        entry is re-read when the file's mtime changed since caching.

        raises KeyError if cache_only=True and the path is not cached,
        FileNotFoundError if the path does not exist
        """
        path = Path(path)
        key = DatasetMeta._to_cache_key(path)
        ds = DatasetMeta.cached_registry.get(key)

        if cache_only and ds is None:
            raise KeyError(f"path is not registered in the metadata cache: {key}")

        if ds is None:
            return DatasetMeta(path)

        if ds.mtime == path.stat().st_mtime:
            return ds

        else:
            ds._read(path)
            return ds


# -------------- helpers ---------------
def _err_ctx(path: PathLike, var_name: str | None = None) -> str:
    """standardized context suffix appended to error messages"""
    if var_name is None:
        return f" (path={path})"

    return f" (path={path}, var='{var_name}')"


def _as_str_tuple(value: StrOrStrs) -> tuple[str, ...]:
    return value if isinstance(value, tuple) else (value,)


def _as_int_tuple(value: IntOrInts) -> tuple[int, ...]:
    return value if isinstance(value, tuple) else (value,)


def _has_variable(path: PathLike, var_name: str) -> bool:
    try:
        _get_variable_meta(path, var_name)
    except (FileNotFoundError, KeyError):
        return False

    return True


def _get_variable_meta(path: PathLike, var_name: str) -> VariableMeta:
    ds = DatasetMeta.try_from_cache(path)

    try:
        return ds.variables[var_name]

    except KeyError as e:
        available = tuple(ds.variables.keys())
        msg = (
            f"variable '{var_name}' does not exist in {path}, "
            f"available variables: {available}"
        )
        raise KeyError(msg) from e


def _get_attribute_metas(path: PathLike, var_name: str) -> dict[str, Any]:
    if var_name == "/":  # root attributes
        ds = DatasetMeta.try_from_cache(path)
        return ds.attributes

    else:
        return _get_variable_meta(path, var_name).attributes


# ----------- meta data queries ------
def var_names(path: PathLike) -> tuple[str, ...]:
    """
    get the variable names in the file, in file order

    raises FileNotFoundError if the path does not exist
    """
    ds = DatasetMeta.try_from_cache(path)
    return tuple(ds.variables.keys())


def dim_names(path: PathLike, var_name: str) -> tuple[str, ...]:
    """
    get the dimension names of a variable in the file, in dimension order

    raises KeyError if the variable does not exist, FileNotFoundError if
    the path does not exist
    """
    return _get_variable_meta(path, var_name).dimensions


def shape(path: PathLike, var_name: str) -> tuple[int, ...]:
    """
    get the shape of a variable in the file

    raises KeyError if the variable does not exist, FileNotFoundError if
    the path does not exist
    """
    return _get_variable_meta(path, var_name).shape


def ndim(path: PathLike, var_name: str) -> int:
    """
    get the ndim (rank) of a variable in the file

    raises KeyError if the variable does not exist, FileNotFoundError if
    the path does not exist
    """
    return len(_get_variable_meta(path, var_name).shape)


def attr_names(path: PathLike, var_name: str) -> tuple[str, ...]:
    """
    get the attribute names of
        the file       if var_name == '/'
        the variable   else

    raises KeyError if the variable does not exist, FileNotFoundError if
    the path does not exist
    """
    attributes = _get_attribute_metas(path, var_name)
    return tuple(attributes.keys())


def attr_val(path: PathLike, var_name: str, attr_name: str) -> Any:
    """
    get an attribute value of
        the file       if var_name == '/'
        the variable   else

    raises KeyError if the variable or the attribute does not exist,
    FileNotFoundError if the path does not exist
    """
    attributes = _get_attribute_metas(path, var_name)
    return attributes[attr_name]


def var_names_include(
    path: PathLike,
    name_includes: StrOrStrs,
    accept_ndims: IntOrInts,
    accept_counts: IntOrInts = 1,
) -> tuple[str, ...]:
    """
    find variable names containing any of the given substrings

    name_includes: substring(s) to search for; a plain str is one pattern
    accept_ndims: accepted variable rank(s); a plain int is one value
    accept_counts: acceptable number of matches for an unambiguous result

    returns all matching variable names in file order

    raises ValueError unless the number of matches is in accept_counts,
    FileNotFoundError if the path does not exist
    """
    name_includes = _as_str_tuple(name_includes)
    accept_ndims = _as_int_tuple(accept_ndims)
    accept_counts = _as_int_tuple(accept_counts)

    ds = DatasetMeta.try_from_cache(path)
    candidates = tuple(
        var_name
        for var_name, var_meta in ds.variables.items()
        if any(ni in var_name for ni in name_includes) and var_meta.ndim in accept_ndims
    )

    if len(candidates) in accept_counts:
        return candidates

    msg = (
        f"unable to locate variable names including {name_includes} "
        f"with ndim in {accept_ndims} and count in {accept_counts}, "
        f"found {candidates}{_err_ctx(path)}"
    )
    raise ValueError(msg)


def attr_names_include(
    path: PathLike,
    var_name: str,
    name_includes: StrOrStrs,
    accept_counts: IntOrInts = 1,
) -> tuple[str, ...]:
    """
    find attribute names containing any of the given substrings

    name_includes: substring(s) to search for; a plain str is one pattern
    accept_counts: acceptable number of matches for an unambiguous result;
        pass (0, 1) to tolerate absence
    var_name selects whose attributes to search: '/' for the file's root
    attributes, a variable name otherwise

    returns all matching attribute names in file order

    raises ValueError unless the number of matches is in accept_counts,
    KeyError if the variable does not exist, FileNotFoundError if the path
    does not exist
    """
    name_includes = _as_str_tuple(name_includes)
    accept_counts = _as_int_tuple(accept_counts)

    candidates = tuple(
        attr_name
        for attr_name in attr_names(path, var_name)
        if any(ni in attr_name for ni in name_includes)
    )

    if len(candidates) in accept_counts:
        return candidates

    msg = (
        f"unable to locate attribute names of '{var_name}' including {name_includes} "
        f"with count in {accept_counts}, found {candidates}{_err_ctx(path, var_name)}"
    )
    raise ValueError(msg)


def _guess_time_name(path: PathLike) -> str:
    guesses = ("time", "date", "TIME", "DATE", "Time", "Date")
    try:
        return var_names_include(path, guesses, accept_ndims=(1,), accept_counts=(1,))[
            0
        ]

    except ValueError as e:
        available = var_names(path)
        msg = (
            f"unable to guess the time variable among {guesses}, "
            f"available variables: {available}{_err_ctx(path)}"
        )
        raise ValueError(msg) from e


# ----- readers ----
def read(
    path: PathLike, var_name: str, subsets: tuple[slice, ...] | None = None
) -> np.ndarray:
    """
    read a variable into a numpy array

    subsets: one slice per variable dimension, applied as
        h[var_name][subsets]; None reads the whole variable

    raises KeyError if the variable does not exist, FileNotFoundError if
    the path does not exist, ValueError if the slices fail or would select
    an empty array
    """
    # check var_name here and hide the KeyError inside
    var_meta = _get_variable_meta(path, var_name)

    with nc.Dataset(path, "r") as h:
        if subsets is None:
            data = h[var_name][:]
        else:
            try:
                data = h[var_name][subsets]
            except Exception as e:
                msg = (
                    f"unable to read '{var_name}'({var_meta.shape}) "
                    f"with {subsets=}{_err_ctx(path, var_name)}"
                )
                raise ValueError(msg) from e

    if any(s == 0 for s in data.shape):
        raise ValueError(
            f"an empty array of {data.shape} is read with {subsets=} "
            f"for '{var_name}'({var_meta.shape}) in {path}"
        )

    return np.array(data)


_DELTA_TABLE: dict[str, tuple[str, float]] = {
    "second": ("day", 1 / 86400),
    "seconds": ("day", 1 / 86400),
    "minute": ("day", 1 / 1440),
    "minutes": ("day", 1 / 1440),
    "hour": ("day", 1 / 24),
    "hours": ("day", 1 / 24),
    "day": ("day", 1),
    "days": ("day", 1),
    "month": ("month", 1),
    "months": ("month", 1),
    "year": ("month", 12),
    "years": ("month", 12),
}

_FIXED_CALENDARS = ("365_day", "365_days", "360_day", "360_days")


def _parse_units(units: str, ctx: str) -> tuple[str, float, str]:
    """parse CF-style units '<delta> since <epoch>' to (method, factor, epoch-string)"""
    str_delta, sep, str_epoch = units.partition("since")

    if not sep:
        msg = f"cannot parse {units=} by the 'since' keyword{ctx}"
        raise ValueError(msg)

    try:
        delta_method, delta = _DELTA_TABLE[str_delta.strip()]

    except KeyError as e:
        msg = f"unrecognized time unit delta={str_delta!r} in {units=}{ctx}"
        raise ValueError(msg) from e

    return delta_method, delta, str_epoch.strip()


def _epoch_from_string(str_epoch: str) -> float:
    if str_epoch == "1-1-1 00:00:00":  # weird origin time (CMIP6 models)
        return dn.from_ymd(1, 1, 1)

    return dn.from_string(str_epoch)


def _decode_fixed_calendar(
    epoch: float, time_value: np.ndarray, calendar: str
) -> np.ndarray:
    """decode time values of a fixed-length calendar (360_day or noleap/365_day)"""
    days_per_year = int(calendar[:3])

    year_epoch = dn.year(epoch)
    remainder_epoch = epoch - dn.from_ymd(year_epoch, 1, 1)

    years = year_epoch + time_value // days_per_year
    remainders = remainder_epoch + time_value % days_per_year

    cross_year = remainders >= days_per_year
    years[cross_year] += 1
    remainders[cross_year] -= days_per_year

    datenums = dn.from_ymd(years, 1, 1) + remainders

    # get rid of the extra day from Feb29
    datenums[(remainders >= 59) & dn.is_leap_year(datenums)] += 1
    return datenums


def read_time(
    path: PathLike,
    time_name: str | None = None,
    unit_name: str | None = None,
    calendar_name: str | None = None,
    subset: slice | None = None,
) -> np.ndarray:
    """
    read and decode a time coordinate into datenum serial values

    names of the time variable, of its units attribute and of its calendar
    attribute are guessed when not given. units follow the CF convention
    '<delta> since <epoch>' where delta is seconds/minutes/hours/days/
    months/years. month-based units decode via datenum.add_month; day-based
    ones decode against the calendar attribute:

        standard/gregorian/... : epoch + value * delta
        360_day or 365_day     : fixed-length-calendar decoding

    subset limits which time values are read.

    raises ValueError when names cannot be guessed or units cannot be
    parsed, TypeError when the units value is not a string,
    KeyError/FileNotFoundError when variables or attributes do not exist
    """
    time_name = time_name or _guess_time_name(path)

    if unit_name is None:
        try:
            unit_name = attr_names_include(
                path, time_name, ("unit", "Unit", "UNIT"), accept_counts=(1,)
            )[0]

        except ValueError as e:
            msg = f"unable to guess the attribute name of time units{_err_ctx(path, time_name)}"
            raise ValueError(msg) from e

    subset = subset or slice(None)

    # ----------------------------------
    # ---- parse the time units to delta and epoch
    units = attr_val(path, time_name, unit_name)
    if not isinstance(units, str):
        msg = (
            f"time units must be a string, found {type(units).__name__}: {units!r} "
            f"(path={path}, var='{time_name}', unit='{unit_name}')"
        )
        raise TypeError(msg)

    ctx = _err_ctx(path, time_name)
    delta_method, delta, str_epoch = _parse_units(units, ctx)
    epoch = _epoch_from_string(str_epoch)

    # ----------------------------------
    # ----- lazy guess the calendar name here
    if delta_method == "day" and calendar_name is None:
        try:
            calendar_names = attr_names_include(
                path,
                time_name,
                ("calendar", "Calendar", "CALENDAR"),
                accept_counts=(0, 1),
            )
            calendar_name = calendar_names[0] if calendar_names else None

        except ValueError as e:
            msg = f"unable to guess the attribute name of time calendar{ctx}"
            raise ValueError(msg) from e

    calendar = (
        attr_val(path, time_name, calendar_name) if calendar_name is not None else None
    )

    time_value = read(path, time_name, (subset,))
    if delta_method == "month":
        return dn.add_month(epoch, time_value * delta)

    if isinstance(calendar, str) and calendar in _FIXED_CALENDARS:
        return _decode_fixed_calendar(epoch, time_value, calendar)

    return epoch + time_value * delta


def _dim_bounds(
    within: tuple[float | None, float | None],
    values: np.ndarray,
    *,
    dim_name: str,
    var_name: str,
    path: PathLike,
    is_time: bool,
) -> tuple[np.ndarray, slice, bool]:
    """
    locate the index range covering `within` in an ascending or descending array

    returns 1) the bounded dimension values 2) the matching slice 3) descending?
    """

    def fmt(value):  # raw datenum serials are unreadable in error messages
        return dn.to_string(value) if is_time else value

    bot, top = within
    bot = -np.inf if bot is None else bot
    top = np.inf if top is None else top

    if bot > top:
        msg = f"lower boundary {bot} exceeds upper boundary {top} in {within}{_err_ctx(path, dim_name)}"
        raise ValueError(msg)

    ctx = f", dim='{dim_name}' (path={path}, var='{var_name}')"
    value_max = np.max(values)
    if bot > value_max:
        raise ValueError(
            f"requested lower boundary={fmt(bot)} is above the maximum "
            f"dimension value ({fmt(value_max)}){ctx}"
        )

    value_min = np.min(values)
    if top < value_min:
        raise ValueError(
            f"requested upper boundary={fmt(top)} is below the minimum "
            f"dimension value ({fmt(value_min)}){ctx}"
        )

    diff = values[1:] - values[:-1]
    if np.all(diff >= 0):
        is_reversed = False

    elif np.all(diff <= 0):
        is_reversed = True

    else:
        msg = f"dimension values sometimes increase and sometimes decrease{ctx}"
        raise ValueError(msg)

    if np.any(diff == 0):
        warnings.warn(
            f"some dimension values neither increase nor decrease{ctx}",
            RuntimeWarning,
        )

    (is_within,) = np.where((bot <= values) & (values <= top))
    slice_within = slice(
        is_within[0], is_within[-1] + 1
    )  # need one more to include the last boundary

    bounded = values[is_within]
    return bounded[::-1] if is_reversed else bounded, slice_within, is_reversed


def read_within(
    path: PathLike,
    var_name: str,
    withins: tuple[tuple[float | None, float | None], ...],
    idim_time: int | None = None,
    decode_time: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, ...]]:
    """
    read a variable within bounds given per dimension, plus the bounded coordinates

    withins holds one (lower, upper) pair per dimension of the variable;
    None means unbounded on that side. each pair must satisfy
    lower <= upper and intersect the coordinate range.

    the time dimension is located by idim_time, an index into the
    variable's dimensions, or - by default - by guessing a time-named
    coordinate variable. decode_time=False disables time handling
    entirely; idim_time must then be None. bounds are matched against
    decoded datenum values when time decoding applies.

    coordinates are returned in ascending order even when stored
    descending; the data is flipped accordingly so it stays aligned.

    returns (values, coords) with coords a tuple of the bounded coordinate
    array of each dimension, in dimension order

    raises ValueError on conflicting arguments, a bound-count mismatch,
    invalid or non-intersecting bounds or non-monotonic dimensions,
    KeyError if the variable or one of its coordinates is missing,
    FileNotFoundError if the path does not exist
    """
    var_meta = _get_variable_meta(path, var_name)

    ### figure out the time dimension name if needed
    if idim_time is not None and not decode_time:
        raise ValueError(f"{idim_time=} conflicts with {decode_time=}")

    if idim_time is not None:
        try:
            time_name: str | None = var_meta.dimensions[idim_time]

        except IndexError as e:
            msg = (
                f"'{var_name}' only has ndim={var_meta.ndim} but dimension "
                f"{idim_time} is requested{_err_ctx(path, var_name)}"
            )
            raise IndexError(msg) from e

    elif decode_time:
        time_name = _guess_time_name(path)

    else:
        time_name = None

    ### validate the withins
    if len(withins) != var_meta.ndim:
        msg = (
            f"expecting {var_meta.ndim} bounds but received {len(withins)}: "
            f"{withins}, var='{var_name}' (path={path})"
        )
        raise ValueError(msg)

    ### locate the bounds of every dimension
    dims = []
    slices = []
    are_reversed = []
    for dim_name, within in zip(var_meta.dimensions, withins):
        is_time = dim_name == time_name
        values = read_time(path, dim_name) if is_time else read(path, dim_name)
        bounded, slice_within, is_reversed = _dim_bounds(
            within,
            values,
            dim_name=dim_name,
            var_name=var_name,
            path=path,
            is_time=is_time,
        )
        dims.append(bounded)
        slices.append(slice_within)
        are_reversed.append(is_reversed)

    ### read the variable and un-flip dimensions that were reversed
    value = np.flip(
        read(path, var_name, tuple(slices)),
        axis=[i for i, is_reversed in enumerate(are_reversed) if is_reversed],
    )

    return value, tuple(dims)


# ----- writers ----
def _set_my_attrs(h_file: nc.Dataset, name: str) -> None:
    h = h_file[name]
    if name.lower() in ("lon", "longitude", "longitudes"):
        h.axis = "X"
        h.units = "degrees_east"
        h.long_name = "longitude"
        h.standard_name = "longitude"

    if name.lower() in ("lat", "latitude", "latitudes"):
        h.axis = "Y"
        h.units = "degrees_north"
        h.long_name = "latitude"
        h.standard_name = "latitude"

    if name.lower() in (
        "lev",
        "level",
        "levs",
        "levels",
        "pressure",
        "plev",
        "dep",
        "depth",
        "depths",
    ):
        h.axis = "Z"

    if any(guess in name.lower() for guess in ("time", "date")):
        h.axis = "T"
        h.units = "days since 2000-01-01 00:00:00"
        h.long_name = "time"
        h.standard_name = "time"


def write_attr(path: PathLike, var_name: str, attr_name: str, value: Any) -> None:
    """
    set an attribute of
        the file       if var_name == '/'
        the variable   else

    raises KeyError if the variable does not exist, FileNotFoundError if
    the path does not exist
    """
    # check var_name here; KeyError and FileNotFoundError surface from here
    _get_variable_meta(path, var_name)
    with nc.Dataset(path, "a") as h_file:
        if var_name == "/":
            h_file.setncattr(attr_name, value)
        else:
            setattr(h_file[var_name], attr_name, value)


def write(
    path: PathLike,
    var_name: str,
    data: ArrayLike,
    subsets: tuple[slice, ...] | None = None,
) -> None:
    """
    write data into an existing variable, fully or by slices

    without subsets, data.shape must equal the variable's shape. with
    subsets (one slice per dimension), len(subsets) must equal the
    variable's ndim; netCDF4 then applies its own shape/broadcast rules.

    raises ValueError on empty data, mismatched shapes or failing slices,
    KeyError if the variable does not exist, FileNotFoundError if the path
    does not exist
    """
    data = np.asarray(data)
    if any(s == 0 for s in data.shape):
        raise ValueError(
            f"I'm not going to help you write an empty data of shape={data.shape} to {var_name=} in {path!s}"
        )

    # check var_name here; FileNotFoundError and KeyError surface from here
    var_meta = _get_variable_meta(path, var_name)

    # check the data shape
    if subsets is None and var_meta.shape != data.shape:
        raise ValueError(
            f"data shapes are different in the input and the file. input={data.shape}, file={var_meta.shape}. {var_name=} in {path!s}"
        )

    elif subsets is not None:
        # check the ndim
        received = len(subsets)
        expected = var_meta.ndim
        if received != expected:
            raise ValueError(
                f"length of input slices does not match ndim(variable), {expected=} {received=}"
            )

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        with nc.Dataset(path, "a") as h:
            if subsets is None:
                h[var_name][:] = data
            else:
                try:
                    h[var_name][subsets] = data
                except Exception as e:
                    msg = (
                        f"unable to write data of shape={data.shape} to "
                        f"'{var_name}'({var_meta.shape}) with {subsets=}"
                        f"{_err_ctx(path, var_name)}"
                    )
                    raise ValueError(msg) from e


def create(
    path: PathLike,
    var_name: str,
    dim_specs: dict[str, int],
    use_my_attrs: bool = True,
    significant_digits: None | int = None,
    dtype: DTypeSpec = np.float32,
    complevel: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | None = 4,
    shuffle: bool = True,
    chunksizes: Sequence[int] | None = None,
) -> None:
    """
    create a variable and its dimensions in a file, skipping existing parts

    dim_specs maps dimension names to sizes, in the variable's dimension
    order; the variable takes exactly this shape. missing dimensions are
    created first (each as a coordinate variable of the same name); a
    missing file is created on the fly.

    if the variable already exists, its shape and dimension names must
    match dim_specs exactly and nothing is written - calls are idempotent.
    when the variable doubles as its own first dimension (e.g. a
    coordinate), it is created as part of the dimension step above.

    use_my_attrs stamps CF-ish axis/units/long_name attributes onto
    created coordinate variables recognised as lon/lat/level/time-like.
    dtype accepts anything np.dtype understands ('f4', np.float32, ...);
    complevel > 0 enables zlib compression with the given level and
    shuffle filter. chunksizes sets the requested HDF5 chunk shape of the
    variable (per dimension, in order); None leaves chunking to netCDF4's
    default.

    raises ValueError when an existing variable disagrees with dim_specs
    """
    shape = tuple(dim_specs.values())
    dim_names = tuple(dim_specs.keys())

    # ----- check the consistency of shape and dimnames if variable already exists
    try:
        var_meta = _get_variable_meta(path, var_name)
    except (KeyError, FileNotFoundError):
        var_meta = None

    if var_meta is not None:  # if the var_name exists in the file
        if shape != var_meta.shape:
            raise ValueError(
                f"shape of '{var_name}' differs between the input ({shape}) "
                f"and the existing file ({var_meta.shape}) (path={path})"
            )

        if dim_names != var_meta.dimensions:
            raise ValueError(
                f"dimension names of '{var_name}' differ between the input ({dim_names}) "
                f"and the existing file ({var_meta.dimensions}) (path={path})"
            )

        return  # everything checks out and nothing to do here

    try:
        existing_names = var_names(path)
    except FileNotFoundError:
        existing_names = ()

    missing_dims = {
        name: size for name, size in dim_specs.items() if name not in existing_names
    }

    kwargs: dict[str, Any] = {
        "significant_digits": significant_digits,
        "compression": "zlib",
        "complevel": complevel,
        "shuffle": shuffle,
        "chunksizes": chunksizes,
    }

    with nc.Dataset(path, "a") as h_file:
        # create the dimension variables first
        # note: nc.DatatypeType only exists in netCDF4's type stubs, hence cast()
        for name, size in missing_dims.items():
            h_file.createDimension(name, size)
            h_file.createVariable(name, cast("nc.DatatypeType", dtype), (name,))
            if use_my_attrs:
                _set_my_attrs(h_file, name)

        # then the variable itself (unless it is one of the dimensions above,
        # i.e. the variable is its own dimension and already exists)
        if var_name not in missing_dims:
            h_file.createVariable(
                var_name, cast("nc.DatatypeType", dtype), dim_names, **kwargs
            )


def _confirm_overwrite(path: PathLike, var_name: str) -> None:
    """interactively confirm overwriting; declines after 10 invalid answers"""
    print(f"{var_name=} already exists in the file path {path!s}")

    valid = ("y", "n", "yes", "no")
    for _ in range(10):
        answer = input("Confirm to overwrite the values in the path? (y/n) ").lower()
        if answer in valid:
            break
    else:
        answer = "n"

    if answer not in ("y", "yes"):
        raise FileExistsError(f"{var_name=} already exists in {path}")


def save(
    path: PathLike,
    data: Mapping[str, ArrayLike],
    overwrite_var: bool = False,
    overwrite_dim: bool = False,
    use_my_attrs: bool = True,
    significant_digits: None | int = None,
    dtype: DTypeSpec = np.float32,
    complevel: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | None = 4,
    shuffle: bool = True,
) -> None:
    """
    save a variable and its dimension coordinates to a file in one call

    the first entry is the variable, the rest are its dimension
    coordinates (each 1-D), or - when the variable is its own dimension -
    it is the only entry:

        data = {
            var_name: var_data,
            dim1_name: dim1_data,
            ...
            dimN_name: dimN_data,
        }
        data = {var_name: var_data}   # variable == its own dimension

    a missing file is created; existing dimensions are reused.

    overwrite protection is split by role: overwrite_var guards the
    variable (the FIRST key) and overwrite_dim guards its dimension
    coordinates (the REMAINING keys). when a guarded name already exists
    in the file, _confirm_overwrite asks on the terminal; pass True for a
    role to skip the question. a single-entry data dict has no separate
    dimension coordinates, so overwrite_var covers it alone.

    remaining parameters pass through to create().

    raises ValueError when entries are empty, cannot be cast to arrays or
    a dimension is not 1-D, FileExistsError when overwrite is declined
    """
    names = tuple(data.keys())
    var_name = names[0]

    # ---- overwrite protection
    if not overwrite_var and _has_variable(path, var_name):
        _confirm_overwrite(path, var_name)

    if not overwrite_dim:
        for dim_name in names[1:]:
            if _has_variable(path, dim_name):
                _confirm_overwrite(path, dim_name)

    # ---- data check - can cast to array?
    arrays: dict[str, np.ndarray] = {}
    for name, value in data.items():
        try:
            arrays[name] = np.asarray(value)

        except Exception as e:
            msg = f"cannot cast data of '{name}' to a numpy array"
            raise ValueError(msg) from e

    # ---- data check - is ndim(dim) == 1?
    for i, (name, value) in enumerate(arrays.items()):
        if (
            (len(arrays) == 1 and i == 0)  # the only entry (variable == dimension)
            or (i > 0)  # the dimensions
        ) and value.ndim != 1:
            msg = (
                f"expecting dimension '{name}' with ndim==1, found shape={value.shape}"
            )
            raise ValueError(msg)

        if any(s == 0 for s in value.shape):
            raise ValueError(
                f"I'm not going to help you write an empty data of shape={value.shape} to {name=}"
            )

    # ---- assign parameters to create and write functions
    names = tuple(data.keys())

    if len(names) == 1:
        dim_names = (names[0],)
    else:
        dim_names = names[1:]

    dim_specs = {name: len(arrays[name]) for name in dim_names}

    create(
        path,
        names[0],
        dim_specs,
        use_my_attrs,
        significant_digits,
        dtype,
        complevel,
        shuffle,
    )

    for name, array in arrays.items():
        write(path, name, array)
