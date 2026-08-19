from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, ClassVar, Literal

import netCDF4 as nc
import numpy as np

PathLike = str | Path


# -------------metadata assets-----------------------
@dataclass
class VariableMeta:
    shape: tuple[int, ...]
    dimensions: tuple[str, ...]
    attributes: dict[str, Any] = field(default_factory=dict)
    ndim: int = field(init=False)

    def __post_init__(self) -> None:
        self.ndim = len(self.shape)


class DatasetMeta:
    variables: dict[str, VariableMeta]
    attributes: dict[str, Any]
    cached_registry: ClassVar[dict[str, DatasetMeta]] = {}

    def __init__(self, path: PathLike) -> None:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(str(path))

        self.mtime = path.stat().st_mtime
        self._read(path)
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

        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                f"{e!r}\nnetcdf4 library is unable to read the file: {path!s}\n(see message above)"
            )

    @staticmethod
    def _to_cache_key(path: PathLike) -> str:
        return str(Path(path).resolve())

    @staticmethod
    def try_from_cache(path: PathLike, cache_only: bool = False) -> DatasetMeta:
        key = DatasetMeta._to_cache_key(path)
        ds = DatasetMeta.cached_registry.get(key)

        if cache_only and ds is None:
            raise KeyError("path is not registered")

        if ds is None:
            return DatasetMeta(path)

        if ds.mtime == Path(path).stat().st_mtime:
            return ds

        else:
            return DatasetMeta(path)


# ------------- helpers ---------------
def _get_variable_meta(path: PathLike, var_name: str) -> VariableMeta:
    ds = DatasetMeta.try_from_cache(path)

    try:
        return ds.variables[var_name]

    except KeyError as e:
        raise KeyError(
            f"'{e!r}\n{var_name}' is not a variable name in {path!s} {tuple(ds.variables.keys())!s}"
        )


def _get_attribute_metas(path: PathLike, var_name: str) -> dict[str, Any]:
    if var_name == "/":  # root attributes
        ds = DatasetMeta.try_from_cache(path)
        return ds.attributes

    else:
        return _get_variable_meta(path, var_name).attributes


# ---------- metat data queries ------
def var_names(path: PathLike) -> tuple[str, ...]:
    """get the variable names in the file"""
    ds = DatasetMeta.try_from_cache(path)
    return tuple(ds.variables.keys())


def dim_names(path: PathLike, var_name: str) -> tuple[str, ...]:
    """get the dimension names of a variable in the file"""
    return _get_variable_meta(path, var_name).dimensions


def shape(path: PathLike, var_name: str) -> tuple[int, ...]:
    """get the shape of a variable in the file"""
    return _get_variable_meta(path, var_name).shape


def ndim(path: PathLike, var_name: str) -> int:
    """get the shape of a variable in the file"""
    return len(_get_variable_meta(path, var_name).shape)


def attr_names(path: PathLike, var_name: str) -> tuple[str, ...]:
    """
    get the attribute names of
        the file       if var_name == '/'
        the variable   else
    """
    attributes = _get_attribute_metas(path, var_name)
    return tuple(attributes.keys())


def attr_val(path: PathLike, var_name: str, attr_name: str) -> Any:
    """
    get the attribute values of
        the file       if var_name == '/'
        the variable   else
    """
    attributes = _get_attribute_metas(path, var_name)
    return attributes[attr_name]


def var_names_include(
    path: PathLike,
    name_includes: str | tuple[str, ...],
    accept_ndims: int | tuple[int, ...],
    accept_counts: int | tuple[int, ...] = 1,
) -> tuple[str, ...]:
    if isinstance(name_includes, str):
        name_includes = (name_includes,)

    if isinstance(accept_ndims, int):
        accept_ndims = (accept_ndims,)

    if isinstance(accept_counts, int):
        accept_counts = (accept_counts,)

    ds = DatasetMeta.try_from_cache(path)
    candidates = tuple(
        var_name
        for var_name, var_meta in ds.variables.items()
        if any(ni in var_name for ni in name_includes) and var_meta.ndim in accept_ndims
    )

    if len(candidates) in accept_counts:
        return candidates

    else:
        raise ValueError(
            f"unable to locate var names include {name_includes} with {accept_ndims=}, {accept_counts=} in path {path!s} -> found {candidates=}"
        )


def attr_names_include(
    path: PathLike,
    var_name: str,
    name_includes: str | tuple[str, ...],
    accept_counts: int | tuple[int, ...] = 1,
) -> tuple[str, ...]:
    if isinstance(name_includes, str):
        name_includes = (name_includes,)

    if isinstance(accept_counts, int):
        accept_counts = (accept_counts,)

    candidates = tuple(
        attr_name
        for attr_name in attr_names(path, var_name)
        if any(ni in attr_name for ni in name_includes)
    )

    if len(candidates) in accept_counts:
        return candidates

    else:
        raise ValueError(
            f"unable to locate attr names of {var_name} include {name_includes} with {accept_counts=} in path {path!s} -> found {candidates=}"
        )


def _guess_time_name(path: str | PathLike) -> str:
    time_name_guesses = ("time", "date", "TIME", "DATE", "Time", "Date")
    try:
        return var_names_include(
            path, time_name_guesses, accept_ndims=(1,), accept_counts=(1,)
        )[0]

    except ValueError as e:
        raise ValueError(
            f"{e!r}\nunable to guess the variable name of time dimension (from {var_names(path)} in {path!s})"
        )


# ---- readers ----
def read(
    path: PathLike, var_name: str, subsets: tuple[slice, ...] | None = None
) -> np.ndarray:
    # check var_name here and hide the KeyError inside
    var_meta = _get_variable_meta(path, var_name)

    with nc.Dataset(path, "r") as h:
        if subsets is None:
            data = h[var_name][:]
        else:
            try:
                data = h[var_name][subsets]
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"{e!r}\nunable to read {var_name=}({var_meta.shape}) with {subsets=} from {path!s}"
                )

    if any(s == 0 for s in data.shape):
        raise ValueError(
            f"an empty array of {data.shape} is read with the given {subsets=} for {var_name=}({var_meta.shape}) in {path!r}"
        )

    return np.array(data)


def read_time(
    path: PathLike,
    time_name: str | None = None,
    unit_name: str | None = None,
    calendar_name: str | None = None,
    subset: slice | None = None,
) -> np.ndarray:
    import datenum as dn

    # ---------------------------------
    # --- auto configure the parameters

    time_name = time_name or _guess_time_name(path)

    if unit_name is None:
        unit_name_guesses = ("unit", "Unit", "UNIT")
        try:
            unit_name = attr_names_include(
                path,
                time_name,
                unit_name_guesses,
                accept_counts=(1,),
            )[0]

        except ValueError as e:
            raise ValueError(f"{e!r}\nunable to guess the attribute name of time units")

    subset = subset or slice(None)

    # ---------------------------------
    # --- parse the time units to delta and epoch
    units = attr_val(path, time_name, unit_name)
    if not isinstance(units, str):
        raise TypeError(
            f"{units=} is not of string type, found {type(units)}: ({time_name=}, {unit_name=}, {path!s})"
        )
    try:
        str_delta, str_epoch = str(units).split("since")
    except ValueError as e:
        raise ValueError(
            f"{e!r}\ncannot parse {units=} to (delta and epoch) by 'since': ({time_name=}, {unit_name=}, {path!s})"
        )

    str_epoch = str_epoch.strip()

    # ---------------------------------
    # --- delta string -> value
    str_delta = str_delta.strip()

    dict_delta = {
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

    try:
        delta_method, delta = dict_delta[str_delta]

    except KeyError as e:
        raise ValueError(
            f"{e!r}\nunrecognized {str_delta=} ({time_name=}, {unit_name=}, {path!s})"
        )

    # ---------------------------------
    # --- epoch string -> value
    if str_epoch == "1-1-1 00:00:00":  # weird origin time (CMIP6 models)
        epoch = dn.from_ymd(1, 1, 1)
    else:
        epoch = dn.from_string(str_epoch)

    # ---------------------------------
    # ---- lazy guess the calendar name here
    if delta_method == "day" and calendar_name is None:
        calander_name_guesses = ("calendar", "Calendar", "CALENDAR")
        try:
            calendar_names = attr_names_include(
                path,
                time_name,
                calander_name_guesses,
                accept_counts=(0, 1),
            )

            if calendar_names:
                calendar_name = calendar_names[0]

        except ValueError as e:
            raise ValueError(
                f"{e!r}\nunable to guess the attribute name of time calendar"
            )

    if calendar_name is not None:
        calendar = attr_val(path, time_name, calendar_name)
    else:
        calendar = None

    time_value = read(path, time_name, (subset,))
    if delta_method == "month":
        return dn.add_month(epoch, time_value)

    elif delta_method == "year":
        return dn.add_month(epoch, time_value * 12)

    elif calendar not in ("365_day", "365_days", "360_day", "360_days"):
        return epoch + time_value * delta
    else:
        days_per_year = int(calendar[:3])

        year_epoch = dn.year(epoch)
        remainder_epoch = epoch - dn.from_ymd(year_epoch, 1, 1)
        ret = []

        for value in time_value:
            year, remainder = int(value // days_per_year), value % days_per_year

            year += year_epoch
            remainder += remainder_epoch

            if remainder > days_per_year:
                year += 1
                remainder -= days_per_year

            t = dn.from_ymd(year, 1, 1) + remainder
            # get rid of the extra day from Feb29
            if remainder >= 59 and dn.is_leap_year(t):
                t += 1

            ret.append(t)

        return np.array(ret)


def read_within(
    path: str | PathLike,
    var_name: str,
    withins: tuple[tuple[float | None, float | None], ...],
    idim_time: int | None = None,
    decode_time=True,
) -> tuple[np.ndarray, tuple[np.ndarray]]:
    """withins specifies the values of dimensions for subset reading"""
    ### figure out idim_time if not given
    var_meta = _get_variable_meta(path, var_name)

    if idim_time is not None and not decode_time:
        raise ValueError(f"{idim_time=} is specified by {decode_time=}")

    elif idim_time is not None and decode_time:
        try:
            time_name = var_meta.dimensions[idim_time]

        except IndexError as e:
            raise IndexError(
                f"{e!r}\n{var_name=} only has ndim={var_meta.ndim} but {idim_time=} is requested: {path!s}"
            )
    elif idim_time is None and decode_time:
        time_name = _guess_time_name(path)

    elif idim_time is None and not decode_time:
        time_name = None

    ### validate the withins
    if len(withins) != var_meta.ndim:
        raise ValueError(
            f"ndim={var_meta.ndim} but only {len(withins)} elements are given in {withins}. {var_name=}, {path!s}"
        )

    def dim_info(
        dim_name: str, within: tuple[float | None, float | None]
    ) -> tuple[np.ndarray, slice, bool]:
        """
        return 1) the dimension 2) slice for the within in the file 3) needs to be flipped
        """

        bot, top = within
        bot = bot or -np.inf
        top = top or np.inf

        if bot > top:
            raise ValueError(f"value1 > value2 in {within} does not make sense")

        if dim_name != time_name:
            values = read(path, dim_name)
        else:
            values = read_time(path, dim_name)

        value_max = np.max(values)
        if bot > value_max:
            if dim_name == time_name:
                from datenum import to_string

                bot, value_max = to_string(bot), to_string(value_max)
            raise ValueError(
                f"requested lower boundary={bot} is larger than the maximum value of the array ({value_max}). {dim_name=}, {var_name=} {path!s}"
            )

        value_min = np.min(values)
        if top < value_min:
            if dim_name == time_name:
                from datenum import to_string

                top, value_min = to_string(top), to_string(value_min)
            raise ValueError(
                f"requested upper boundary={top} is smaller than the minimum value of the array ({value_min}). {dim_name=} {time_name=} {var_name=} {path!s}"
            )

        diff = values[1:] - values[:-1]
        if all(diff >= 0):
            is_reversed = False

        elif all(diff <= 0):
            is_reversed = True

        else:
            raise RuntimeError(
                f"dimension value sometimes increases and sometimes decreases. {dim_name=} {path!s}"
            )

        if any(diff == 0):
            print(
                f"warning: some dimension values does not increase or decrease in {dim_name=} {path!s}"
            )

        if is_reversed:
            bot, top = top, bot

        (is_within,) = np.where(np.logical_and(bot <= values, values <= top))
        slice_within = slice(
            is_within[0], is_within[-1] + 1
        )  # need one more to include the last boundary

        if is_reversed:
            values = np.flip(values[is_within])

        return values[is_within], slice_within, is_reversed

    dims, slices, are_reversed = list(
        zip(
            *[
                dim_info(dim_name, within)
                for dim_name, within in zip(var_meta.dimensions, withins)
            ]
        )
    )

    value = np.flip(
        read(path, var_name, slices),
        axis=tuple(i for i, is_reversed in enumerate(are_reversed) if is_reversed),
    )

    return value, dims


# ---- writers ----
def _set_my_attrs(h_file: nc.Dataset, name: str):
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


def write_attr(path: PathLike, var_name: str, attr_name: str, attr_val: str) -> None:
    # hide the KeyError and FileNotFoundError here
    _get_variable_meta(path, var_name)
    with nc.Dataset(path, "a") as h_file:
        if attr_name == "/":
            h_file.setncattr(attr_name, attr_val)
        else:
            setattr(h_file[var_name], attr_name, attr_val)


def write(
    path: PathLike,
    var_name: str,
    data: np.ndarray,
    subsets: tuple[slice, ...] | None = None,
) -> None:
    if any(s == 0 for s in data.shape):
        raise ValueError(
            f"I'm not going to help you write an empty data of shape={data.shape} to {var_name=} in {path!s}"
        )

    # check var_name here and hide the and FileNotFoundError KeyError inside
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

    with nc.Dataset(path, "a") as h:
        if subsets is None:
            h[var_name][:] = data
        else:
            try:
                h[var_name][subsets] = data
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(
                    f"{e!r}\nunable to write {var_name}({data.shape}) with {subsets=} to {var_names}({var_meta.shape}) in  {path!s}"
                )


def create(
    path: PathLike,
    var_name: str,
    dim_specs: dict[str, int],
    use_my_attrs: bool = True,
    significant_digits: None | int = None,
    dtype: nc.DatatypeType = np.float32,
    complevel: Literal[0, 1, 2, 3, 4, 5, 6, 7, 8, 9] | None = 4,
    shuffle: bool = True,
):
    """create a varaible and its dimensions in a file if any of them do not exist yet"""
    shape = tuple(dim_specs.values())
    dim_names = tuple(dim_specs.keys())

    # ---- check the consistency of shape and dimnames if variable already exists
    try:
        var_meta = _get_variable_meta(path, var_name)
    except KeyError, FileNotFoundError:
        var_meta = None

    if var_meta is not None:  # if the var_name exists in the file
        if shape != var_meta.shape:
            raise ValueError(
                f"The shape of the {var_name=} is different in the input ({shape}) and the existing file ({var_meta.shape}) in {path!s}"
            )
        elif dim_names != var_meta.dimensions:
            raise ValueError(
                f"The dimension names of the {var_name=} is different in the input ({dim_names}) and the existing file ({var_meta.dimensions}) in {path!s}"
            )
        else:  # everythin checks and nothing to do here
            return

    # ---- create the dimensions
    try:
        existing_names = var_names(path)
    except FileNotFoundError:
        existing_names = ()

    # ---- prepare specs that need to be created
    missing_spec = {
        name: (shape if name == var_name else (dim_specs[name],))
        for name in dict.fromkeys((*dim_names, var_name))
        if name not in existing_names
    }

    with nc.Dataset(path, "a") as h_file:
        for name, shape in missing_spec.items():
            # create dimension and write attributes first
            if name in dim_specs:
                h_file.createDimension(name, shape[0])
                if use_my_attrs:
                    _set_my_attrs(h_file, name)
                create_var = h_file.createVariable

            else:
                create_var = partial(
                    h_file.createVariable,
                    significant_digits=significant_digits,
                    compression="zlib",
                    complevel=complevel,
                    shuffle=shuffle,
                )

            create_var(var_name, dtype, dim_names)
