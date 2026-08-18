from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import netCDF4 as nc

PathLike = str | Path


def _abort_missing_path(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(str(path))


# -------------metadata assets-----------------------
@dataclass
class VariableMeta:
    shape: tuple[int, ...]
    dimensions: tuple[str, ...]
    attributes: dict[str, Any] = field(default_factory=dict)


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


def _get_variable_metas(path: PathLike, var_name: str) -> VariableMeta:
    ds = DatasetMeta.try_from_cache(path)

    try:
        return ds.variables[var_name]

    except KeyError:
        raise KeyError(
            f"'{var_name}' is not a variable name in {path!s} {tuple(ds.variables.keys())!s}"
        )


def _get_attribute_metas(path: PathLike, var_name: str) -> dict[str, Any]:
    if var_name == "/":  # root attributes
        ds = DatasetMeta.try_from_cache(path)
        return ds.attributes

    else:
        return _get_variable_metas(path, var_name).attributes


def var_names(path: PathLike) -> tuple[str, ...]:
    """get the variable names in the file"""
    ds = DatasetMeta.try_from_cache(path)
    return tuple(ds.variables.keys())


def dim_names(path: PathLike, var_name: str) -> tuple[str, ...]:
    """get the dimension names of a variable in the file"""
    return _get_variable_metas(path, var_name).dimensions


def attr_names(path: PathLike, var_name: str) -> tuple[str, ...]:
    """
    get the attribute names of
        the file       if var_name == '/'
        the variable   else
    """
    attributes = _get_attribute_metas(path, var_name)
    return tuple(attributes.keys())


def attr_val(path: PathLike, var_name: str, attr_name: str):
    """
    get the attribute values of
        the file       if var_name == '/'
        the variable   else
    """
    attributes = _get_attribute_metas(path, var_name)
    return attributes[attr_name]
