# ncfunc

Functional-style reading and writing of NetCDF data files.

`ncfunc` wraps [netCDF4](https://github.com/Unidata/netcdf4-python) and trades
its open-handle, object-oriented style for small stateless functions: every
call opens the file, does one thing, closes it, and leaves nothing behind.
Along the way it decodes CF time coordinates, subsets by coordinate bounds,
and turns opaque netCDF4 errors into messages that say what failed, where,
and why - keeping the original exception chained underneath.

## Features

- **Stateless functions** - no `Dataset` handles to open, close or leak
- **Readable errors** - Exceptions are raised with details such as file path,
variable name, dimension name, etc.
- **Time decoding** - `read_time` decodes CF `'delta since epoch'` units to
[datenum](https://pypi.org/project/datenum/) serials, including fixed-length
calendars (`360_day`, `365_day`)
- **Bounds-based subsetting** - `read_within` slices variables by coordinate
ranges instead of index arithmetic
- **Cached metadata** - structure queries reuse a path-keyed cache that
auto-refreshes when a file's mtime changes (`DatasetMeta`)
- **One-shot writer** - `save()` creates dimensions, coordinates and the
variable in a single call, with overwrite protection

## Installation

Requires Python >= 3.10, with dependencies numpy, netCDF4 and datenum installed
automatically:

```sh
pip install ncfunc
# or
uv add ncfunc
```

## Quickstart

```python
import ncfunc as ncf

file = "tests/ersst_2022-2024.nc"
```

## API overview

| function | purpose |
| --- | --- |
| `var_names(path)` | variable names |
| `dim_names(path, var)` | dimension names of a variable |
| `shape(path, var)` / `ndim(path, var)` | shape / rank of a variable |
| `attr_names(path, var)` / `attr_val(path, var, attr)` | attributes of a variable, or of the file with `var='/'` |
| `var_names_include(...)` | find variables by substring and rank |
| `attr_names_include(...)` | find attributes by substring |
| `read(path, var, subsets?)` | read a variable into `np.ndarray` |
| `read_time(path, ...)` | read and decode a time coordinate to datenum values |
| `read_within(path, var, withins, ...)` | bounds-based subset read with coordinates |
| `write(path, var, data, subsets?)` | write into an existing variable |
| `create(path, var, dim_specs, ...)` | create a variable (+ dimensions), idempotently |
| `save(path, data, ...)` | create + write a variable and its coordinates in one call |
| `write_attr(path, var, attr, value)` | set a variable or root (`'/'`) attribute |
| `DatasetMeta(path)` | static structural snapshot, cached per resolved path |


### Inspect metadata

```python
>>> ncf.var_names(file)
('time', 'lon', 'lat', 'sst', 'ssta')

>>> ncf.shape(file, 'sst')
(36, 121, 240)

>>> ncf.dim_names(file, 'sst')
('time', 'lat', 'lon')

>>> ncf.attr_val(file, '/', 'title')      # '/' selects the file's root attributes
'NOAA monthly ERSSTv6 (in situ only)'
```

Search helpers locate variables and attributes by substring, and insist on an
unambiguous match (by rank or count) before returning:

```python
>>> ncf.var_names_include(file, ('sst',), accept_ndims=3)
('sst', 'ssta')

>>> ncf.var_names_include(file, name_includes=('sst',), accept_ndims=3, accept_counts=(2,))
('sst', 'ssta')
```

### Read data

```python
sst = ncf.read(file, 'sst')                          # whole variable
top = ncf.read(file, 'sst', ((slice(-4, None),) * 3))  # last 4 steps of every dim
```

### Decode time

```python
>>> import datenum as dn
>>> t = ncf.read_time(file)
>>> dn.to_string(t[0]), dn.to_string(t[-1])
('2022-01-15 00:00:00', '2024-12-15 00:00:00')
```

The time variable, its `units` attribute and its `calendar` attribute are all
guessed; pass `time_name`, `unit_name`, `calendar_name` explicitly to override.
Month/year-based units decode via month arithmetic, day-based ones against the
declared calendar - including `360_day` and `365_day` fixed calendars.

### Subset by bounds, not indices

`read_within` takes one `(lower, upper)` pair per dimension of the variable
(`None` = unbounded), reads only what intersects, and returns both the data
and the bounded coordinates. Coordinates come back ascending even when stored
descending; the data is flipped to stay aligned:

```python
>>> sst, (time, lat, lon) = ncf.read_within(
...     file,
...     'sst',
...     withins=((None, None), (-30.0, 30.0), (150.0, 210.0)),
... )
>>> sst.shape, lat[0], lat[-1]
((36, 41, 41), -30.0, 30.0)
```

The time dimension is found by guessing a time-named coordinate; point at it
explicitly with `idim_time=<index>` if the guess would be wrong, or disable
time handling with `decode_time=False`.

### Write data

`save` writes a variable plus its dimensions in one shot. The first entry is
the variable, the rest are its 1-D dimensions:

```python
import numpy as np

ncf.save(
    "out.nc",
    {
        "tas": np.arange(12, dtype="f4").reshape(3, 4),
        "time": np.array([0, 31, 59]),
        "lon": np.linspace(0.5, 3.5, 4),
    },
)
```

This creates `out.nc` with dimensions `time` and `lon`, coordinate variables
stamped with CF-ish attributes (`axis`, `units`, `standard_name`), and the
compressed `tas` variable. If `tas` already exists in the file, `save` asks
for confirmation on the terminal; `overwrite_var=True` and `overwrite_dim=True`
skips the question.

For finer control, use the pieces directly:

```python
ncf.create(path, "tas", {"time": 3, "lon": 4})   # idempotent; missing dims created
ncf.write(path, "tas", data)                     # full write, shape must match
ncf.write(path, "tas", data, ((slice(0, 1), slice(None)),))   # or by slices
ncf.write_attr(path, "/", "history", "created today")         # '/' = root attribute
```

## Development

```sh
uv sync            # install dependencies
uv run pytest      # run the test suite
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check src/ tests/
```

## License

MIT - see [LICENSE](LICENSE).
