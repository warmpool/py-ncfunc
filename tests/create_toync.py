import ncfunc as ncf


def main() -> None:
    dir = "./tests/data"
    # ----
    x0, y0, nx, ny = 0, 0, 2, 3

    path = f"{dir}a1a.nc"
    ncf.save(
        path,
        {
            "var": [[0] * nx] * ny,
            "y": list(range(y0, y0 + ny)),
            "x": list(range(x0, x0 + nx)),
        },
    )

    path = f"{dir}a1b.nc"
    x0 += nx
    ncf.save(
        path,
        {
            "var": [[0] * nx] * ny,
            "y": list(range(y0, y0 + ny)),
            "x": list(range(x0, x0 + nx)),
        },
    )

    path = f"{dir}a1c.nc"
    x0 += nx
    ncf.save(
        path,
        {
            "var": [[0] * nx] * ny,
            "y": list(range(y0, y0 + ny)),
            "x": list(range(x0, x0 + nx)),
        },
    )

    # ----
    x0, y0, nx, ny = 0, 3, 2, 3

    path = f"{dir}a2a.nc"
    ncf.save(
        path,
        {
            "var": [[0] * nx] * ny,
            "y": list(range(y0, y0 + ny)),
            "x": list(range(x0, x0 + nx)),
        },
    )

    y0 += ny
    path = f"{dir}a3a.nc"
    ncf.save(
        path,
        {
            "var": [[0] * nx] * ny,
            "y": list(range(y0, y0 + ny)),
            "x": list(range(x0, x0 + nx)),
        },
    )


if __name__ == "__main__":
    main()
