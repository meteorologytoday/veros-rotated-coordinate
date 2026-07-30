import numpy as np
import xarray as xr
from pathlib import Path


def load_era5_landsea_mask(era5_file: str | Path):
    """
    Load the ERA5 land-sea mask netCDF file and return grid centers
    (degrees, ascending order) together with the land fraction data laid
    out as (nlon, nlat).

    ERA5's native grid has latitude descending (90 -> -90, poles included
    as grid points) and longitude ascending (0 -> 360, exclusive). Both
    are reordered/transposed here to (lat ascending, array shape (nlon, nlat)).
    """
    ds = xr.open_dataset(era5_file)
    lsm = ds["lsm"]
    for dim in ("valid_time", "time", "number"):
        if dim in lsm.dims:
            lsm = lsm.isel({dim: 0})

    lat_centers = ds["latitude"].to_numpy()
    lon_centers = ds["longitude"].to_numpy()
    lsm_data = lsm.to_numpy()  # shape (nlat, nlon), lat descending

    if lat_centers[0] > lat_centers[-1]:
        lat_centers = lat_centers[::-1]
        lsm_data = lsm_data[::-1, :]

    # (nlat, nlon) -> (nlon, nlat)
    lsm_data = np.ascontiguousarray(lsm_data.T)

    ds.close()

    return lon_centers, lat_centers, lsm_data
