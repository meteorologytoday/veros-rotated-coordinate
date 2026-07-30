"""
Sanity-check plot for the coarsened land-sea masks produced by
convert_era5_landsea_mask.py, rendered using each grid's actual cell
bounds (not just cell-center scatter/imshow), so cell shape and coverage
can be checked directly:

  - ERA5 native lsm (imshow)
  - JCM land_fraction, rendered with its lat_bnds/lon_bnds edges
  - Rotated grid land_fraction, rendered with its pre-rotation lat_bnds/lon_bnds edges
  - Rotated grid land_fraction, rendered as true quadrilaterals (true_lat_bnds/true_lon_bnds)
    at true geographic lon/lat -- this is the one that should show the
    rotated-pole swirl pattern near Greenland/Antarctica.
"""
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from pathlib import Path

from convert_era5_landsea_mask import ERA5_FILE, OUTPUT_DIR, JCM_RESOLUTION, RG_RESOLUTIONS_DEG

FIGURE_FILE = Path("figure") / "landsea_mask_sanity_check.png"


def edges_from_bnds_1d(bnds):
    """bnds: (n, 2) -> (n+1,) shared edge array, assuming a contiguous rectilinear axis."""
    return np.concatenate([bnds[:, 0], bnds[-1:, 1]])


def pcolormesh_true(ax, ds, **kwargs):
    lon_edges = edges_from_bnds_1d(ds["lon_bnds"].values)
    lat_edges = edges_from_bnds_1d(ds["lat_bnds"].values)
    return ax.pcolormesh(lon_edges, lat_edges, ds["land_fraction"].values, **kwargs)


def polycollection_true(ax, lon_corners, lat_corners, data, **kwargs):
    """
    lon_corners, lat_corners: (n0, n1, 4) quadrilateral corners in degrees.
    data: (n0, n1).
    Longitude is unwrapped per-cell relative to its own first corner so
    cells that straddle the 0/360 seam don't get drawn as a spurious
    full-width span.
    """
    n0, n1, ncorner = lon_corners.shape
    lon = lon_corners.reshape(n0 * n1, ncorner).copy()
    lat = lat_corners.reshape(n0 * n1, ncorner)

    # Unwrap each cell's corners relative to its own first corner. Using a
    # single reference point (rather than e.g. the median) is important: a
    # circular median computed on already-wrapped angles can itself land
    # near the wrap boundary (e.g. corners [359.9, 359.9, 0.1, 0.1] have
    # median ~180), which silently breaks the correction for half the cells.
    lon_ref = lon[:, :1]
    lon = lon_ref + (lon - lon_ref + 180.0) % 360.0 - 180.0

    verts = np.stack([lon, lat], axis=-1)
    pc = PolyCollection(verts, array=data.reshape(-1), **kwargs)
    ax.add_collection(pc)
    ax.set_xlim(0, 360)
    ax.set_ylim(-90, 90)
    return pc


def main():
    cmap = "GnBu"
    n_rows = 1 + len(RG_RESOLUTIONS_DEG)
    fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))

    ds_era5 = xr.open_dataset(ERA5_FILE)["lsm"].isel(valid_time=0)
    axes[0, 0].imshow(ds_era5.values, origin="upper", cmap=cmap, vmin=0, vmax=1, extent=[0, 360, -90, 90])
    axes[0, 0].set_title("ERA5 native (0.25deg)")

    ds_jcm = xr.open_dataset(OUTPUT_DIR / f"landsea_mask_JCM_T{JCM_RESOLUTION}.nc")
    im = pcolormesh_true(axes[0, 1], ds_jcm, cmap=cmap, vmin=0, vmax=1)
    axes[0, 1].set_title(f"JCM T{JCM_RESOLUTION} land_fraction\n(rendered from lat_bnds/lon_bnds)")
    fig.colorbar(im, ax=axes[0, 1])

    for row, resolution_deg in enumerate(RG_RESOLUTIONS_DEG, start=1):
        print(f"Plotting RG {resolution_deg}deg ...")
        ds_rg = xr.open_dataset(OUTPUT_DIR / f"landsea_mask_RG_{resolution_deg:.2f}deg.nc")

        im = pcolormesh_true(axes[row, 0], ds_rg, cmap=cmap, vmin=0, vmax=1)
        axes[row, 0].set_title(f"RG {resolution_deg}deg land_fraction\n(pre-rotation frame, from un-rotated lat_bnds/lon_bnds)")
        fig.colorbar(im, ax=axes[row, 0])

        pc = polycollection_true(
            axes[row, 1],
            ds_rg["true_lon_bnds"].values,
            ds_rg["true_lat_bnds"].values,
            ds_rg["land_fraction"].values,
            cmap=cmap,
        )
        pc.set_clim(0, 1)
        axes[row, 1].set_title(f"RG {resolution_deg}deg land_fraction\n(true quadrilaterals, from true_lat_bnds/true_lon_bnds)")
        fig.colorbar(pc, ax=axes[row, 1])

    for ax in axes.flatten():
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)

    fig.tight_layout()
    FIGURE_FILE.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(FIGURE_FILE, dpi=150)
    print(f"Wrote {FIGURE_FILE}")


if __name__ == "__main__":
    main()
