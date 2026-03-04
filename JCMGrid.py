import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, atan2, asin, stack
from dataclasses import dataclass
from typing import List
from global_land_mask import globe
from pathlib import Path
from jem.tool_scripts.generate_jcm_forcing_and_topography_files import generate_jcm_forcing_and_topography_files
import xarray as xr

@dataclass
class JCMGrid:
    r_spherical: np.ndarray
    r_corners_spherical: np.ndarray
    binary_mask: np.ndarray
    grid_solid_angles: np.ndarray

def normalize(a):
    normalized = a / norm(a, ord=2, axis=0, keepdims=True)
    return normalized

def cartesian_to_spherical(r_cartesian):
    """
    Convert from Cartesian coordinate to spherical coordinate
    The 0th axis is (x, y, z), the 1st axis is the points
    """
    d = norm(r_cartesian, ord=2, axis=0)
    lat = asin(r_cartesian[2] / d)
    lon = atan2(r_cartesian[1], r_cartesian[0])
    return stack((d, lon, lat), axis=0)

def spherical_to_cartesian(r_sphere):
    """
    Convert from spherical coordinate to lon lat
    The 0th axis is (r, lon, lat), the 1st axis is the points
    """
    x = r_sphere[0] * cos(r_sphere[2]) * cos(r_sphere[1])
    y = r_sphere[0] * cos(r_sphere[2]) * sin(r_sphere[1])
    z = r_sphere[0] * sin(r_sphere[2])
    return stack((x, y, z), axis=0)

def compute_solid_angle(r_corners_spherical):

    r_corners_cartesian = spherical_to_cartesian(r_corners_spherical)
    vec1 = (  ( r_corners_cartesian[:, 1, :, :] - r_corners_cartesian[:, 0, :, :] ) 
            + ( r_corners_cartesian[:, 2, :, :] - r_corners_cartesian[:, 3, :, :] ) ) / 2 
    vec2 = (  ( r_corners_cartesian[:, 2, :, :] - r_corners_cartesian[:, 1, :, :] ) 
            + ( r_corners_cartesian[:, 3, :, :] - r_corners_cartesian[:, 0, :, :] ) ) / 2 

    # outer product in-place
    d0 =   vec1[1] * vec2[2] - vec1[2] * vec2[1]
    d1 = - vec1[0] * vec2[2] + vec1[2] * vec2[0]
    d2 =   vec1[0] * vec2[1] - vec1[1] * vec2[0]
    areas = (d0**2 + d1**2 + d2**2)**0.5
    solid_angles = areas / np.sum(areas) * np.pi * 4

    return solid_angles

def generate_JCMGrid(
    resolution: int,
):
    files = generate_jcm_forcing_and_topography_files(resolution) 
    ds_terrain = xr.open_dataset(files["terrain"]) 
   
    lat_centers = ds_terrain.coords["lat"].to_numpy() * np.pi/180 
    lon_centers = ds_terrain.coords["lon"].to_numpy() * np.pi/180
    dlat = lat_centers[1] - lat_centers[0]    
    dlon = lon_centers[1] - lon_centers[0]
   
    if dlat <= 0:
        raise ValueError("dlat is negative")
    if dlon <= 0:
        raise ValueError("dlon is negative")
    if np.any( np.abs( (lat_centers[1:] - lat_centers[:-1]) - dlat ) > 1e-3 ):
        raise Exception("Error: latitudes spacings are not equal")
    if np.any( np.abs( (lon_centers[1:] - lon_centers[:-1]) - dlon ) > 1e-3 ):
        raise Exception("Error: longtitudes spacings are not equal")
    
    nlat = len(lat_centers) 
    nlon = len(lon_centers)

    #lat_bounds = np.linspace(lat_centers[0] - dlat/2, lat_centers[-1] + dlat/2, nlat+1) 
    #lon_bounds = np.linspace(lon_centers[0] - dlon/2, lon_centers[-1] + dlon/2, nlon+1)
 
    lat_bounds = np.linspace(-90, 90, nlat+1) * np.pi/180.0 
    lon_bounds = np.linspace(0, 360, nlon+1) * np.pi/180.0
 
    lat_centers = (lat_bounds[1:] + lat_bounds[:-1])/2
    lon_centers = (lon_bounds[1:] + lon_bounds[:-1])/2
 
    # JCM is lon-lat
    r_spherical = np.zeros((3, nlon, nlat))
    r_corners_spherical = np.zeros((3, 4, nlon, nlat))
    for i in range(nlon):
        for j in range(nlat):
            r_spherical[:, i, j] = [1.0, lon_centers[i], lat_centers[j]]
            r_corners_spherical[:, 0, i, j] = [1.0, lon_bounds[i], lat_bounds[j]]
            r_corners_spherical[:, 1, i, j] = [1.0, lon_bounds[i+1], lat_bounds[j]]
            r_corners_spherical[:, 2, i, j] = [1.0, lon_bounds[i+1], lat_bounds[j+1]]
            r_corners_spherical[:, 3, i, j] = [1.0, lon_bounds[i], lat_bounds[j+1]]

    # Construct land-sea mask
    lon_deg = r_spherical[1, :] * 180/np.pi
    lat_deg = r_spherical[2, :] * 180/np.pi
    lon_deg[lon_deg > 180] -= 360.0

    binary_mask = np.ones_like(lon_deg)#globe.is_land( lat_deg, lon_deg )
        
    # Construct solid angles
    grid_solid_angles = compute_solid_angle(r_corners_spherical)

    return JCMGrid(
        r_spherical = r_spherical,
        r_corners_spherical = r_corners_spherical,
        binary_mask = binary_mask,
        grid_solid_angles = grid_solid_angles,
    )

    
def write_to_SCRIP_grid_file(grid: JCMGrid, output_file: str | Path, flatten:bool = True):
 
    grid_size = grid.binary_mask.size
    grid_corners = 4
    if flatten:
        grid_dims = [ grid.binary_mask.size ]
    else:
        grid_dims = list(grid.binary_mask.shape)
    grid_shape = grid.binary_mask.shape 
    # After testing, I found the order assumed in ESGM_RegridWeightGen is reversed.
    # This is undocumented in their user manual
    grid_dims = grid_dims[::-1] 
    
    grid_dim_names = ["lon", "lat"]
    grid_center_lon = grid.r_spherical[1]
    grid_center_lat = grid.r_spherical[2]
    grid_imask = grid.binary_mask
    grid_area = grid.grid_solid_angles
  
    # dim => (corners(4), lon, lat) 
    grid_corner_lon = np.permute_dims( grid.r_corners_spherical[1], axes=(1, 2, 0))
    grid_corner_lat = np.permute_dims( grid.r_corners_spherical[2], axes=(1, 2, 0))
    
    rad2deg = 180 / np.pi
    # Need copy. I am not using "*=" because it is in-place
    grid_corner_lon = grid_corner_lon * rad2deg   
    grid_corner_lat = grid_corner_lat * rad2deg   
    grid_center_lat = grid_center_lat * rad2deg
    grid_center_lon = grid_center_lon * rad2deg


    if flatten:
        ds = xr.Dataset(
            data_vars = dict(
                grid_dims = ( ["grid_rank", ], grid_dims),
                grid_imask = ( ["grid_size", ], grid_imask.flatten()),
                grid_center_lat = ( ["grid_size", ], grid_center_lat.flatten() , {"units" : "degrees"} ),
                grid_center_lon = ( ["grid_size", ], grid_center_lon.flatten() , {"units" : "degrees"} ),
                grid_corner_lat = ( ["grid_size", "grid_corners"], grid_corner_lat.reshape((-1, grid_corners)), {"units" : "degrees"} ),
                grid_corner_lon = ( ["grid_size", "grid_corners"], grid_corner_lon.reshape((-1, grid_corners)), {"units" : "degrees"} ),
                grid_area = ( ["grid_size",], grid_area.flatten(), {"units" : "radians^2"} ),
            ),
        )

    else:
        ds = xr.Dataset(
            data_vars = dict(
                grid_dims = ( ["grid_rank", ], grid_dims),
                grid_imask = ( [*grid_dim_names], grid_imask),
                grid_center_lat = ( [*grid_dim_names], grid_center_lat, {"units" : "degrees"} ),
                grid_center_lon = ( [*grid_dim_names], grid_center_lon, {"units" : "degrees"} ),
                grid_corner_lat = ( [*grid_dim_names, "grid_corners"], grid_corner_lat, {"units" : "degrees"} ),
                grid_corner_lon = ( [*grid_dim_names, "grid_corners"], grid_corner_lon, {"units" : "degrees"} ),
                grid_area = ( [*grid_dim_names], grid_area, {"units" : "radians^2"} ),
            ),
        )

    print(f"grid_shape = {grid_shape}")
    da_shape = xr.DataArray(
        name ="grid_shape",
        dims = ["shape_dimension",],
        data = list(grid_shape),
    )

    ds = xr.merge([ds, da_shape])

    ds.to_netcdf(output_file)


def test_output_SCRIP_file():
    
    resolutions = [ 31 ]

    for resolution in resolutions:
        output_file = f"grid_JCM_T{resolution:d}.nc"
        output_file_SCRIP = f"grid_JCM_T{resolution:d}.SCRIP.nc"
        print("Generating grid...") 
        grid = generate_JCMGrid(resolution)
        print("Writing to file: ", output_file)
        write_to_SCRIP_grid_file(grid, output_file, flatten=False)
        print("Writing to file: ", output_file_SCRIP)
        write_to_SCRIP_grid_file(grid, output_file_SCRIP, flatten=True)

if __name__ == "__main__":
    test_output_SCRIP_file()
     
