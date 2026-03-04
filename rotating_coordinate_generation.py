import numpy as np
from numpy.linalg import norm
from numpy import sin, cos, atan2, asin, stack
from dataclasses import dataclass
from typing import List, Dict
from global_land_mask import globe
from pathlib import Path

@dataclass
class RotatingGaussianGrid:
    r_spherical: np.ndarray
    r_corners_spherical: np.ndarray
#    r_spherical_true: np.ndarray
#    r_corners_spherical_true: np.ndarray
    binary_mask: np.ndarray
    solid_angles: np.ndarray

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

def rotate_along_z_axis(r_cartesian: np.ndarray, angle_rad: float):
    cos_angle = cos(angle_rad)
    sin_angle = sin(angle_rad)
    x = r_cartesian[0]
    y = r_cartesian[1]
    z = r_cartesian[2]
    new_x = cos_angle * x - sin_angle * y
    new_y = sin_angle * x + cos_angle * y
    return stack((new_x, new_y, z), axis=0)
 
def rotate_along_a_given_vector(
    r_cartesian: np.ndarray,
    rotate_vec: np.ndarray,
    angle_rad: float,
):
    cos_angle = cos(angle_rad)
    sin_angle = sin(angle_rad)
    
    unit_rotate_vec = normalize(rotate_vec)
    newaxis_expand = [None,] * len(r_cartesian.shape[1:])
    colon_expand = [slice(None),] * len(r_cartesian.shape[1:])
   
    z = dot(r_cartesian, unit_rotate_vec)
    x_vec = r_cartesian - z[None, *colon_expand] * unit_rotate_vec[:, *newaxis_expand]
    unit_x_vec = normalize(x_vec)
    x = dot(r_cartesian, unit_x_vec)
    
    # outer product in-place
    vec1 = unit_rotate_vec
    vec2 = unit_x_vec
    d0 =   vec1[1] * vec2[2] - vec1[2] * vec2[1]
    d1 = - vec1[0] * vec2[2] + vec1[2] * vec2[0]
    d2 =   vec1[0] * vec2[1] - vec1[1] * vec2[0]
    unit_y_vec = np.stack([d0, d1, d2], axis=0)
    
    new_x = cos_angle * x
    new_y = sin_angle * x
    
    return stack(
        (
            new_x * unit_x_vec[0] + new_y * unit_y_vec[0] + z * unit_rotate_vec[0],
            new_x * unit_x_vec[1] + new_y * unit_y_vec[1] + z * unit_rotate_vec[1],
            new_x * unit_x_vec[2] + new_y * unit_y_vec[2] + z * unit_rotate_vec[2],
        ),
        axis=0
    )

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]

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

def generate_rotating_gaussian_grid(
    lat_bounds: np.ndarray,
    lon_bounds: np.ndarray,
    unit: str = "degree",
    rotation_along_longitude_degree: float = 0.0,
    rotation_degree: float = 0.0,
):
    """
        lat_bounds: The founds of latitudes.
        lon_bounds: The founds of longitudes.
    """
    
    if unit == "degree":
        print("Input unit is 'degree', need conversion to radian")
        lat_bounds *= np.pi / 180.0
        lon_bounds *= np.pi / 180.0
    elif unit == "radian":
        print("Input unit is 'radian'.")
    else:
        raise Exception(f"Unknown unit: '{unit}'.")
  
    dlons = lon_bounds[1:] - lon_bounds[:-1]
    dlats = lat_bounds[1:] - lat_bounds[:-1]
    lon_centers = ( lon_bounds[:-1] + lon_bounds[1:] ) / 2.0
    lat_centers = ( lat_bounds[:-1] + lat_bounds[1:] ) / 2.0

    r_spherical = np.zeros((3, len(lat_centers), len(lon_centers)))
    r_corners_spherical = np.zeros((3, 4, len(lat_centers), len(lon_centers)))

    for i in range(len(lon_centers)):
        for j in range(len(lat_centers)):
            r_spherical[:, j, i] = [1.0, lon_centers[i], lat_centers[j]]
            r_corners_spherical[:, 0, j, i] = [1.0, lon_bounds[i], lat_bounds[j]]
            r_corners_spherical[:, 1, j, i] = [1.0, lon_bounds[i+1], lat_bounds[j]]
            r_corners_spherical[:, 2, j, i] = [1.0, lon_bounds[i+1], lat_bounds[j+1]]
            r_corners_spherical[:, 3, j, i] = [1.0, lon_bounds[i], lat_bounds[j+1]]

    def my_rotate(pts, longitude_degree, rotation_degree):
        longitude_radian = longitude_degree * np.pi / 180.0
        rotation_vec = np.array([np.cos(longitude_radian), np.sin(longitude_radian), 0.0]) 
        return cartesian_to_spherical(rotate_along_a_given_vector(
            spherical_to_cartesian(pts),
            rotation_vec,
            rotation_degree * np.pi / 180.0
        ))
    
    def confine_longitude(lon):
        lon = lon % (2*np.pi)
        lon[lon > np.pi] -= 2*np.pi
        return lon

    r_spherical = my_rotate(r_spherical, rotation_along_longitude_degree, rotation_degree)
    r_corners_spherical = my_rotate(r_corners_spherical, rotation_along_longitude_degree, rotation_degree)
    
    #r_spherical[1, :, :] = confine_longitude(r_spherical[1, :, :])
    #r_corners_spherical[1, :, :, :] = confine_longitude(r_corners_spherical[1, :, :, :])
    
    """
    # Construct the other three faces by rotation
    for i in range(3):
        _stack_r_spherical.append(rotate_90deg_along_z_axis(_stack_r_spherical[-1]))
        _stack_r_corners_spherical.append(rotate_90deg_along_z_axis(_stack_r_corners_spherical[-1]))
    """

    # Construct land-sea mask
    lon = r_spherical[1, :] * 180/np.pi
    lat = r_spherical[2, :] * 180/np.pi
    print(f"lat range:  {np.amin(lat)} to {np.amax(lat)}")
    print(f"lon range:  {np.amin(lon)} to {np.amax(lon)}")
    binary_mask = np.ones_like(globe.is_land(lat, confine_longitude(lon)))
    #binary_mask = globe.is_land(lat, lon)
        
    # Construct solid angles
    solid_angles = compute_solid_angle(r_corners_spherical)

    return RotatingGaussianGrid(
        r_spherical = r_spherical,
        r_corners_spherical = r_corners_spherical,
        binary_mask = binary_mask,
        solid_angles = solid_angles,
    )


def write_to_SCRIP_grid_file(
    rotating_gaussian_grid,
    output_file: str | Path,
    flatten: bool = True
):
    
    import xarray as xr
   
    grid_size = rotating_gaussian_grid.binary_mask.size
    grid_corners = 4
    if flatten:
        grid_dims = [ rotating_gaussian_grid.binary_mask.size ]
    else:
        grid_dims = list(rotating_gaussian_grid.binary_mask.shape)
    
    # grid_shape preserves the original shape, whereas grid_dims may be flattened
    grid_shape = list(rotating_gaussian_grid.binary_mask.shape)

    grid_center_lon = np.permute_dims(rotating_gaussian_grid.r_spherical[1], axes=(0, 1)).flatten() 
    grid_center_lat = np.permute_dims(rotating_gaussian_grid.r_spherical[2], axes=(0, 1)).flatten() 

    grid_imask = np.permute_dims(rotating_gaussian_grid.binary_mask, axes=(0, 1)).flatten()
    grid_area = np.permute_dims(rotating_gaussian_grid.solid_angles, axes=(0, 1)).flatten() 
    
    grid_corner_lon = np.permute_dims( rotating_gaussian_grid.r_corners_spherical[1], axes=(1, 2, 0)).reshape((-1, 4))
    grid_corner_lat = np.permute_dims( rotating_gaussian_grid.r_corners_spherical[2], axes=(1, 2, 0)).reshape((-1, 4))
        
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
                grid_imask = ( ["grid_size", ], grid_imask),
                grid_center_lat = ( ["grid_size", ], grid_center_lat, {"units" : "degrees"} ),
                grid_center_lon = ( ["grid_size", ], grid_center_lon, {"units" : "degrees"} ),
                grid_corner_lat = ( ["grid_size", "grid_corners"], grid_corner_lat, {"units" : "degrees"} ),
                grid_corner_lon = ( ["grid_size", "grid_corners"], grid_corner_lon, {"units" : "degrees"} ),
                grid_area = ( ["grid_size",], grid_area, {"units" : "radians^2"} ),
            ),
        )
    else:
        dim_names = ["j", "i"]
        print("dim_names : ", dim_names)
        print("grid_dims : ", grid_dims)
        ds = xr.Dataset(
            data_vars = dict(
                grid_dims = ( ["grid_rank", ], grid_dims),
                grid_imask = ( [*dim_names], grid_imask.reshape(grid_dims)),
                grid_center_lat = ( [*dim_names], grid_center_lat.reshape(grid_dims), {"units" : "degrees"} ),
                grid_center_lon = ( [*dim_names], grid_center_lon.reshape(grid_dims), {"units" : "degrees"} ),
                grid_corner_lat = ( [*dim_names, "grid_corners"], grid_corner_lat.reshape(grid_dims + [grid_corners,]), {"units" : "degrees_east"} ),
                grid_corner_lon = ( [*dim_names, "grid_corners"], grid_corner_lon.reshape(grid_dims + [grid_corners,]), {"units" : "degrees_east"} ),
                grid_area = ( [*dim_names], grid_area.reshape(grid_dims), {"units" : "radians^2"} ),
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

def test_rotation():

    rotation_angles = np.linspace(0, 1, 12)[1:-3] * 2 * np.pi
    rotate_vec = normalize(np.array([-1, 1, -1]))
    r_cartesian = np.stack(
        [
            np.array([1, 1, 0]),
            np.array([-1, 0.5, 0.2]),
            np.array([-0.8, 1.1, -0.7]),
            np.array([0.8, -1.7, 0.1]),
        ],
        axis=-1
    )
    new_pts = []
    for i, rotation_angle in enumerate(rotation_angles):
        new_pts.append(rotate_along_a_given_vector(
            r_cartesian = r_cartesian,
            rotate_vec = rotate_vec,
            angle_rad = rotation_angle,
        ))
 
    new_pts = np.stack(new_pts, axis=-1)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d'})
    ax.view_init(azim=-30, elev=45, roll=0) 

    ax.scatter(0, 0, 0, color="black", s=10)
    ax.quiver(0, 0, 0, *rotate_vec, color='blue', arrow_length_ratio=0.2, colors="black")
    
    ax.scatter(*r_cartesian, color="red", s=10)
    for k in range(r_cartesian.shape[1]): # the k-th pointn
        _pts = new_pts[:, k, :] # Result: 0th=x,y,z 1th=rotations
        ax.scatter(_pts[0, :], _pts[1, :], _pts[2, :])

    ax.set_xlabel("x-direction")
    ax.set_ylabel("y-direction")
    ax.set_zlabel("z-direction")
    ax.set_xlim([-2, 2])
    ax.set_ylim([-2, 2])
    ax.set_zlim([-2, 2])
    ax.set_aspect('equal')
    ax.set_title("Generic LLC")
   
    plt.show()

def print_worldmap():
    
    import numpy as np
  
    resolutions = [1, 4, 5]#[1, 2, 3, 4]
    rotating_gaussian_grids = {}

    rotation_along_longitude_degree = -42 + 90
    rotation_degree = 12.0
    for resolution in resolutions: 
        print(f"Generating grid resolution = {resolution}...")
        output_file_SCRIP = f"rotating_gaussian_grid_{resolution:.2f}deg.SCRIP.nc"
        output_file = f"rotating_gaussian_grid_{resolution:.2f}deg.nc"
        rotating_gaussian_grid = generate_rotating_gaussian_grid(
            lat_bounds = np.linspace(-90, 90, int(180/resolution+1)),
            lon_bounds = np.linspace(0, 360, int(360/resolution+1)),
            rotation_along_longitude_degree = rotation_along_longitude_degree,
            rotation_degree = rotation_degree,
        )
        print("Writing to file: ", output_file)
        write_to_SCRIP_grid_file(rotating_gaussian_grid, output_file, flatten=False)
        print("Writing to file: ", output_file_SCRIP)
        write_to_SCRIP_grid_file(rotating_gaussian_grid, output_file_SCRIP, flatten=True)
        rotating_gaussian_grids[resolution] = rotating_gaussian_grid

    import matplotlib.pyplot as plt 
    from matplotlib.colors import ListedColormap

    fig, ax = plt.subplots(len(resolutions)//2, 2, figsize=(10, 6))

    my_cmap = ListedColormap(['white', 'gray'])

    for i, resolution in enumerate(rotating_gaussian_grids.keys()):
        _ax = ax.flatten()[i]
        rotating_gaussian_grid = rotating_gaussian_grids[resolution]
        _ax.imshow(rotating_gaussian_grid.binary_mask, cmap=my_cmap)
        _ax.set_title(f"({'abcdefg'[i]}) Rotated land-sea mask resolution = ${resolution}^{{\\circ}}$")
        _ax.invert_yaxis()

    fig.suptitle(f"Rotate ${rotation_degree:.1f}^{{\\circ}}$ along longitude ${rotation_along_longitude_degree:.1f}^{{\\circ}}$ (right-hand rule)")

    fig.savefig("rotating_gaussian_landsea_mask.svg")
    plt.show()
    
if __name__ == "__main__":
    #test_rotation()
    #test_output_SCRIP_file()
    print_worldmap()
     
