import os
import pickle
import numpy as np
import pyvista as pv

# Define the folder where your pickle files are located
folder_path = '/Users/wenyun/Desktop/research/Speckle_project/speckle_git/Speckle_project/speckle/FUS_condensate/June_fus/20230609/#4'

# Load the pickle files and sort them
pickle_files = [f for f in os.listdir(folder_path) if f.endswith('.pkl')]
pickle_files.sort()  # make sure this sorts in the order you want

# Load the data from pickle files into a list
slices = []
for file_name in pickle_files:
    with open(os.path.join(folder_path, file_name), 'rb') as file:
        slice = pickle.load(file)
        slices.append(slice)

# Stack the slices into a 3D array
volume_data = np.stack(slices, axis=2)

# Create a PyVista grid where each cell represents a pixel/voxel
grid = pv.UniformGrid()

# Set the grid dimensions based on the number of voxels/cells in each axis
# Here, the dimensions should match the shape of the volume data
grid.dimensions = volume_data.shape

# Set the spacing between points in the grid
grid.spacing = (1, 1, 8.7)  # Assuming each slice is 1 unit apart

# Assign the numpy array data to the PyVista grid
# The number of points is the product of the dimensions of the grid
num_points = grid.dimensions[0] * grid.dimensions[1] * grid.dimensions[2]
if volume_data.size != num_points:
    raise ValueError(f"The volume data size {volume_data.size} does not match the expected number of points {num_points}")

grid.point_data['values'] = volume_data.flatten(order='F')  # Flatten the array for PyVista

# Define the color limit range from 0 to 2
clim = (0, 2)

# # Create a linear opacity transfer function (array) for increased transparency
# opacity_tf = np.linspace(0.5, 1, 256)  # Linear ramp from 0 (transparent) to 1 (opaque)

# # Visualize the volume with the specified color limits and opacity transfer function
# grid.plot(volume=True, cmap='viridis', clim=clim, opacity=opacity_tf)

# Visualize the volume with the specified color limits and opacity transfer function
grid.plot(volume=True, cmap='viridis', clim=clim)