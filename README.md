# LAZ Grid Generator

A Python application with a Graphical User Interface (GUI) to generate rotated grid points from LAZ/LAS point cloud files. This tool is designed to extract Ground elevations at intervals along a user-specified rotation angle.

## Features

- **GUI Interface**: Easy-to-use interface built with Tkinter.
- **LAZ/LAS Support**: Reads standard point cloud formats using `laspy`.
- **Rotated Grid**: Generates a grid aligned to a specific angle (e.g., -6 degrees).
- **Classification Filtering**: Allows selection of specific classes (e.g., Ground) to use for elevation data.
- **Elevation Interpolation**: Uses Nearest Neighbor interpolation to assign Z values to grid points.
- **Valid Distance Check**: Sets elevation to `0.0` if the grid point is too far from any source point (configurable).
- **CSV Export**: Outputs `X, Y, Z` coordinates to a standard CSV file.

## Installation

1. **Install Python**: Ensure Python 3.8+ is installed.
2. **Install Dependencies**:
   Open a terminal and run:
   ```bash
   pip install laspy[lazrs] numpy scipy
   ```

## Usage

1. Run the script:
   ```bash
   python laz_grid_generator_gui.py
   ```
2. **Select File**: Click "Browse" to choose your `.laz` or `.las` file.
3. **Parameters**:
   - **Grid Size**: Distance between grid points (e.g., `1.0` meters).
   - **Rotation Angle**: Angle to rotate the grid logic. 
     - Positive = Counter-Clockwise.
     - Negative = Clockwise.
   - **Max Distance**: Maximum distance a grid point can be from a source point to get a valid elevation. Points further away are set to `0.0`.
4. **Classifications**: Select the classes you want to use (usually "2 - Ground").
5. **Generate**: Click "GENERATE CSV" and choose a save location.

## Logic Flow

1. **Read**: The script reads the LAZ file (in chunks for memory efficiency) and filters points by the selected classification.
2. **Rotate (Virtual)**: The source points are mathematically rotated by `-Angle` to align them with the X/Y axes.
3. **Bounding Box**: A bounding box is calculated for the rotated data.
4. **Grid Generation**: A regular grid is created within this bounding box.
5. **Rotate Back**: The grid points are rotated by `+Angle` to return them to the original global coordinate system.
6. **Interpolate**: 
   - A KDTree is built from the source points.
   - For each grid point, the nearest source point is found.
   - If distance < `Max Distance`, the Z value is assigned.
   - Else, Z is set to `0.0`.
7. **Export**: The resulting X, Y, Z data is saved to CSV.

## Troubleshooting

- **"No classifications found"**: Ensure the LAZ file works and has standard classifications.
- **Memory Errors**: Processing very large files (>20 million points) may require significant RAM. The script uses chunked reading, but the interpolation step requires all *filtered* points in memory.
