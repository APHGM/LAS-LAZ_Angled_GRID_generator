# LAZ Grid Generator Walkthrough

This document describes the `laz_grid_generator_gui.py` script, which allows users to generate a rotated grid of elevation points from a LAZ/LAS point cloud file.

## Requirements

The script requires the following Python libraries:
- `laspy` (with `lazrs` backend for LAZ support)
- `numpy`
- `scipy`
- `tkinter` (usually included with Python)

Installation:
```bash
pip install laspy[lazrs] numpy scipy
```

## How to Use

1. **Run the Script**:
   Execute the script from your terminal or IDE:
   ```bash
   python laz_grid_generator_gui.py
   ```

2. **Select File**:
   - Click **Browse** to select your `.laz` or `.las` file.
   - The script will automatically scan the file for available classifications (this may take a moment for large files).

3. **Set Parameters**:
   - **Grid Size**: Enter the desired grid cell size in meters (e.g., `1.0`).
   - **Rotation Angle**: Enter the angle in degrees (e.g., `-6.0`). 
     - Positive angles rotate Counter-Clockwise.
   - **Max Distance**: Maximum distance (in meters) from a grid point to a LAZ point to be considered valid. Points further away will be set to `0.0` elevation. Default is `2.0`.

4. **Select Classifications**:
   - By default, **Class 2 (Ground)** is selected if present.
   - You can select multiple classes. Only points matching the selected classes will be used for elevation interpolation.

5. **Generate Output**:
   - Click **GENERATE CSV**.
   - Choose a location to save the output `.csv` file.
   - The script will process the file, generating a grid covering the extent of the points, rotated by the specified angle.
   - **Note**: For very large files (e.g., >1GB), processing may take several minutes depending on your system's RAM and CPU.

## implementation Details

- **Memory Optimization**: The script reads LAZ files in chunks to minimize memory usage during the loading phase.
- **Rotation Logic**: 
  1. Data is logically rotated by `-angle` to align with the grid axes.
  2. An axis-aligned grid is generated to cover the extent.
  3. The grid points are rotated back by `+angle` to their true global coordinates.
- **Interpolation**: Elevation values are derived using **Nearest Neighbor** interpolation from the nearest available ground point.

## Logic Flowchart

![Flowchart](/C:/Users/Administrator/.gemini/antigravity/brain/558d053d-cbb0-4736-ab70-c363423309f6/flowchart.md)

## Verification

The script has been tested with large LAZ files. 
- **Bounds Check**: The generated grid correctly covers the rotated bounding box of the input data.
- **Output**: The resulting CSV contains `X, Y, Z` columns suitable for import into CAD or GIS software.
