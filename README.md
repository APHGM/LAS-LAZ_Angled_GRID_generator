# LAZ / LAS Grid Generator — v0.8

A Python GUI application for generating survey-grade grid points from LAZ/LAS point cloud files.  
Supports angle-based grids, DXF centreline chainage grids, and DXF point elevation updates.

---

## Features

| Feature | Details |
|---|---|
| **Three grid modes** | Angle-based, DXF Centreline (chainage), Update DXF Point Elevations |
| **Single file or folder** | Process one LAZ/LAS file or a full folder of tiled files |
| **DXF centreline** | Generates cross-sections perpendicular to a polyline with smooth corner handling |
| **Polyline picker** | Lists all polylines in the DXF (layer, length, vertices) for easy selection |
| **UCS transform** | Offset X/Y + rotation to align local-UCS DXF coordinates with world LAZ coordinates |
| **Classification filter** | Select specific point classes (Ground, Vegetation, etc.) per file |
| **CSV + DXF export** | Output grid points as CSV and/or DXF POINT entities |
| **Elevation update** | Read POINT entities from a DXF, assign elevations from LAZ, output XYZ CSV and/or updated DXF |
| **Standalone EXE** | PyInstaller build via `build_clean.bat` |

---

## Requirements

```
Python 3.8+
laspy[lazrs]
numpy
scipy
ezdxf
pillow
```

Install all:
```bash
pip install -r requirements_clean.txt
```

---

## Usage

### Run from source

```bash
python laz_grid_generator_gui_v0.8.py
```

### Build standalone EXE

```bash
build_clean.bat
```

Output: `dist\LAZ_Grid_Generator_Clean_v0.8.exe`

---

## Grid Modes

### 1. Angle-based Grid
Generates a regular grid rotated to a specified angle.  
Useful when the survey area has a known orientation.

- **Grid Size (m)** — spacing between grid points  
- **Rotation Angle (deg)** — positive = CCW, negative = CW

### 2. DXF Centreline (Chainage)
Generates cross-section points perpendicular to a road/track centreline polyline.

- Load a DXF file containing the centreline polyline  
- Pick the correct polyline from the list  
- Set chainage spacing, cross-section half-width and point spacing  
- Use **DXF Local UCS → World Transform** if the DXF was drawn in a local coordinate system

### 3. Update DXF Point Elevations
Updates Z values of POINT entities in an existing DXF using elevations from LAZ/LAS data.

- Load the DXF file containing the POINT entities  
- Select output: XYZ CSV, updated DXF, or both

---

## Output Options

- **CSV** — `X, Y, Z` (plus `Chainage, Offset` for DXF centreline mode)  
- **DXF** — grid points as POINT entities on layer `GRID_POINTS` (tick *Also export DXF* before generating)

---

## Coordinate System Notes

If the DXF centreline was drawn in a local UCS and the LAZ data is in a national/world grid, use the **DXF Local UCS → World Transform** fields:

- **Offset X / Offset Y** — translation from local origin to world coordinates  
- **UCS Rotation (deg)** — rotation of the local UCS relative to world north

The tool will warn with suggested offset values if a coordinate mismatch is detected automatically.
