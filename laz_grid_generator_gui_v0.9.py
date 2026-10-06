import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import laspy
import numpy as np
import threading
import os
import glob

try:
    from lazrs import LazrsError  # type: ignore[import-untyped]
except ImportError:
    LazrsError = None

try:
    import ezdxf
    HAS_EZDXF = True
except ImportError:
    HAS_EZDXF = False


# ---------------------------------------------------------------------------
# DXF helpers
# ---------------------------------------------------------------------------

def apply_ucs_transform(pts, offset_x, offset_y, rotation_deg):
    """
    Transform DXF local-UCS coordinates to world coordinates.
    1. Rotate by rotation_deg (CCW) around the local origin.
    2. Translate by (offset_x, offset_y).
    All three default to 0 → no-op.
    """
    if rotation_deg != 0.0:
        theta = np.radians(rotation_deg)
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        x_rot = pts[:, 0] * cos_t - pts[:, 1] * sin_t
        y_rot = pts[:, 0] * sin_t + pts[:, 1] * cos_t
        pts = np.column_stack((x_rot, y_rot))
    if offset_x != 0.0 or offset_y != 0.0:
        pts = pts + np.array([offset_x, offset_y])
    return pts


def get_all_polylines_from_dxf(dxf_path):
    """
    Return a list of dicts, one per polyline found in modelspace.
    Each dict has: vertices (np array Nx2), layer, arc_length, n_vertices, entity_type
    """
    if not HAS_EZDXF:
        raise ImportError("ezdxf required.  pip install ezdxf")

    doc = ezdxf.readfile(dxf_path)  # type: ignore[attr-defined]
    msp = doc.modelspace()
    results = []

    for entity in msp:
        pts = []
        layer = getattr(entity.dxf, "layer", "0")
        etype = entity.dxftype()

        if etype == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in entity.get_points()]  # type: ignore[union-attr]
        elif etype == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]  # type: ignore[union-attr]

        if len(pts) >= 2:
            arr = np.array(pts, dtype=np.float64)
            d = np.diff(arr, axis=0)
            arc_len = float(np.sum(np.hypot(d[:, 0], d[:, 1])))
            results.append({
                "vertices":   arr,
                "layer":      layer,
                "arc_length": arc_len,
                "n_vertices": len(pts),
                "etype":      etype,
            })

    return results


# ---------------------------------------------------------------------------
# Centerline geometry
# ---------------------------------------------------------------------------

def build_parameterization(centerline):
    """Arc-length param, removing duplicate consecutive vertices."""
    keep = np.ones(len(centerline), dtype=bool)
    for i in range(1, len(centerline)):
        if np.hypot(*(centerline[i] - centerline[i - 1])) < 1e-9:
            keep[i] = False
    pts = centerline[keep]
    diffs = np.diff(pts, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
    return pts, diffs, seg_len, cumlen


def generate_chainage_grid(centerline, chainage_spacing, cross_width, cross_spacing):
    """
    Grid points perpendicular to centerline. +offset = left of travel.

    Tangent direction is smoothed at each vertex by averaging the incoming and
    outgoing segment tangents, then linearly interpolated along each segment.
    This prevents abrupt perpendicular-direction jumps at bends/corners.
    """
    pts, diffs, seg_len, cumlen = build_parameterization(centerline)
    total_length = cumlen[-1]
    n_pts = len(pts)

    # Unit tangent for each segment
    seg_tan = diffs / seg_len[:, np.newaxis]          # (n_segs, 2)

    # Smoothed tangent at each vertex = normalised average of adjacent segments
    vtx_tan = np.zeros((n_pts, 2))
    vtx_tan[0]  = seg_tan[0]
    vtx_tan[-1] = seg_tan[-1]
    for i in range(1, n_pts - 1):
        avg = seg_tan[i - 1] + seg_tan[i]
        norm = np.hypot(avg[0], avg[1])
        vtx_tan[i] = avg / norm if norm > 1e-12 else seg_tan[i]

    chainages = np.arange(0.0, total_length + chainage_spacing * 0.5, chainage_spacing)
    chainages = chainages[chainages <= total_length + 1e-6]

    n_steps = int(np.ceil(cross_width / cross_spacing))
    offsets = np.linspace(-n_steps * cross_spacing, n_steps * cross_spacing, 2 * n_steps + 1)
    offsets = offsets[np.abs(offsets) <= cross_width + 1e-9]

    all_x, all_y, all_ch, all_off = [], [], [], []
    for ch in chainages:
        idx = int(np.clip(np.searchsorted(cumlen, ch, side="right") - 1, 0, len(pts) - 2))
        t = (ch - cumlen[idx]) / seg_len[idx] if seg_len[idx] > 1e-12 else 0.0

        # Centreline position
        cx = pts[idx, 0] + t * diffs[idx, 0]
        cy = pts[idx, 1] + t * diffs[idx, 1]

        # Interpolate tangent between the two vertex tangents of this segment,
        # then re-normalise so the perpendicular is always unit-length
        tx_raw = (1.0 - t) * vtx_tan[idx, 0] + t * vtx_tan[idx + 1, 0]
        ty_raw = (1.0 - t) * vtx_tan[idx, 1] + t * vtx_tan[idx + 1, 1]
        norm = np.hypot(tx_raw, ty_raw)
        if norm > 1e-12:
            tx, ty = tx_raw / norm, ty_raw / norm
        else:
            tx, ty = seg_tan[idx]

        px, py = -ty, tx          # perpendicular: 90° CCW from tangent

        for off in offsets:
            all_x.append(cx + off * px)
            all_y.append(cy + off * py)
            all_ch.append(ch)
            all_off.append(off)

    return np.array(all_x), np.array(all_y), np.array(all_ch), np.array(all_off)


def generate_angle_grid(points_x, points_y, grid_size, angle_deg):
    """Axis-aligned grid rotated by angle_deg."""
    theta_rad = np.radians(angle_deg)
    cos_t, sin_t = np.cos(-theta_rad), np.sin(-theta_rad)
    x_rot = points_x * cos_t - points_y * sin_t
    y_rot = points_x * sin_t + points_y * cos_t

    u_start = np.floor(x_rot.min() / grid_size) * grid_size
    u_end   = np.ceil(x_rot.max()  / grid_size) * grid_size
    v_start = np.floor(y_rot.min() / grid_size) * grid_size
    v_end   = np.ceil(y_rot.max()  / grid_size) * grid_size

    n_est = ((u_end - u_start) / grid_size) * ((v_end - v_start) / grid_size)
    if n_est > 50_000_000:
        raise ValueError(f"Grid too large (~{int(n_est):,} pts). Increase grid size.")

    gu, gv = np.meshgrid(
        np.arange(u_start, u_end + grid_size, grid_size),
        np.arange(v_start, v_end + grid_size, grid_size),
    )
    gu, gv = gu.flatten(), gv.flatten()
    cos_inv, sin_inv = np.cos(theta_rad), np.sin(theta_rad)
    return gu * cos_inv - gv * sin_inv, gu * sin_inv + gv * cos_inv


# ---------------------------------------------------------------------------
# LAZ reading helper
# ---------------------------------------------------------------------------

def _read_single_laz(laz_path, selected_classes, progress_callback, base_pct, range_pct):
    px_list, py_list, pz_list = [], [], []
    with laspy.open(laz_path) as f:
        total_pts = f.header.point_count
        processed = 0
        try:
            for chunk in f.chunk_iterator(2_000_000):
                if selected_classes is not None:
                    mask = np.isin(chunk.classification, selected_classes)
                    if np.any(mask):
                        px_list.append(chunk.x[mask])
                        py_list.append(chunk.y[mask])
                        pz_list.append(chunk.z[mask])
                else:
                    px_list.append(chunk.x)
                    py_list.append(chunk.y)
                    pz_list.append(chunk.z)
                processed += len(chunk)
                if total_pts > 0:
                    pct = base_pct + int(processed / total_pts * range_pct)
                    progress_callback(
                        f"  {os.path.basename(laz_path)}: {int(processed/total_pts*100)}%", pct
                    )
        except Exception as e:
            is_lazrs = (LazrsError and isinstance(e, LazrsError)) or \
                       "failed to fill whole buffer" in str(e)
            if is_lazrs:
                if not px_list:
                    raise e
            else:
                raise e
    if not px_list:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(px_list), np.concatenate(py_list), np.concatenate(pz_list)


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process(
    laz_files,
    grid_mode,           # "angle" | "dxf"
    grid_size, angle_deg,
    centerline_pts,      # np array Nx2, already resolved (None if angle mode)
    chainage_spacing, cross_width, cross_spacing,
    max_dist, selected_classes,
    output_path,
    out_dxf_path,        # None or str — optional DXF of grid points
    progress_callback, finish_callback,
):
    try:
        n_tiles = len(laz_files)

        # ── Read LAZ ─────────────────────────────────────────────────────────
        progress_callback(f"Reading {n_tiles} LAZ/LAS file(s)...", 0)
        all_px, all_py, all_pz = [], [], []
        per_tile = max(1, int(40 / n_tiles))

        for i, laz_path in enumerate(laz_files):
            base = i * per_tile
            progress_callback(
                f"Reading tile {i+1}/{n_tiles}: {os.path.basename(laz_path)}", base
            )
            px, py, pz = _read_single_laz(
                laz_path, selected_classes, progress_callback, base, per_tile
            )
            if len(px):
                all_px.append(px); all_py.append(py); all_pz.append(pz)

        if not all_px:
            raise ValueError("No points found with the selected classifications.")

        points_x = np.concatenate(all_px)
        points_y = np.concatenate(all_py)
        points_z = np.concatenate(all_pz)

        progress_callback(f"Loaded {len(points_x):,} points from {n_tiles} tile(s).", 42)

        laz_xmin, laz_xmax = points_x.min(), points_x.max()
        laz_ymin, laz_ymax = points_y.min(), points_y.max()

        # ── Generate grid ─────────────────────────────────────────────────────
        if grid_mode == "angle":
            progress_callback(f"Generating angle grid (angle={angle_deg}°, cell={grid_size}m)...", 45)
            grid_x, grid_y = generate_angle_grid(points_x, points_y, grid_size, angle_deg)
            grid_ch = grid_off = np.zeros(len(grid_x))
            extra_cols = False
            total_length = 0.0
            n_sections = 0
            progress_callback(f"Grid: {len(grid_x):,} points", 50)

        else:  # dxf
            progress_callback("Generating chainage grid from centreline...", 45)
            _, _, _, cumlen = build_parameterization(centerline_pts)
            total_length = cumlen[-1]
            n_sections = int(total_length / chainage_spacing) + 1

            grid_x, grid_y, grid_ch, grid_off = generate_chainage_grid(
                centerline_pts, chainage_spacing, cross_width, cross_spacing
            )
            extra_cols = True

            if len(grid_x) > 10_000_000:
                raise ValueError(
                    f"Grid too large ({len(grid_x):,} pts). "
                    "Increase chainage spacing or reduce cross-section size."
                )
            progress_callback(f"Grid: {len(grid_x):,} points", 50)

            # ── Bounding-box overlap check ────────────────────────────────────
            gx_min, gx_max = grid_x.min(), grid_x.max()
            gy_min, gy_max = grid_y.min(), grid_y.max()
            x_ok = gx_min <= laz_xmax and gx_max >= laz_xmin
            y_ok = gy_min <= laz_ymax and gy_max >= laz_ymin
            if not (x_ok and y_ok):
                raise ValueError(
                    "Coordinate mismatch: the DXF centreline and LAZ data are in different "
                    "coordinate systems (they do not overlap).\n\n"
                    f"DXF grid extent:  X [{gx_min:.1f} – {gx_max:.1f}]  "
                    f"Y [{gy_min:.1f} – {gy_max:.1f}]\n"
                    f"LAZ data extent:  X [{laz_xmin:.1f} – {laz_xmax:.1f}]  "
                    f"Y [{laz_ymin:.1f} – {laz_ymax:.1f}]\n\n"
                    "If the DXF was drawn in a local UCS, enter the world-coordinate "
                    "offset in the 'DXF Local UCS → World Transform' fields:\n"
                    f"  Offset X ≈ {laz_xmin:.1f} − {gx_min:.1f} = "
                    f"{laz_xmin - gx_min:.1f}\n"
                    f"  Offset Y ≈ {laz_ymin:.1f} − {gy_min:.1f} = "
                    f"{laz_ymin - gy_min:.1f}"
                )

        # ── KD-tree elevation assignment ──────────────────────────────────────
        progress_callback(f"Building KD-tree for {len(points_x):,} LAZ points...", 55)
        from scipy.spatial import cKDTree  # type: ignore[import-untyped]
        tree = cKDTree(np.column_stack((points_x, points_y)))

        progress_callback("Interpolating elevations (nearest neighbour)...", 65)
        dists, idxs = tree.query(np.column_stack((grid_x, grid_y)), k=1)
        grid_z = points_z[idxs]
        mask_far = dists > max_dist
        grid_z[mask_far] = 0.0
        n_zero = int(np.sum(mask_far))

        if n_zero == len(grid_z):
            progress_callback(
                "WARNING: ALL grid points are beyond max_dist — check max_dist value "
                "or verify the correct polyline is selected.", 75
            )
        else:
            progress_callback(
                f"{n_zero:,} grid points set to Z=0 (>{max_dist} m from data).", 75
            )

        # ── Write CSV ─────────────────────────────────────────────────────────
        progress_callback("Writing output CSV...", 80)
        if extra_cols:
            header = "X,Y,Z,Chainage,Offset"
            data = np.column_stack((grid_x, grid_y, grid_z, grid_ch, grid_off))
        else:
            header = "X,Y,Z"
            data = np.column_stack((grid_x, grid_y, grid_z))

        total_rows = len(data)
        with open(output_path, "w", newline="") as f_out:
            f_out.write(header + "\n")
            wc = 100_000
            for i in range(0, total_rows, wc):
                chunk = data[i:i+wc]
                np.savetxt(f_out, chunk, delimiter=",", fmt="%.3f")
                done = i + len(chunk)
                progress_callback(
                    f"Writing CSV... {int(done/total_rows*100)}%",
                    80 + int(done/total_rows*15),
                )

        # ── Write DXF (grid points) ───────────────────────────────────────────
        if out_dxf_path:
            progress_callback("Writing grid points DXF...", 95)
            write_points_dxf(out_dxf_path, grid_x, grid_y, grid_z)

        if grid_mode == "angle":
            detail = f"Rotation angle : {angle_deg}°\nGrid cell size : {grid_size} m\n"
        else:
            detail = (
                f"Centreline length  : {total_length:.1f} m\n"
                f"Cross-sections     : {n_sections}\n"
            )

        saved_msg = f"CSV:  {output_path}"
        if out_dxf_path:
            saved_msg += f"\nDXF:  {out_dxf_path}"

        finish_callback(
            True,
            f"Grid generated successfully!\n\n"
            f"LAZ tiles read   : {n_tiles}\n"
            f"Total LAZ points : {len(points_x):,}\n"
            + detail +
            f"Grid points      : {total_rows:,}\n"
            f"Zero-elev points : {n_zero:,}\n\n"
            f"Saved to:\n{saved_msg}",
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        finish_callback(False, f"An error occurred:\n{str(e)}")


# ---------------------------------------------------------------------------
# DXF point elevation update helpers
# ---------------------------------------------------------------------------

def read_dxf_points(dxf_path):
    """
    Read all POINT entities from a DXF file.
    Returns an (N, 3) array of (x, y, z_original).
    """
    if not HAS_EZDXF:
        raise ImportError("ezdxf required.  pip install ezdxf")
    doc = ezdxf.readfile(dxf_path)  # type: ignore[attr-defined]
    msp = doc.modelspace()
    pts = []
    for entity in msp:
        if entity.dxftype() == "POINT":
            loc = entity.dxf.location
            pts.append([loc.x, loc.y, loc.z])
    if not pts:
        raise ValueError(
            "No POINT entities found in the DXF file.\n"
            "Make sure the file contains POINT (not INSERT or BLOCK) entities."
        )
    return np.array(pts, dtype=np.float64)


def write_dxf_with_updated_z(input_dxf_path, output_dxf_path, new_z_values):
    """
    Read input DXF, update each POINT entity's Z with the corresponding value
    from new_z_values (same order as read_dxf_points), and save to output path.
    """
    doc = ezdxf.readfile(input_dxf_path)  # type: ignore[attr-defined]
    msp = doc.modelspace()
    idx = 0
    for entity in msp:
        if entity.dxftype() == "POINT" and idx < len(new_z_values):
            loc = entity.dxf.location
            entity.dxf.location = (loc.x, loc.y, float(new_z_values[idx]))
            idx += 1
    doc.saveas(output_dxf_path)


def process_update_elevations(
    laz_files,
    dxf_points_path,
    max_dist, selected_classes,
    out_xyz,   # output CSV path or None
    out_dxf,   # output DXF path or None
    progress_callback, finish_callback,
):
    try:
        n_tiles = len(laz_files)

        # ── Read DXF points ───────────────────────────────────────────────────
        progress_callback("Reading POINT entities from DXF...", 0)
        dxf_pts = read_dxf_points(dxf_points_path)
        n_pts = len(dxf_pts)
        progress_callback(f"Found {n_pts:,} POINT entities in DXF.", 5)

        # ── Read LAZ tiles ────────────────────────────────────────────────────
        progress_callback(f"Reading {n_tiles} LAZ/LAS file(s)...", 8)
        all_px, all_py, all_pz = [], [], []
        per_tile = max(1, int(45 // n_tiles))

        for i, laz_path in enumerate(laz_files):
            base = 8 + i * per_tile
            progress_callback(
                f"Reading tile {i+1}/{n_tiles}: {os.path.basename(laz_path)}", base
            )
            px, py, pz = _read_single_laz(
                laz_path, selected_classes, progress_callback, base, per_tile
            )
            if len(px):
                all_px.append(px); all_py.append(py); all_pz.append(pz)

        if not all_px:
            raise ValueError("No LAZ points found with the selected classifications.")

        points_x = np.concatenate(all_px)
        points_y = np.concatenate(all_py)
        points_z = np.concatenate(all_pz)

        progress_callback(f"Loaded {len(points_x):,} LAZ points. Building KD-tree...", 56)

        # ── KD-tree lookup ────────────────────────────────────────────────────
        from scipy.spatial import cKDTree  # type: ignore[import-untyped]
        tree = cKDTree(np.column_stack((points_x, points_y)))

        progress_callback("Updating point elevations...", 65)
        dists, idxs = tree.query(dxf_pts[:, :2], k=1)
        new_z = points_z[idxs].copy()
        mask_far = dists > max_dist
        new_z[mask_far] = 0.0
        n_zero = int(np.sum(mask_far))
        progress_callback(
            f"{n_zero:,} points set to Z=0 (>{max_dist} m from LAZ data).", 72
        )

        # ── Write XYZ CSV ─────────────────────────────────────────────────────
        if out_xyz:
            progress_callback("Writing XYZ CSV...", 78)
            data = np.column_stack((dxf_pts[:, 0], dxf_pts[:, 1], new_z))
            with open(out_xyz, "w", newline="") as f:
                f.write("X,Y,Z\n")
                np.savetxt(f, data, delimiter=",", fmt="%.3f")
            progress_callback("XYZ CSV written.", 88)

        # ── Write updated DXF ─────────────────────────────────────────────────
        if out_dxf:
            progress_callback("Writing updated DXF...", 90)
            write_dxf_with_updated_z(dxf_points_path, out_dxf, new_z)
            progress_callback("Updated DXF written.", 98)

        outputs = []
        if out_xyz: outputs.append(f"XYZ CSV : {out_xyz}")
        if out_dxf: outputs.append(f"DXF     : {out_dxf}")

        finish_callback(
            True,
            f"Elevation update complete!\n\n"
            f"POINT entities updated : {n_pts:,}\n"
            f"LAZ tiles read         : {n_tiles}\n"
            f"Total LAZ points       : {len(points_x):,}\n"
            f"Zero-elevation points  : {n_zero:,}\n\n"
            + "\n".join(outputs),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        finish_callback(False, f"An error occurred:\n{str(e)}")


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Grid DXF export helper
# ---------------------------------------------------------------------------

def write_points_dxf(output_path, x_arr, y_arr, z_arr, layer="GRID_POINTS"):
    if not HAS_EZDXF:
        raise ImportError("ezdxf required.  pip install ezdxf")
    doc = ezdxf.new(dxfversion="R2010")  # type: ignore[attr-defined]
    msp = doc.modelspace()
    for x, y, z in zip(x_arr, y_arr, z_arr):
        msp.add_point((float(x), float(y), float(z)), dxfattribs={"layer": layer})
    doc.saveas(output_path)


# ---------------------------------------------------------------------------
# Hybrid DTM helpers
# ---------------------------------------------------------------------------

def detect_csv_columns(csv_path):
    """
    Inspect the first non-empty line of a CSV and return (col_x, col_y, col_z, has_header).
    Tries header-name matching first, then falls back to guessing from value magnitudes.
    Returns (col_x, col_y, col_z) as 0-based column indices.
    """
    _X_NAMES = {"x", "easting", "east", "e", "lon", "longitude", "long"}
    _Y_NAMES = {"y", "northing", "north", "n", "lat", "latitude"}
    _Z_NAMES = {"z", "rl", "elev", "elevation", "height", "h", "level", "reduced level"}

    with open(csv_path, newline="") as f:
        lines = [l.strip().rstrip(",") for l in f if l.strip().rstrip(",")]

    if not lines:
        return 1, 2, 3  # safe default

    first = [p.strip() for p in lines[0].split(",")]

    # Try header matching
    lower = [h.lower() for h in first]
    col_x = col_y = col_z = None
    for i, h in enumerate(lower):
        if h in _X_NAMES and col_x is None:
            col_x = i
        elif h in _Y_NAMES and col_y is None:
            col_y = i
        elif h in _Z_NAMES and col_z is None:
            col_z = i

    if col_x is not None and col_y is not None and col_z is not None:
        return col_x, col_y, col_z

    # No header — guess from value magnitudes using first data row
    data_line = lines[0]
    try:
        float(first[0])  # numeric → no header
    except ValueError:
        data_line = lines[1] if len(lines) > 1 else lines[0]

    vals = []
    for p in data_line.split(","):
        try:
            vals.append(float(p.strip()))
        except ValueError:
            vals.append(None)

    # Heuristic: Z (RL/elevation) is typically 0–5000 m;
    # X (Easting) and Y (Northing) are typically >10 000 or very similar large numbers.
    # The smaller large number pair is often [X, Y]; the small one is Z.
    numeric = [(i, v) for i, v in enumerate(vals) if v is not None]
    if len(numeric) >= 3:
        by_mag = sorted(numeric, key=lambda iv: abs(iv[1]))
        # smallest absolute value → Z; remaining two ordered by index → X, Y
        z_idx = by_mag[0][0]
        xy = sorted([iv[0] for iv in by_mag[1:3]])
        return xy[0], xy[1], z_idx

    return 1, 2, 3  # fallback


def read_survey_csv(csv_path, col_x=None, col_y=None, col_z=None):
    """
    Read ground survey CSV with flexible column order.
    col_x/col_y/col_z are 0-based column indices; if None they are auto-detected.
    Returns (N, 3) float64 array of [X, Y, Z].
    """
    if col_x is None or col_y is None or col_z is None:
        col_x, col_y, col_z = detect_csv_columns(csv_path)

    need = max(col_x, col_y, col_z) + 1
    pts = []
    with open(csv_path, newline="") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < need:
                continue
            try:
                x = float(parts[col_x])
                y = float(parts[col_y])
                z = float(parts[col_z])
                pts.append([x, y, z])
            except ValueError:
                continue  # skip header rows or bad lines
    if not pts:
        raise ValueError(
            "No valid survey points found in CSV.\n"
            f"Tried columns X={col_x}, Y={col_y}, Z={col_z} (0-based).\n"
            "Check the Column Mapping in the Hybrid DTM panel."
        )
    return np.array(pts, dtype=np.float64)


def points_in_polygon(pts, polygon):
    """
    Vectorised ray-casting point-in-polygon test.
    pts     : (N, 2)  –  query points
    polygon : (M, 2)  –  boundary vertices (open or closed)
    Returns boolean (N,) array.
    """
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    n = len(polygon)
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i, 0], polygon[i, 1]
        xj, yj = polygon[j, 0], polygon[j, 1]
        dy = yj - yi
        dy_safe = np.where(np.abs(dy) > 1e-12, dy, 1e-12)
        cond = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / dy_safe + xi
        )
        inside ^= cond
        j = i
    return inside


def generate_boundary_grid(boundary_poly, spacing):
    """
    Regular XY grid at `spacing` metres, clipped to boundary_poly.
    Returns (N, 2) array.
    """
    xs, ys = boundary_poly[:, 0], boundary_poly[:, 1]
    gx = np.arange(xs.min(), xs.max() + spacing * 0.5, spacing)
    gy = np.arange(ys.min(), ys.max() + spacing * 0.5, spacing)
    GX, GY = np.meshgrid(gx, gy)
    all_pts = np.column_stack((GX.ravel(), GY.ravel()))
    return all_pts[points_in_polygon(all_pts, boundary_poly)]


def write_landxml(output_path, x_arr, y_arr, z_arr, grid_spacing,
                  surface_name="Ground_Surface"):
    """
    Write a TIN surface to LandXML 1.2 using scipy Delaunay triangulation.
    Triangle edges longer than 2.5 × grid_spacing are filtered as boundary artefacts.
    LandXML P elements use Northing Easting Elevation order.
    """
    import datetime
    from scipy.spatial import Delaunay

    pts2d = np.column_stack((x_arr, y_arr))
    tri = Delaunay(pts2d)
    max_edge = grid_spacing * 2.5

    good = []
    for s in tri.simplices:
        p0, p1, p2 = pts2d[s[0]], pts2d[s[1]], pts2d[s[2]]
        if (np.hypot(*(p0 - p1)) <= max_edge and
                np.hypot(*(p1 - p2)) <= max_edge and
                np.hypot(*(p2 - p0)) <= max_edge):
            good.append(s)

    today = datetime.date.today().isoformat()
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<LandXML xmlns="http://www.landxml.org/schema/LandXML-1.2" '
        f'version="1.2" date="{today}" time="00:00:00">',
        "  <Surfaces>",
        f'    <Surface name="{surface_name}">',
        '      <Definition surfType="TIN">',
        "        <Pnts>",
    ]
    for i, (x, y, z) in enumerate(zip(x_arr, y_arr, z_arr), start=1):
        lines.append(f'          <P id="{i}">{y:.3f} {x:.3f} {z:.3f}</P>')
    lines += ["        </Pnts>", "        <Faces>"]
    for s in good:
        lines.append(f"          <F>{s[0]+1} {s[1]+1} {s[2]+1}</F>")
    lines += [
        "        </Faces>",
        "      </Definition>",
        "    </Surface>",
        "  </Surfaces>",
        "</LandXML>",
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_dxf_tin_mesh(output_path, x_arr, y_arr, z_arr, grid_spacing,
                       layer="DTM_SURFACE"):
    """
    Write a Delaunay TIN as 3DFACE entities in DXF.
    Edges longer than 2.5 × grid_spacing are filtered to remove boundary artefacts.
    """
    if not HAS_EZDXF:
        raise ImportError("ezdxf required.  pip install ezdxf")
    from scipy.spatial import Delaunay

    pts2d = np.column_stack((x_arr, y_arr))
    tri = Delaunay(pts2d)
    max_edge = grid_spacing * 2.5

    doc = ezdxf.new(dxfversion="R2010")  # type: ignore[attr-defined]
    msp = doc.modelspace()

    for s in tri.simplices:
        p0, p1, p2 = pts2d[s[0]], pts2d[s[1]], pts2d[s[2]]
        if (np.hypot(*(p0 - p1)) > max_edge or
                np.hypot(*(p1 - p2)) > max_edge or
                np.hypot(*(p2 - p0)) > max_edge):
            continue
        v0 = (float(x_arr[s[0]]), float(y_arr[s[0]]), float(z_arr[s[0]]))
        v1 = (float(x_arr[s[1]]), float(y_arr[s[1]]), float(z_arr[s[1]]))
        v2 = (float(x_arr[s[2]]), float(y_arr[s[2]]), float(z_arr[s[2]]))
        msp.add_3dface([v0, v1, v2, v2], dxfattribs={"layer": layer})

    doc.saveas(output_path)


def process_hybrid_dtm(
    laz_files,
    boundary_poly,      # (M, 2) array — closed boundary polygon
    csv_pts,            # (N, 3) array — ground survey [X, Y, Z]
    grid_spacing,
    csv_radius,         # use CSV elevation when nearest survey pt <= this distance
    laz_min_radius,     # max distance to nearest LAZ point; beyond this → Z=0
    selected_classes,
    out_csv, out_dxf_pts, out_landxml, out_dxf_mesh,
    progress_callback, finish_callback,
):
    try:
        from scipy.spatial import cKDTree  # type: ignore[import-untyped]

        # ── Generate grid within boundary ────────────────────────────────────
        progress_callback("Generating grid within boundary polygon...", 3)
        grid_pts = generate_boundary_grid(boundary_poly, grid_spacing)
        n_grid = len(grid_pts)
        if n_grid == 0:
            raise ValueError(
                "No grid points generated inside the boundary. "
                "Check that the correct polyline is selected and it forms a closed polygon."
            )
        progress_callback(f"Grid: {n_grid:,} points within boundary", 8)

        # ── Read LAZ ─────────────────────────────────────────────────────────
        n_tiles = len(laz_files)
        progress_callback(f"Reading {n_tiles} LAZ/LAS file(s)...", 10)
        all_px, all_py, all_pz = [], [], []
        per_tile = max(1, int(30 / n_tiles))
        for i, laz_path in enumerate(laz_files):
            base = 10 + i * per_tile
            progress_callback(
                f"Reading tile {i+1}/{n_tiles}: {os.path.basename(laz_path)}", base
            )
            px, py, pz = _read_single_laz(
                laz_path, selected_classes, progress_callback, base, per_tile
            )
            if len(px):
                all_px.append(px); all_py.append(py); all_pz.append(pz)

        if not all_px:
            raise ValueError("No LAZ points found with the selected classifications.")
        laz_x = np.concatenate(all_px)
        laz_y = np.concatenate(all_py)
        laz_z_all = np.concatenate(all_pz)
        progress_callback(f"Loaded {len(laz_x):,} LAZ points.", 42)

        # ── CSV elevation assignment ──────────────────────────────────────────
        progress_callback(
            f"Assigning elevations  "
            f"(CSV: {len(csv_pts):,} survey pts, "
            f"radius: {csv_radius} m)...", 44
        )
        csv_tree = cKDTree(csv_pts[:, :2])
        csv_dists, csv_idxs = csv_tree.query(grid_pts, k=1)
        use_csv = csv_dists <= csv_radius
        n_csv = int(np.sum(use_csv))
        progress_callback(
            f"CSV covers {n_csv:,} / {n_grid:,} grid points", 48
        )

        grid_z      = np.zeros(n_grid)
        grid_source = np.zeros(n_grid, dtype=np.int8)  # 0=LAZ, 1=CSV
        grid_z[use_csv]      = csv_pts[csv_idxs[use_csv], 2]
        grid_source[use_csv] = 1

        # ── LAZ minimum-Z assignment ──────────────────────────────────────────
        laz_mask = ~use_csv
        n_laz_pts = int(np.sum(laz_mask))
        n_zero = 0

        if n_laz_pts > 0:
            progress_callback(
                f"Finding nearest LAZ Z for {n_laz_pts:,} grid points "
                f"(max gap: {laz_min_radius} m)...", 50
            )
            laz_tree = cKDTree(np.column_stack((laz_x, laz_y)))
            query_pts = grid_pts[laz_mask]
            # Vectorised nearest-neighbour — fast, no chunking needed
            dists, idxs = laz_tree.query(query_pts, k=1)
            laz_z_vals = laz_z_all[idxs]
            too_far = dists > laz_min_radius
            laz_z_vals[too_far] = 0.0
            n_zero = int(np.sum(too_far))
            progress_callback(f"Nearest-Z assigned ({n_zero:,} pts beyond max gap)", 78)
            grid_z[laz_mask] = laz_z_vals

        grid_x = grid_pts[:, 0]
        grid_y = grid_pts[:, 1]
        progress_callback(
            f"Elevations assigned. "
            f"{n_zero:,} pts have Z=0 (no LAZ within {laz_min_radius} m).", 80
        )

        # ── Write outputs ─────────────────────────────────────────────────────
        saved = []

        if out_csv:
            progress_callback("Writing XYZ CSV...", 82)
            data = np.column_stack((grid_x, grid_y, grid_z, grid_source))
            with open(out_csv, "w", newline="") as f:
                f.write("X,Y,Z,Source\n")  # Source: 1=CSV survey, 0=LAZ nearest
                np.savetxt(f, data, delimiter=",", fmt=["%.3f", "%.3f", "%.3f", "%d"])
            saved.append(f"XYZ CSV:   {out_csv}")

        if out_dxf_pts:
            progress_callback("Writing DXF POINT entities...", 86)
            write_points_dxf(out_dxf_pts, grid_x, grid_y, grid_z,
                             layer="DTM_POINTS")
            saved.append(f"DXF pts:   {out_dxf_pts}")

        if out_landxml:
            progress_callback(
                f"Building Delaunay TIN for LandXML ({n_grid:,} pts)...", 89
            )
            write_landxml(out_landxml, grid_x, grid_y, grid_z, grid_spacing)
            saved.append(f"LandXML:   {out_landxml}")

        if out_dxf_mesh:
            progress_callback(
                f"Building Delaunay TIN for DXF mesh ({n_grid:,} pts)...", 94
            )
            write_dxf_tin_mesh(out_dxf_mesh, grid_x, grid_y, grid_z,
                               grid_spacing)
            saved.append(f"DXF mesh:  {out_dxf_mesh}")

        finish_callback(
            True,
            "Hybrid DTM generated successfully!\n\n"
            f"Boundary grid points  : {n_grid:,}\n"
            f"CSV survey coverage   : {n_csv:,} pts\n"
            f"LAZ min-Z coverage    : {n_laz_pts:,} pts\n"
            f"Z=0 (no LAZ data)     : {n_zero:,} pts\n\n"
            "Saved:\n" + "\n".join(saved),
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        finish_callback(False, f"An error occurred:\n{str(e)}")


# GUI
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    0: "Never Classified", 1: "Unclassified", 2: "Ground",
    3: "Low Vegetation",   4: "Medium Vegetation", 5: "High Vegetation",
    6: "Building",         7: "Low Point",  9: "Water", 12: "Overlap",
}


def find_laz_files(folder):
    return sorted(
        glob.glob(os.path.join(folder, "*.laz")) +
        glob.glob(os.path.join(folder, "*.las"))
    )


class LazGridGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("LAZ Grid Generator v0.9")
        self.root.geometry("820x920")
        self.root.resizable(False, True)

        # LAZ state
        self.laz_mode    = tk.StringVar(value="single")
        self.laz_display = tk.StringVar()
        self._laz_files  = []

        # Grid method state
        self.grid_mode = tk.StringVar(value="dxf")

        # Angle-mode params
        self.grid_size = tk.DoubleVar(value=1.0)
        self.angle     = tk.DoubleVar(value=0.0)

        # DXF-mode state
        self.dxf_display      = tk.StringVar()
        self._dxf_polylines   = []       # list of dicts from get_all_polylines_from_dxf
        self._selected_poly   = tk.IntVar(value=-1)   # index into _dxf_polylines

        # DXF-mode params
        self.chainage_spacing = tk.DoubleVar(value=5.0)
        self.cross_width      = tk.DoubleVar(value=10.0)
        self.cross_spacing    = tk.DoubleVar(value=1.0)

        # DXF UCS transform (local → world)
        self.dxf_offset_x  = tk.DoubleVar(value=0.0)
        self.dxf_offset_y  = tk.DoubleVar(value=0.0)
        self.dxf_rotation  = tk.DoubleVar(value=0.0)

        # Elevation-update mode
        self.elev_dxf_path = tk.StringVar()
        self.elev_out_xyz  = tk.BooleanVar(value=True)
        self.elev_out_dxf  = tk.BooleanVar(value=True)

        # Grid output options
        self.grid_out_dxf = tk.BooleanVar(value=False)

        # Hybrid DTM mode state
        self.hybrid_bdy_display    = tk.StringVar()
        self._hybrid_bdy_polylines = []
        self._selected_bdy_poly    = tk.IntVar(value=-1)
        self.hybrid_csv_path       = tk.StringVar()
        self.hybrid_csv_col_x      = tk.IntVar(value=1)
        self.hybrid_csv_col_y      = tk.IntVar(value=2)
        self.hybrid_csv_col_z      = tk.IntVar(value=3)
        self.hybrid_grid_spacing   = tk.DoubleVar(value=0.5)
        self.hybrid_csv_radius     = tk.DoubleVar(value=5.0)
        self.hybrid_laz_radius     = tk.DoubleVar(value=0.5)
        self.hybrid_laz_pct        = tk.DoubleVar(value=10.0)
        self.hybrid_out_csv        = tk.BooleanVar(value=True)
        self.hybrid_out_dxf_pts    = tk.BooleanVar(value=False)
        self.hybrid_out_landxml    = tk.BooleanVar(value=True)
        self.hybrid_out_dxf_mesh   = tk.BooleanVar(value=False)

        # Shared
        self.max_dist         = tk.DoubleVar(value=2.0)
        self.status           = tk.StringVar(value="Ready")
        self.selected_classes = {}

        self._warn_no_ezdxf()
        self._build_ui()

    # ── Startup ───────────────────────────────────────────────────────────────

    def _warn_no_ezdxf(self):
        if not HAS_EZDXF:
            messagebox.showwarning(
                "Missing dependency",
                "ezdxf is not installed.\n\n"
                "DXF mode requires:  pip install ezdxf\n\n"
                "Angle-based mode works without it.",
            )

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # Fixed bottom bars
        bar = tk.Frame(self.root, bd=1, relief=tk.SUNKEN)
        bar.pack(side=tk.BOTTOM, fill=tk.X)
        tk.Label(bar, textvariable=self.status, anchor="w").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        self.progress = ttk.Progressbar(
            self.root, orient=tk.HORIZONTAL, length=100, mode="determinate"
        )
        self.progress.pack(side=tk.BOTTOM, fill=tk.X)

        main = tk.Frame(self.root, padx=10, pady=8)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # ── 1. LAZ input ─────────────────────────────────────────────────────
        tk.Label(main, text="1. LAZ/LAS Input:", font=("Arial", 10, "bold")).pack(
            anchor="w", pady=(0, 3)
        )
        mr = tk.Frame(main)
        mr.pack(anchor="w", pady=(0, 3))
        tk.Radiobutton(mr, text="Single File", variable=self.laz_mode, value="single",
                        command=self._on_laz_mode_change).pack(side=tk.LEFT, padx=(0, 15))
        tk.Radiobutton(mr, text="Folder  (all .laz / .las tiles)",
                        variable=self.laz_mode, value="folder",
                        command=self._on_laz_mode_change).pack(side=tk.LEFT)

        laz_row = tk.Frame(main)
        laz_row.pack(fill=tk.X, pady=(0, 2))
        tk.Entry(laz_row, textvariable=self.laz_display, state="readonly").pack(
            side=tk.LEFT, fill=tk.X, expand=True
        )
        tk.Button(laz_row, text="Browse", command=self.load_laz).pack(side=tk.RIGHT, padx=5)

        self.laz_info_var = tk.StringVar()
        tk.Label(main, textvariable=self.laz_info_var, fg="grey").pack(
            anchor="w", pady=(0, 6)
        )

        # ── 2. Grid Method ───────────────────────────────────────────────────
        tk.Label(main, text="2. Grid Method:", font=("Arial", 10, "bold")).pack(
            anchor="w", pady=(0, 3)
        )
        gmr = tk.Frame(main)
        gmr.pack(anchor="w", pady=(0, 4))
        gmr_row1 = tk.Frame(gmr)
        gmr_row1.pack(anchor="w")
        tk.Radiobutton(gmr_row1, text="Angle-based  (no DXF required)",
                        variable=self.grid_mode, value="angle",
                        command=self._on_grid_mode_change).pack(side=tk.LEFT, padx=(0, 20))
        tk.Radiobutton(gmr_row1, text="DXF Centreline  (chainage)",
                        variable=self.grid_mode, value="dxf",
                        command=self._on_grid_mode_change).pack(side=tk.LEFT)
        gmr_row2 = tk.Frame(gmr)
        gmr_row2.pack(anchor="w", pady=(2, 0))
        tk.Radiobutton(gmr_row2, text="Update DXF Point Elevations",
                        variable=self.grid_mode, value="elev",
                        command=self._on_grid_mode_change).pack(side=tk.LEFT, padx=(0, 20))
        tk.Radiobutton(gmr_row2, text="Hybrid DTM  (boundary + survey CSV + LAZ)",
                        variable=self.grid_mode, value="hybrid",
                        command=self._on_grid_mode_change).pack(side=tk.LEFT)

        # Angle frame
        self.angle_frame = tk.LabelFrame(main, text="Angle-based Parameters", padx=8, pady=5)
        self._build_angle_params(self.angle_frame)

        # DXF frame
        self.dxf_frame = tk.LabelFrame(main, text="DXF Centreline Parameters", padx=8, pady=5)
        self._build_dxf_params(self.dxf_frame)

        # Elevation update frame
        self.elev_frame = tk.LabelFrame(main, text="Update DXF Point Elevations", padx=8, pady=5)
        self._build_elev_params(self.elev_frame)

        # Hybrid DTM frame
        self.hybrid_frame = tk.LabelFrame(main, text="Hybrid DTM Parameters", padx=8, pady=5)
        self._build_hybrid_params(self.hybrid_frame)

        # Shared max-dist row — keep a reference so pack ordering works
        self._shared_frame = tk.Frame(main)
        self._shared_frame.pack(fill=tk.X, pady=(4, 2))
        tk.Label(self._shared_frame, text="Max Distance (m):").pack(side=tk.LEFT)
        tk.Entry(self._shared_frame, textvariable=self.max_dist, width=10).pack(
            side=tk.LEFT, padx=8
        )
        tk.Label(self._shared_frame,
                  text="Grid points further than this from any LAZ point → Z=0",
                  fg="grey").pack(side=tk.LEFT)

        # DXF export checkbox (grid modes only)
        self._dxf_export_frame = tk.Frame(main)
        self._dxf_export_frame.pack(fill=tk.X, pady=(0, 6))
        tk.Checkbutton(
            self._dxf_export_frame,
            text="Also export DXF  (grid points as POINT entities)",
            variable=self.grid_out_dxf,
        ).pack(side=tk.LEFT)

        self._on_grid_mode_change()   # show correct param frame initially

        # ── 3. Classifications ───────────────────────────────────────────────
        tk.Label(main, text="3. Point Classifications:", font=("Arial", 10, "bold")).pack(
            anchor="w", pady=(0, 3)
        )
        tk.Label(main, text="Load LAZ input first to populate.", fg="grey").pack(
            anchor="w", pady=(0, 2)
        )
        cls_lf = tk.LabelFrame(main, text="Available Classes")
        cls_lf.pack(fill=tk.BOTH, expand=True, pady=(0, 6))

        self.cls_canvas = tk.Canvas(cls_lf, height=75)
        sb_cls = tk.Scrollbar(cls_lf, orient="vertical", command=self.cls_canvas.yview)
        self.scrollable_frame = tk.Frame(self.cls_canvas)
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.cls_canvas.configure(scrollregion=self.cls_canvas.bbox("all")),
        )
        self.cls_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.cls_canvas.configure(yscrollcommand=sb_cls.set)
        self.cls_canvas.pack(side="left", fill="both", expand=True)
        sb_cls.pack(side="right", fill="y")

        # ── Generate ─────────────────────────────────────────────────────────
        self.generate_btn = tk.Button(
            main, text="GENERATE GRID CSV",
            command=self.start_generation,
            bg="#4CAF50", fg="white",
            font=("Arial", 11, "bold"), height=2,
            state=tk.DISABLED,
        )
        self.generate_btn.pack(fill=tk.X)

    # ── Parameter sub-frames ──────────────────────────────────────────────────

    def _build_angle_params(self, parent):
        for i, (lbl, var, hint) in enumerate([
            ("Grid Size (m):",        self.grid_size, "Size of each grid cell"),
            ("Rotation Angle (deg):", self.angle,     "Positive = CCW, Negative = CW"),
        ]):
            tk.Label(parent, text=lbl).grid(row=i, column=0, sticky="w", pady=2)
            tk.Entry(parent, textvariable=var, width=10).grid(row=i, column=1, padx=8, sticky="w")
            tk.Label(parent, text=hint, fg="grey").grid(row=i, column=2, padx=5, sticky="w")

    def _build_dxf_params(self, parent):
        # DXF file row
        tk.Label(parent, text="DXF File:").grid(row=0, column=0, sticky="w", pady=2)
        dxf_ef = tk.Frame(parent)
        dxf_ef.grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)
        tk.Entry(dxf_ef, textvariable=self.dxf_display, state="readonly", width=36).pack(
            side=tk.LEFT
        )
        tk.Button(dxf_ef, text="Browse", command=self.load_dxf).pack(side=tk.LEFT, padx=4)

        # Polyline picker label
        tk.Label(parent, text="Select Polyline:").grid(row=1, column=0, sticky="nw", pady=(4, 2))

        # Scrollable radio list for polylines
        poly_lf = tk.Frame(parent, relief=tk.SUNKEN, bd=1)
        poly_lf.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(4, 4))

        self.poly_canvas = tk.Canvas(poly_lf, height=90, bg="white")
        poly_sb = tk.Scrollbar(poly_lf, orient="vertical", command=self.poly_canvas.yview)
        self.poly_inner = tk.Frame(self.poly_canvas, bg="white")
        self.poly_inner.bind(
            "<Configure>",
            lambda e: self.poly_canvas.configure(scrollregion=self.poly_canvas.bbox("all")),
        )
        self.poly_canvas.create_window((0, 0), window=self.poly_inner, anchor="nw")
        self.poly_canvas.configure(yscrollcommand=poly_sb.set)
        self.poly_canvas.pack(side="left", fill="both", expand=True)
        poly_sb.pack(side="right", fill="y")

        self.poly_hint = tk.Label(
            self.poly_inner,
            text="  Load a DXF file to see available polylines.",
            fg="grey", bg="white",
        )
        self.poly_hint.pack(anchor="w", padx=4, pady=4)

        # Chainage / cross-section params
        for i, (lbl, var, hint) in enumerate([
            ("Chainage Spacing (m):",      self.chainage_spacing, "Along-centreline interval between cross-sections"),
            ("Cross-Section Width (m):",   self.cross_width,      "Half-width — extends ±this from centreline"),
            ("Cross-Section Spacing (m):", self.cross_spacing,    "Point spacing within each cross-section"),
        ], start=2):
            tk.Label(parent, text=lbl).grid(row=i, column=0, sticky="w", pady=2)
            tk.Entry(parent, textvariable=var, width=10).grid(row=i, column=1, padx=8, sticky="w")
            tk.Label(parent, text=hint, fg="grey").grid(row=i, column=2, padx=5, sticky="w")

        # UCS Transform separator
        sep = ttk.Separator(parent, orient="horizontal")
        sep.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 4))

        tk.Label(
            parent,
            text="DXF Local UCS → World Transform  (leave at 0 if DXF is already in world coords)",
            fg="#555555", font=("Arial", 8, "italic"),
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(0, 4))

        for i, (lbl, var, hint) in enumerate([
            ("Offset X (m):",      self.dxf_offset_x, "Add to every DXF X coordinate"),
            ("Offset Y (m):",      self.dxf_offset_y, "Add to every DXF Y coordinate"),
            ("UCS Rotation (°):",  self.dxf_rotation, "CCW rotation of local UCS relative to world (applied before offset)"),
        ], start=7):
            tk.Label(parent, text=lbl).grid(row=i, column=0, sticky="w", pady=2)
            tk.Entry(parent, textvariable=var, width=14).grid(row=i, column=1, padx=8, sticky="w")
            tk.Label(parent, text=hint, fg="grey").grid(row=i, column=2, padx=5, sticky="w")

    def _build_elev_params(self, parent):
        tk.Label(parent, text="DXF Points File:").grid(row=0, column=0, sticky="w", pady=2)
        ef = tk.Frame(parent)
        ef.grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)
        tk.Entry(ef, textvariable=self.elev_dxf_path, state="readonly", width=36).pack(side=tk.LEFT)
        tk.Button(ef, text="Browse", command=self._load_elev_dxf).pack(side=tk.LEFT, padx=4)

        tk.Label(
            parent,
            text="DXF file containing POINT entities at arbitrary Z — Z will be replaced from LAZ.",
            fg="grey", font=("Arial", 8, "italic"),
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 6))

        tk.Label(parent, text="Output:").grid(row=2, column=0, sticky="w", pady=2)
        out_frame = tk.Frame(parent)
        out_frame.grid(row=2, column=1, columnspan=2, sticky="w")
        tk.Checkbutton(out_frame, text="XYZ CSV", variable=self.elev_out_xyz).pack(
            side=tk.LEFT, padx=(0, 15)
        )
        tk.Checkbutton(out_frame, text="Updated DXF  (same entities, corrected Z)",
                        variable=self.elev_out_dxf).pack(side=tk.LEFT)

    def _load_elev_dxf(self):
        path = filedialog.askopenfilename(
            title="Select DXF file with POINT entities",
            filetypes=[("DXF Files", "*.dxf"), ("All Files", "*.*")],
        )
        if path:
            self.elev_dxf_path.set(path)
            self._check_enable()

    def _build_hybrid_params(self, parent):
        parent.columnconfigure(2, weight=1)

        # Boundary DXF
        tk.Label(parent, text="Boundary DXF:").grid(
            row=0, column=0, sticky="w", pady=2)
        bef = tk.Frame(parent)
        bef.grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)
        tk.Entry(bef, textvariable=self.hybrid_bdy_display,
                 state="readonly", width=36).pack(side=tk.LEFT)
        tk.Button(bef, text="Browse",
                  command=self._load_hybrid_boundary).pack(side=tk.LEFT, padx=4)

        # Boundary polyline picker
        tk.Label(parent, text="Boundary\nPolyline:").grid(
            row=1, column=0, sticky="nw", pady=(4, 2))
        bdy_lf = tk.Frame(parent, relief=tk.SUNKEN, bd=1)
        bdy_lf.grid(row=1, column=1, columnspan=2, sticky="ew", pady=(4, 4))
        self.bdy_canvas = tk.Canvas(bdy_lf, height=70, bg="white")
        bdy_sb = tk.Scrollbar(bdy_lf, orient="vertical",
                              command=self.bdy_canvas.yview)
        self.bdy_inner = tk.Frame(self.bdy_canvas, bg="white")
        self.bdy_inner.bind(
            "<Configure>",
            lambda e: self.bdy_canvas.configure(
                scrollregion=self.bdy_canvas.bbox("all"))
        )
        self.bdy_canvas.create_window((0, 0), window=self.bdy_inner, anchor="nw")
        self.bdy_canvas.configure(yscrollcommand=bdy_sb.set)
        self.bdy_canvas.pack(side="left", fill="both", expand=True)
        bdy_sb.pack(side="right", fill="y")

        # Survey CSV
        tk.Label(parent, text="Survey CSV:").grid(
            row=2, column=0, sticky="w", pady=(6, 2))
        cef = tk.Frame(parent)
        cef.grid(row=2, column=1, columnspan=2, sticky="ew", pady=(6, 2))
        tk.Entry(cef, textvariable=self.hybrid_csv_path,
                 state="readonly", width=36).pack(side=tk.LEFT)
        tk.Button(cef, text="Browse",
                  command=self._load_hybrid_csv).pack(side=tk.LEFT, padx=4)
        # CSV column mapping
        col_f = tk.Frame(parent)
        col_f.grid(row=3, column=1, columnspan=2, sticky="w", pady=(2, 0))
        tk.Label(col_f, text="Col mapping (0-based):").pack(side=tk.LEFT)
        tk.Label(col_f, text="  X").pack(side=tk.LEFT)
        tk.Spinbox(col_f, textvariable=self.hybrid_csv_col_x,
                   from_=0, to=20, width=3).pack(side=tk.LEFT, padx=(1, 6))
        tk.Label(col_f, text="Y").pack(side=tk.LEFT)
        tk.Spinbox(col_f, textvariable=self.hybrid_csv_col_y,
                   from_=0, to=20, width=3).pack(side=tk.LEFT, padx=(1, 6))
        tk.Label(col_f, text="Z").pack(side=tk.LEFT)
        tk.Spinbox(col_f, textvariable=self.hybrid_csv_col_z,
                   from_=0, to=20, width=3).pack(side=tk.LEFT, padx=(1, 6))
        tk.Button(col_f, text="Auto-detect",
                  command=self._autodetect_csv_cols).pack(side=tk.LEFT, padx=(4, 0))

        # Numeric params
        for i, (lbl, var, hint) in enumerate([
            ("Grid Spacing (m):",    self.hybrid_grid_spacing,
             "0.3–0.5 m recommended"),
            ("CSV Radius (m):",      self.hybrid_csv_radius,
             "Use survey elevation within this distance  (check Source col in CSV: 1=CSV, 0=LAZ)"),
            ("LAZ Max Gap (m):",    self.hybrid_laz_radius,
             "Grid pts with no LAZ point within this distance → Z=0"),
        ], start=4):
            tk.Label(parent, text=lbl).grid(
                row=i, column=0, sticky="w", pady=2)
            tk.Entry(parent, textvariable=var, width=8).grid(
                row=i, column=1, padx=8, sticky="w")
            tk.Label(parent, text=hint, fg="grey").grid(
                row=i, column=2, padx=5, sticky="w")

        # Output checkboxes
        tk.Label(parent, text="Output:").grid(
            row=7, column=0, sticky="w", pady=(6, 2))
        out_f = tk.Frame(parent)
        out_f.grid(row=7, column=1, columnspan=2, sticky="w", pady=(6, 2))
        tk.Checkbutton(out_f, text="XYZ CSV",
                       variable=self.hybrid_out_csv).pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(out_f, text="DXF Points",
                       variable=self.hybrid_out_dxf_pts).pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(out_f, text="LandXML TIN",
                       variable=self.hybrid_out_landxml).pack(side=tk.LEFT, padx=(0, 8))
        tk.Checkbutton(out_f, text="DXF Mesh",
                       variable=self.hybrid_out_dxf_mesh).pack(side=tk.LEFT)

    def _load_hybrid_boundary(self):
        path = filedialog.askopenfilename(
            title="Select Boundary DXF File",
            filetypes=[("DXF Files", "*.dxf"), ("All Files", "*.*")],
        )
        if not path:
            return
        self.hybrid_bdy_display.set(path)
        self._populate_bdy_polylines(path)

    def _load_hybrid_csv(self):
        path = filedialog.askopenfilename(
            title="Select Ground Survey CSV",
            filetypes=[("CSV Files", "*.csv"), ("All Files", "*.*")],
        )
        if not path:
            return
        self.hybrid_csv_path.set(path)
        self._autodetect_csv_cols()
        self._check_enable()

    def _autodetect_csv_cols(self):
        path = self.hybrid_csv_path.get()
        if not path or not os.path.isfile(path):
            return
        try:
            cx, cy, cz = detect_csv_columns(path)
            self.hybrid_csv_col_x.set(cx)
            self.hybrid_csv_col_y.set(cy)
            self.hybrid_csv_col_z.set(cz)
        except Exception:
            pass

    def _populate_bdy_polylines(self, dxf_path):
        for w in self.bdy_inner.winfo_children():
            w.destroy()
        self._hybrid_bdy_polylines = []
        self._selected_bdy_poly.set(-1)
        tk.Label(self.bdy_inner, text="  Scanning DXF...",
                 fg="grey", bg="white").pack(anchor="w", padx=4, pady=4)
        self._update_progress("Scanning DXF for boundary polyline...", 0)

        def scan():
            try:
                polys = get_all_polylines_from_dxf(dxf_path)
                self.root.after(0, lambda: self._show_bdy_polylines(polys))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("DXF Error", str(e)))

        threading.Thread(target=scan, daemon=True).start()

    def _show_bdy_polylines(self, polys):
        for w in self.bdy_inner.winfo_children():
            w.destroy()
        if not polys:
            tk.Label(self.bdy_inner, text="  No polylines found.",
                     fg="red", bg="white").pack(anchor="w", padx=4, pady=4)
            return
        self._hybrid_bdy_polylines = polys
        longest = max(range(len(polys)), key=lambda i: polys[i]["arc_length"])
        self._selected_bdy_poly.set(longest)
        for i, pl in enumerate(polys):
            lbl = (f"  Layer: {pl['layer']:<20s}  "
                   f"Length: {pl['arc_length']:>10.2f} m  "
                   f"Vertices: {pl['n_vertices']:>4d}  [{pl['etype']}]")
            tk.Radiobutton(
                self.bdy_inner, text=lbl, variable=self._selected_bdy_poly,
                value=i, anchor="w", justify=tk.LEFT, bg="white",
                command=self._check_enable, font=("Courier", 9),
            ).pack(anchor="w", fill=tk.X, padx=2)
        self._update_progress(
            f"{len(polys)} polyline(s) found. Select the boundary.", 0
        )
        self._check_enable()

    # ── Mode switching ────────────────────────────────────────────────────────

    def _on_laz_mode_change(self):
        self.laz_display.set("")
        self.laz_info_var.set("")
        self._laz_files = []
        self._clear_classes()
        self._check_enable()

    def _on_grid_mode_change(self):
        mode = self.grid_mode.get()
        for f in (self.angle_frame, self.dxf_frame,
                  self.elev_frame, self.hybrid_frame):
            f.pack_forget()
        if mode == "hybrid":
            self.hybrid_frame.pack(fill=tk.X, before=self._shared_frame)
            self._shared_frame.pack_forget()
            self._dxf_export_frame.pack_forget()
        else:
            if mode == "angle":
                self.angle_frame.pack(fill=tk.X, before=self._shared_frame)
            elif mode == "dxf":
                self.dxf_frame.pack(fill=tk.X, before=self._shared_frame)
            elif mode == "elev":
                self.elev_frame.pack(fill=tk.X, before=self._shared_frame)
            self._shared_frame.pack(fill=tk.X, pady=(4, 2),
                                    before=self._dxf_export_frame)
            self._dxf_export_frame.pack(fill=tk.X, pady=(0, 6))
        if hasattr(self, "generate_btn"):
            btn_labels = {
                "angle":  "GENERATE GRID CSV",
                "dxf":    "GENERATE GRID CSV",
                "elev":   "UPDATE DXF ELEVATIONS",
                "hybrid": "GENERATE HYBRID DTM",
            }
            self.generate_btn.config(text=btn_labels.get(mode, "GENERATE"))
        self._check_enable()

    # ── File loading ──────────────────────────────────────────────────────────

    def load_laz(self):
        if self.laz_mode.get() == "single":
            path = filedialog.askopenfilename(
                title="Select LAZ/LAS File",
                filetypes=[("LAZ/LAS Files", "*.laz *.las")],
            )
            if not path:
                return
            self._laz_files = [path]
            self.laz_display.set(path)
            self.laz_info_var.set("1 file selected")
        else:
            folder = filedialog.askdirectory(title="Select Folder Containing LAZ/LAS Tiles")
            if not folder:
                return
            files = find_laz_files(folder)
            if not files:
                messagebox.showwarning("No Files", f"No .laz/.las files in:\n{folder}")
                return
            self._laz_files = files
            self.laz_display.set(folder)
            self.laz_info_var.set(
                f"{len(files)} file(s):  "
                + ",  ".join(os.path.basename(f) for f in files[:5])
                + ("  …" if len(files) > 5 else "")
            )
        self._populate_classes(self._laz_files)

    def load_dxf(self):
        path = filedialog.askopenfilename(
            title="Select DXF Centreline File",
            filetypes=[("DXF Files", "*.dxf"), ("All Files", "*.*")],
        )
        if not path:
            return
        self.dxf_display.set(path)
        self._populate_polylines(path)

    # ── Polyline scanning ─────────────────────────────────────────────────────

    def _populate_polylines(self, dxf_path):
        # Clear current list
        for w in self.poly_inner.winfo_children():
            w.destroy()
        self._dxf_polylines = []
        self._selected_poly.set(-1)
        self.generate_btn["state"] = tk.DISABLED

        self.poly_hint = tk.Label(
            self.poly_inner, text="  Scanning DXF...", fg="grey", bg="white"
        )
        self.poly_hint.pack(anchor="w", padx=4, pady=4)
        self._update_progress("Scanning DXF for polylines...", 0)

        def scan():
            try:
                polys = get_all_polylines_from_dxf(dxf_path)
                self.root.after(0, lambda: self._show_polylines(polys))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("DXF Error", str(e)))
                self.root.after(0, lambda: self._update_progress("Error reading DXF.", 0))

        threading.Thread(target=scan, daemon=True).start()

    def _show_polylines(self, polys):
        for w in self.poly_inner.winfo_children():
            w.destroy()

        if not polys:
            tk.Label(
                self.poly_inner,
                text="  No LWPOLYLINE/POLYLINE entities found in this DXF.",
                fg="red", bg="white",
            ).pack(anchor="w", padx=4, pady=4)
            self._update_progress("No polylines found in DXF.", 0)
            return

        self._dxf_polylines = polys

        # Auto-select longest
        longest_idx = max(range(len(polys)), key=lambda i: polys[i]["arc_length"])
        self._selected_poly.set(longest_idx)

        for i, pl in enumerate(polys):
            lbl = (
                f"  Layer: {pl['layer']:<20s}  "
                f"Length: {pl['arc_length']:>10.2f} m  "
                f"Vertices: {pl['n_vertices']:>4d}  "
                f"[{pl['etype']}]"
            )
            rb = tk.Radiobutton(
                self.poly_inner,
                text=lbl,
                variable=self._selected_poly,
                value=i,
                anchor="w",
                justify=tk.LEFT,
                bg="white",
                command=self._check_enable,
                font=("Courier", 9),
            )
            rb.pack(anchor="w", fill=tk.X, padx=2)

        n = len(polys)
        self._update_progress(
            f"DXF loaded: {n} polyline(s) found. Select the road/track centreline.", 0
        )
        self._check_enable()

    # ── Classification scanning ───────────────────────────────────────────────

    def _clear_classes(self):
        for w in self.scrollable_frame.winfo_children():
            w.destroy()
        self.selected_classes.clear()

    def _populate_classes(self, file_list):
        self._clear_classes()
        self.generate_btn["state"] = tk.DISABLED
        self._update_progress(f"Scanning {len(file_list)} file(s) for classes...", 0)

        def scan():
            found = set()
            try:
                for fi, path in enumerate(file_list):
                    fname = os.path.basename(path)
                    with laspy.open(path) as f:
                        total = f.header.point_count
                        done = 0
                        try:
                            for chunk in f.chunk_iterator(1_000_000):
                                found.update(np.unique(chunk.classification).tolist())
                                done += len(chunk)
                                pct = int(done / total * 100) if total else 0
                                self.root.after(
                                    0,
                                    lambda m=f"Scanning {fi+1}/{len(file_list)} — {fname}: {pct}%",
                                    p=pct: self._update_progress(m, p),
                                )
                        except Exception as e:
                            is_lazrs = (LazrsError and isinstance(e, LazrsError)) or \
                                       "failed to fill whole buffer" in str(e)
                            if not is_lazrs:
                                raise e
                self.root.after(0, lambda: self._show_classes(sorted(found)))
            except Exception as e:
                self.root.after(0, lambda: messagebox.showerror("Error", f"Failed:\n{e}"))
                self.root.after(0, lambda: self._update_progress("Error loading LAZ.", 0))

        threading.Thread(target=scan, daemon=True).start()

    def _show_classes(self, classes):
        self._update_progress("Input loaded. Select classifications then Generate.", 0)
        if not classes:
            tk.Label(self.scrollable_frame, text="No classifications found.").pack(anchor="w")
            return
        for cls in classes:
            cls = int(cls)
            var = tk.BooleanVar(value=(cls == 2))
            name = CLASS_NAMES.get(cls, "Reserved/User Defined")
            tk.Checkbutton(
                self.scrollable_frame, text=f"  {cls} – {name}", variable=var
            ).pack(anchor="w")
            self.selected_classes[cls] = var
        self._check_enable()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _check_enable(self):
        if not hasattr(self, "generate_btn"):
            return
        laz_ok = bool(self._laz_files)
        mode = self.grid_mode.get()
        if mode == "angle":
            ok = laz_ok
        elif mode == "dxf":
            ok = laz_ok and (self._selected_poly.get() >= 0)
        elif mode == "elev":
            ok = laz_ok and bool(self.elev_dxf_path.get())
        else:  # hybrid
            ok = (laz_ok and
                  self._selected_bdy_poly.get() >= 0 and
                  bool(self.hybrid_csv_path.get()))
        self.generate_btn["state"] = tk.NORMAL if ok else tk.DISABLED

    def _update_progress(self, msg, percent=None):
        def _go():
            self.status.set(msg)
            if percent is not None:
                self.progress["value"] = percent
        self.root.after(0, _go)

    def _on_finish(self, success, msg):
        self.root.after(0, lambda: self.status.set("Ready"))
        if success:
            messagebox.showinfo("Success", msg)
            self.laz_display.set("")
            self.laz_info_var.set("")
            self._laz_files = []
            self.dxf_display.set("")
            self.elev_dxf_path.set("")
            for w in self.poly_inner.winfo_children():
                w.destroy()
            self._dxf_polylines = []
            self._selected_poly.set(-1)
            self._clear_classes()
            self.generate_btn["state"] = tk.DISABLED
            self.status.set("Done. Ready for new input.")
        else:
            messagebox.showerror("Error", msg)
            self.generate_btn["state"] = tk.NORMAL

    # ── Generation ────────────────────────────────────────────────────────────

    def start_generation(self):
        if not self._laz_files:
            messagebox.showwarning("Warning", "Please select a LAZ/LAS file or folder.")
            return

        mode = self.grid_mode.get()
        centerline_pts = None

        if mode == "dxf":
            idx = self._selected_poly.get()
            if idx < 0 or idx >= len(self._dxf_polylines):
                messagebox.showwarning("Warning", "Please select a polyline from the DXF list.")
                return
            centerline_pts = self._dxf_polylines[idx]["vertices"].copy()
            # Apply UCS → world transform
            try:
                ox = self.dxf_offset_x.get()
                oy = self.dxf_offset_y.get()
                rot = self.dxf_rotation.get()
            except tk.TclError:
                messagebox.showerror("Error", "Invalid value in DXF UCS transform fields.")
                return
            centerline_pts = apply_ucs_transform(centerline_pts, ox, oy, rot)

        selected = [cls for cls, var in self.selected_classes.items() if var.get()]
        if not selected:
            if not messagebox.askyesno("Warning", "No classifications selected.\nUse ALL points?"):
                return
            selected = None

        try:
            mx_d = self.max_dist.get()
        except tk.TclError:
            messagebox.showerror("Error", "Invalid Max Distance value.")
            return

        if mx_d <= 0:
            messagebox.showerror("Error", "Max Distance must be positive.")
            return

        self.generate_btn["state"] = tk.DISABLED
        finish_cb = lambda s, m: self.root.after(0, lambda: self._on_finish(s, m))

        # ── Elevation update mode ─────────────────────────────────────────────
        if mode == "elev":
            dxf_pts_path = self.elev_dxf_path.get()
            want_xyz = self.elev_out_xyz.get()
            want_dxf = self.elev_out_dxf.get()

            if not want_xyz and not want_dxf:
                messagebox.showwarning("Warning", "Select at least one output format (XYZ CSV or DXF).")
                self.generate_btn["state"] = tk.NORMAL
                return

            out_xyz = out_dxf = None
            base = os.path.splitext(dxf_pts_path)[0]

            if want_xyz:
                out_xyz = filedialog.asksaveasfilename(
                    title="Save XYZ CSV as…",
                    defaultextension=".csv",
                    filetypes=[("CSV Files", "*.csv")],
                    initialfile=os.path.basename(base) + "_elev_updated.csv",
                )
                if not out_xyz:
                    self.generate_btn["state"] = tk.NORMAL
                    return

            if want_dxf:
                out_dxf = filedialog.asksaveasfilename(
                    title="Save updated DXF as…",
                    defaultextension=".dxf",
                    filetypes=[("DXF Files", "*.dxf")],
                    initialfile=os.path.basename(base) + "_elev_updated.dxf",
                )
                if not out_dxf:
                    self.generate_btn["state"] = tk.NORMAL
                    return

            threading.Thread(
                target=process_update_elevations,
                args=(
                    list(self._laz_files),
                    dxf_pts_path,
                    mx_d, selected,
                    out_xyz, out_dxf,
                    self._update_progress, finish_cb,
                ),
                daemon=True,
            ).start()
            return

        # ── Hybrid DTM mode ───────────────────────────────────────────────────
        if mode == "hybrid":
            idx = self._selected_bdy_poly.get()
            if idx < 0 or idx >= len(self._hybrid_bdy_polylines):
                messagebox.showwarning("Warning", "Please select the boundary polyline.")
                self.generate_btn["state"] = tk.NORMAL
                return

            boundary_poly = self._hybrid_bdy_polylines[idx]["vertices"].copy()
            # Ensure polygon is closed
            if not np.allclose(boundary_poly[0], boundary_poly[-1]):
                boundary_poly = np.vstack([boundary_poly, boundary_poly[0]])

            try:
                csv_pts = read_survey_csv(
                    self.hybrid_csv_path.get(),
                    col_x=self.hybrid_csv_col_x.get(),
                    col_y=self.hybrid_csv_col_y.get(),
                    col_z=self.hybrid_csv_col_z.get(),
                )
            except Exception as e:
                messagebox.showerror("CSV Error", str(e))
                self.generate_btn["state"] = tk.NORMAL
                return

            want_csv     = self.hybrid_out_csv.get()
            want_dxf_pts = self.hybrid_out_dxf_pts.get()
            want_landxml = self.hybrid_out_landxml.get()
            want_dxf_msh = self.hybrid_out_dxf_mesh.get()

            if not any([want_csv, want_dxf_pts, want_landxml, want_dxf_msh]):
                messagebox.showwarning("Warning", "Select at least one output format.")
                self.generate_btn["state"] = tk.NORMAL
                return

            try:
                gs  = self.hybrid_grid_spacing.get()
                cr  = self.hybrid_csv_radius.get()
                lr  = self.hybrid_laz_radius.get()
            except tk.TclError:
                messagebox.showerror("Error", "Invalid value in Hybrid DTM parameters.")
                self.generate_btn["state"] = tk.NORMAL
                return

            base_name = os.path.splitext(self.hybrid_csv_path.get())[0]
            h_csv = h_dxf_pts = h_landxml = h_dxf_msh = None

            if want_csv:
                h_csv = filedialog.asksaveasfilename(
                    title="Save DTM XYZ CSV as…",
                    defaultextension=".csv",
                    filetypes=[("CSV Files", "*.csv")],
                    initialfile=os.path.basename(base_name) + "_dtm.csv",
                )
                if not h_csv:
                    self.generate_btn["state"] = tk.NORMAL; return

            if want_dxf_pts:
                h_dxf_pts = filedialog.asksaveasfilename(
                    title="Save DTM DXF Points as…",
                    defaultextension=".dxf",
                    filetypes=[("DXF Files", "*.dxf")],
                    initialfile=os.path.basename(base_name) + "_dtm_pts.dxf",
                )
                if not h_dxf_pts:
                    self.generate_btn["state"] = tk.NORMAL; return

            if want_landxml:
                h_landxml = filedialog.asksaveasfilename(
                    title="Save LandXML TIN as…",
                    defaultextension=".xml",
                    filetypes=[("LandXML", "*.xml"), ("All Files", "*.*")],
                    initialfile=os.path.basename(base_name) + "_dtm.xml",
                )
                if not h_landxml:
                    self.generate_btn["state"] = tk.NORMAL; return

            if want_dxf_msh:
                h_dxf_msh = filedialog.asksaveasfilename(
                    title="Save DXF TIN Mesh as…",
                    defaultextension=".dxf",
                    filetypes=[("DXF Files", "*.dxf")],
                    initialfile=os.path.basename(base_name) + "_dtm_mesh.dxf",
                )
                if not h_dxf_msh:
                    self.generate_btn["state"] = tk.NORMAL; return

            threading.Thread(
                target=process_hybrid_dtm,
                args=(
                    list(self._laz_files),
                    boundary_poly,
                    csv_pts,
                    gs, cr, lr,
                    selected,
                    h_csv, h_dxf_pts, h_landxml, h_dxf_msh,
                    self._update_progress, finish_cb,
                ),
                daemon=True,
            ).start()
            return

        # ── Grid generation modes ─────────────────────────────────────────────
        try:
            grid_s = self.grid_size.get()
            ang    = self.angle.get()
            ch_sp  = self.chainage_spacing.get()
            cr_w   = self.cross_width.get()
            cr_sp  = self.cross_spacing.get()
        except tk.TclError:
            messagebox.showerror("Error", "Invalid numeric value in parameters.")
            self.generate_btn["state"] = tk.NORMAL
            return

        if mode == "angle" and grid_s <= 0:
            messagebox.showerror("Error", "Grid Size must be positive.")
            self.generate_btn["state"] = tk.NORMAL
            return
        if mode == "dxf" and (ch_sp <= 0 or cr_w <= 0 or cr_sp <= 0):
            messagebox.showerror("Error", "Chainage/cross-section values must be positive.")
            self.generate_btn["state"] = tk.NORMAL
            return

        output = filedialog.asksaveasfilename(
            title="Save grid CSV as…",
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="grid_output.csv",
        )
        if not output:
            self.generate_btn["state"] = tk.NORMAL
            return

        out_dxf = None
        if self.grid_out_dxf.get():
            base = os.path.splitext(output)[0]
            out_dxf = filedialog.asksaveasfilename(
                title="Save grid points DXF as…",
                defaultextension=".dxf",
                filetypes=[("DXF Files", "*.dxf")],
                initialfile=os.path.basename(base) + "_grid.dxf",
            )
            if not out_dxf:
                self.generate_btn["state"] = tk.NORMAL
                return

        threading.Thread(
            target=process,
            args=(
                list(self._laz_files), mode,
                grid_s, ang,
                centerline_pts,
                ch_sp, cr_w, cr_sp,
                mx_d, selected,
                output, out_dxf,
                self._update_progress, finish_cb,
            ),
            daemon=True,
        ).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = LazGridGenerator(root)
    root.mainloop()
