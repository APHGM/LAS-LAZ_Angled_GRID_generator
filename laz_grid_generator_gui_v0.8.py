import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import laspy
import numpy as np
import threading
import os
import glob

try:
    from lazrs import LazrsError
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

    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()
    results = []

    for entity in msp:
        pts = []
        layer = getattr(entity.dxf, "layer", "0")
        etype = entity.dxftype()

        if etype == "LWPOLYLINE":
            pts = [(p[0], p[1]) for p in entity.get_points()]
        elif etype == "POLYLINE":
            pts = [(v.dxf.location.x, v.dxf.location.y) for v in entity.vertices]

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
        from scipy.spatial import cKDTree
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
    doc = ezdxf.readfile(dxf_path)
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
    doc = ezdxf.readfile(input_dxf_path)
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
        from scipy.spatial import cKDTree
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
    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()
    for x, y, z in zip(x_arr, y_arr, z_arr):
        msp.add_point((float(x), float(y), float(z)), dxfattribs={"layer": layer})
    doc.saveas(output_path)


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
        self.root.title("LAZ Grid Generator v0.8")
        self.root.geometry("780x960")
        self.root.resizable(False, False)

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
        tk.Radiobutton(gmr, text="Angle-based  (no DXF required)",
                        variable=self.grid_mode, value="angle",
                        command=self._on_grid_mode_change).pack(side=tk.LEFT, padx=(0, 10))
        tk.Radiobutton(gmr, text="DXF Centreline  (chainage)",
                        variable=self.grid_mode, value="dxf",
                        command=self._on_grid_mode_change).pack(side=tk.LEFT, padx=(0, 10))
        tk.Radiobutton(gmr, text="Update DXF Point Elevations",
                        variable=self.grid_mode, value="elev",
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

        self.cls_canvas = tk.Canvas(cls_lf, height=90)
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

    # ── Mode switching ────────────────────────────────────────────────────────

    def _on_laz_mode_change(self):
        self.laz_display.set("")
        self.laz_info_var.set("")
        self._laz_files = []
        self._clear_classes()
        self._check_enable()

    def _on_grid_mode_change(self):
        mode = self.grid_mode.get()
        self.angle_frame.pack_forget()
        self.dxf_frame.pack_forget()
        self.elev_frame.pack_forget()
        if mode == "angle":
            self.angle_frame.pack(fill=tk.X, before=self._shared_frame)
        elif mode == "dxf":
            self.dxf_frame.pack(fill=tk.X, before=self._shared_frame)
        else:  # elev
            self.elev_frame.pack(fill=tk.X, before=self._shared_frame)
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
        else:  # elev
            ok = laz_ok and bool(self.elev_dxf_path.get())
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
