import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import laspy
import numpy as np
import threading
import os
import glob
import tempfile
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    pts, diffs, seg_len, cumlen = build_parameterization(centerline)
    total_length = cumlen[-1]
    n_pts = len(pts)
    seg_tan = diffs / seg_len[:, np.newaxis]
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
        cx = pts[idx, 0] + t * diffs[idx, 0]
        cy = pts[idx, 1] + t * diffs[idx, 1]
        tx_raw = (1.0 - t) * vtx_tan[idx, 0] + t * vtx_tan[idx + 1, 0]
        ty_raw = (1.0 - t) * vtx_tan[idx, 1] + t * vtx_tan[idx + 1, 1]
        norm = np.hypot(tx_raw, ty_raw)
        if norm > 1e-12:
            tx, ty = tx_raw / norm, ty_raw / norm
        else:
            tx, ty = seg_tan[idx]
        px, py = -ty, tx
        for off in offsets:
            all_x.append(cx + off * px)
            all_y.append(cy + off * py)
            all_ch.append(ch)
            all_off.append(off)

    return np.array(all_x), np.array(all_y), np.array(all_ch), np.array(all_off)


def generate_angle_grid(x_min, x_max, y_min, y_max, rot_x, rot_y, grid_size, angle_deg):
    """Generate rotated grid from already-rotated bounds."""
    u_start = np.floor(x_min / grid_size) * grid_size
    u_end   = np.ceil(x_max  / grid_size) * grid_size
    v_start = np.floor(y_min / grid_size) * grid_size
    v_end   = np.ceil(y_max  / grid_size) * grid_size

    n_est = ((u_end - u_start) / grid_size) * ((v_end - v_start) / grid_size)
    if n_est > 50_000_000:
        raise ValueError(f"Grid too large (~{int(n_est):,} pts). Increase grid size.")

    theta_rad = np.radians(angle_deg)
    gu, gv = np.meshgrid(
        np.arange(u_start, u_end + grid_size, grid_size),
        np.arange(v_start, v_end + grid_size, grid_size),
    )
    gu, gv = gu.flatten(), gv.flatten()
    cos_inv, sin_inv = np.cos(theta_rad), np.sin(theta_rad)
    return gu * cos_inv - gv * sin_inv, gu * sin_inv + gv * cos_inv


# ---------------------------------------------------------------------------
# LAZ → memmap streaming
# ---------------------------------------------------------------------------

def _read_laz_to_memmap(laz_files, selected_classes, tmp_dir, progress_callback, max_pct=40):
    """
    Stream all filtered LAZ points to a disk-backed memmap file.
    Returns (mm_path, n_pts_actual, x_min, x_max, y_min, y_max,
             rot_x_min, rot_x_max, rot_y_min, rot_y_max, angle_deg stored separately)
    The memmap has shape (total_max, 3) with columns [X, Y, Z].
    """
    # Pre-count to size the allocation
    total_max = 0
    for laz_path in laz_files:
        try:
            with laspy.open(laz_path) as f:
                total_max += f.header.point_count
        except Exception:
            total_max += 5_000_000

    mm_path = os.path.join(tmp_dir, "pts.dat")
    mm = np.memmap(mm_path, dtype=np.float64, mode='w+', shape=(total_max, 3))

    write_idx = 0
    xmin = ymin =  np.inf
    xmax = ymax = -np.inf
    n_files = len(laz_files)

    for fi, laz_path in enumerate(laz_files):
        fname = os.path.basename(laz_path)
        base_pct = int(fi / n_files * max_pct)
        with laspy.open(laz_path) as f:
            total_tile = f.header.point_count
            done = 0
            try:
                for chunk in f.chunk_iterator(500_000):
                    if selected_classes is not None:
                        mask = np.isin(chunk.classification, selected_classes)
                        x = np.asarray(chunk.x[mask], dtype=np.float64)
                        y = np.asarray(chunk.y[mask], dtype=np.float64)
                        z = np.asarray(chunk.z[mask], dtype=np.float64)
                    else:
                        x = np.asarray(chunk.x, dtype=np.float64)
                        y = np.asarray(chunk.y, dtype=np.float64)
                        z = np.asarray(chunk.z, dtype=np.float64)

                    n = len(x)
                    if n > 0:
                        mm[write_idx:write_idx + n, 0] = x
                        mm[write_idx:write_idx + n, 1] = y
                        mm[write_idx:write_idx + n, 2] = z
                        write_idx += n
                        xmin = min(xmin, x.min()); xmax = max(xmax, x.max())
                        ymin = min(ymin, y.min()); ymax = max(ymax, y.max())

                    done += len(chunk)
                    if total_tile > 0:
                        tile_pct = int(done / total_tile * (max_pct // max(n_files, 1)))
                        progress_callback(
                            f"Reading {fi+1}/{n_files}: {fname} — {int(done/total_tile*100)}%",
                            base_pct + tile_pct,
                        )
            except Exception as e:
                is_lazrs = (LazrsError and isinstance(e, LazrsError)) or \
                           "failed to fill whole buffer" in str(e)
                if not is_lazrs or write_idx == 0:
                    raise e
        mm.flush()

    return mm_path, write_idx, xmin, xmax, ymin, ymax




# ---------------------------------------------------------------------------
# KD-tree query strategies
# ---------------------------------------------------------------------------

def _direct_parallel_query(mm_path, n_pts, grid_x, grid_y, max_dist,
                            n_workers, progress_callback, base_pct, range_pct):
    """
    Load all points into RAM, build one KD-tree, query in parallel chunks.
    Use when dataset fits comfortably in RAM.
    """
    from scipy.spatial import cKDTree

    mm = np.memmap(mm_path, dtype=np.float64, mode='r', shape=(n_pts, 3))
    progress_callback(f"Loading {n_pts:,} points into RAM...", base_pct)
    pts_xy = np.array(mm[:, :2])
    pts_z  = np.array(mm[:, 2])
    del mm

    progress_callback(f"Building KD-tree for {n_pts:,} points...", base_pct + 3)
    tree = cKDTree(pts_xy)
    del pts_xy

    n_grid = len(grid_x)
    gz_out = np.zeros(n_grid)
    chunk_size = max(200_000, n_grid // (n_workers * 8))
    n_chunks = (n_grid + chunk_size - 1) // chunk_size
    completed = [0]
    lock = threading.Lock()

    def _query_chunk(start, end):
        gxy = np.column_stack([grid_x[start:end], grid_y[start:end]])
        dists, idxs = tree.query(gxy, k=1)
        gz = pts_z[idxs].copy()
        gz[dists > max_dist] = 0.0
        return start, end, gz

    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(_query_chunk, i, min(i + chunk_size, n_grid)): i
            for i in range(0, n_grid, chunk_size)
        }
        for fut in as_completed(futs):
            start, end, gz = fut.result()
            gz_out[start:end] = gz
            with lock:
                completed[0] += 1
                pct = base_pct + int(completed[0] / n_chunks * range_pct)
            progress_callback(f"Query chunk {completed[0]}/{n_chunks}", pct)

    return gz_out


def _tiled_parallel_query(mm_path, n_pts, grid_x, grid_y, max_dist, tile_size,
                           n_workers, progress_callback, base_pct, range_pct):
    """
    Memory-safe tiled KD-tree query using streaming batched reads.

    Instead of sorting all N points (which requires a contiguous N*8-byte
    index array that fails for billion-point clouds), we:
      1. Divide grid into spatial tiles and pre-compute their bboxes.
      2. Group tiles into batches sized to keep RAM usage ≤ ~8 GB.
      3. For each batch: one sequential pass through the memmap, routing
         points to the tiles in the batch.
      4. Process collected points per tile with a local KD-tree in parallel.

    Peak RAM per batch ≈ (READ_CHUNK * 24 bytes) + (batch point buffers).
    No giant sort-index array is ever allocated.
    """
    from scipy.spatial import cKDTree

    n_grid = len(grid_x)
    gz_out = np.zeros(n_grid)

    # ── Assign each grid point to a tile ────────────────────────────────────
    x_min, x_max = grid_x.min(), grid_x.max()
    y_min, y_max = grid_y.min(), grid_y.max()
    x_bins = np.arange(x_min, x_max + tile_size, tile_size)
    y_bins = np.arange(y_min, y_max + tile_size, tile_size)
    xi = np.clip(np.searchsorted(x_bins, grid_x, side='right') - 1, 0, len(x_bins) - 1)
    yi = np.clip(np.searchsorted(y_bins, grid_y, side='right') - 1, 0, len(y_bins) - 1)
    tile_ids = xi * len(y_bins) + yi
    unique_tiles = np.unique(tile_ids)
    n_tiles = len(unique_tiles)

    # Pre-compute per-tile masks and bboxes (cheap — just grid points)
    tile_masks = {}
    tile_bboxes = {}
    for tid in unique_tiles:
        tm = tile_ids == tid
        tile_masks[tid] = tm
        tgx = grid_x[tm];  tgy = grid_y[tm]
        tile_bboxes[tid] = (
            tgx.min() - max_dist, tgx.max() + max_dist,
            tgy.min() - max_dist, tgy.max() + max_dist,
        )

    # ── Compute batch size to target ≤ 8 GB for point buffers ───────────────
    pts_per_tile_est = max(1, n_pts // n_tiles)
    bytes_per_tile   = pts_per_tile_est * 3 * 8
    TARGET_BYTES     = 8 * 1024 ** 3            # 8 GiB ceiling
    batch_size = max(1, min(n_tiles, int(TARGET_BYTES / bytes_per_tile)))
    tile_batches = [unique_tiles[i:i + batch_size]
                    for i in range(0, n_tiles, batch_size)]
    n_batches = len(tile_batches)

    READ_CHUNK = 1_000_000   # points read from memmap at a time (≈24 MB)
    tiles_done = [0]

    progress_callback(
        f"Tiled mode: {n_tiles} tiles × {tile_size:.0f} m  |  "
        f"{n_batches} pass(es) through data  |  {n_workers} workers",
        base_pct,
    )

    for batch_idx, tile_batch in enumerate(tile_batches):

        progress_callback(
            f"Pass {batch_idx + 1}/{n_batches}: collecting points for "
            f"{len(tile_batch)} tiles...",
            base_pct + int(batch_idx / n_batches * range_pct * 0.75),
        )

        # Initialise per-tile buffers
        tile_bufs   = {tid: [] for tid in tile_batch}
        bboxes_now  = {tid: tile_bboxes[tid] for tid in tile_batch}

        # Single sequential pass through the memmap
        mm = np.memmap(mm_path, dtype=np.float64, mode='r', shape=(n_pts, 3))
        for start in range(0, n_pts, READ_CHUNK):
            end   = min(start + READ_CHUNK, n_pts)
            chunk = np.array(mm[start:end])        # copy READ_CHUNK rows to RAM
            cx = chunk[:, 0];  cy = chunk[:, 1]

            for tid, (xlo, xhi, ylo, yhi) in bboxes_now.items():
                mask = (cx >= xlo) & (cx <= xhi) & (cy >= ylo) & (cy <= yhi)
                if mask.any():
                    tile_bufs[tid].append(chunk[mask])
        del mm

        # ── Process each tile in this batch in parallel ──────────────────────
        lock = threading.Lock()

        def _process_tile(tid):
            bufs = tile_bufs[tid]
            tm   = tile_masks[tid]
            tgx  = grid_x[tm];  tgy = grid_y[tm]

            if not bufs:
                return tid, tm, np.zeros(len(tgx))

            local = np.concatenate(bufs)
            tree  = cKDTree(local[:, :2])
            dists, idxs = tree.query(np.column_stack([tgx, tgy]), k=1)
            gz = local[idxs, 2]
            gz[dists > max_dist] = 0.0
            return tid, tm, gz

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(_process_tile, tid): tid for tid in tile_batch}
            for fut in as_completed(futs):
                tid, tm, gz = fut.result()
                gz_out[tm] = gz
                with lock:
                    tiles_done[0] += 1
                    pct = base_pct + int(tiles_done[0] / n_tiles * range_pct)
                progress_callback(f"Tiles done: {tiles_done[0]}/{n_tiles}", pct)

        del tile_bufs   # free point buffers before next batch

    return gz_out


# ---------------------------------------------------------------------------
# Core processing — memory-adaptive
# ---------------------------------------------------------------------------

# Points above this threshold → tiled mode (avoids one giant KD-tree)
_TILE_MODE_THRESHOLD = 15_000_000   # ~15 M pts ≈ ~4 GB KD-tree overhead


def process(
    laz_files,
    grid_mode,
    grid_size, angle_deg,
    centerline_pts,
    chainage_spacing, cross_width, cross_spacing,
    max_dist, selected_classes,
    output_path,
    progress_callback, finish_callback,
    n_workers=None, tile_size=500.0,
):
    tmp_dir = None
    try:
        from scipy.spatial import cKDTree

        if n_workers is None:
            n_workers = max(1, min(4, (os.cpu_count() or 4) // 2))

        n_laz = len(laz_files)

        # ── Phase 1: Stream LAZ → memmap ────────────────────────────────────
        progress_callback(f"Streaming {n_laz} LAZ file(s) to disk cache...", 0)
        tmp_dir = tempfile.mkdtemp(prefix="lazgrid_")

        mm_path, n_pts, laz_xmin, laz_xmax, laz_ymin, laz_ymax = _read_laz_to_memmap(
            laz_files, selected_classes, tmp_dir, progress_callback, max_pct=40,
        )

        if n_pts == 0:
            raise ValueError("No points found with the selected classifications.")

        progress_callback(f"Cached {n_pts:,} points to disk. Generating grid...", 41)

        # ── Phase 2: Generate grid ───────────────────────────────────────────
        if grid_mode == "angle":
            theta_rad = np.radians(angle_deg)
            cos_t = np.cos(-theta_rad);  sin_t = np.sin(-theta_rad)

            # Compute rotated bounds from memmap (read X,Y only — avoids loading Z)
            mm_ro = np.memmap(mm_path, dtype=np.float64, mode='r', shape=(n_pts, 3))
            px = mm_ro[:, 0];  py = mm_ro[:, 1]
            rx = px * cos_t - py * sin_t
            ry = px * sin_t + py * cos_t
            rx_min, rx_max = float(rx.min()), float(rx.max())
            ry_min, ry_max = float(ry.min()), float(ry.max())
            del px, py, rx, ry, mm_ro

            grid_x, grid_y = generate_angle_grid(
                rx_min, rx_max, ry_min, ry_max, None, None, grid_size, angle_deg
            )
            grid_ch = grid_off = np.zeros(len(grid_x))
            extra_cols = False
            total_length = n_sections = 0

        else:  # dxf
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
            gx_min, gx_max = grid_x.min(), grid_x.max()
            gy_min, gy_max = grid_y.min(), grid_y.max()
            x_ok = gx_min <= laz_xmax and gx_max >= laz_xmin
            y_ok = gy_min <= laz_ymax and gy_max >= laz_ymin
            if not (x_ok and y_ok):
                raise ValueError(
                    "Coordinate mismatch: DXF centreline and LAZ data do not overlap.\n\n"
                    f"DXF grid extent:  X [{gx_min:.1f} – {gx_max:.1f}]  "
                    f"Y [{gy_min:.1f} – {gy_max:.1f}]\n"
                    f"LAZ data extent:  X [{laz_xmin:.1f} – {laz_xmax:.1f}]  "
                    f"Y [{laz_ymin:.1f} – {laz_ymax:.1f}]\n\n"
                    "Use the DXF Local UCS → World Transform fields to apply an offset:\n"
                    f"  Offset X ≈ {laz_xmin - gx_min:.1f}\n"
                    f"  Offset Y ≈ {laz_ymin - gy_min:.1f}"
                )

        n_grid = len(grid_x)
        progress_callback(f"Grid: {n_grid:,} points. Starting elevation query...", 44)

        # ── Phase 3: KD-tree elevation query ─────────────────────────────────
        use_tiled = n_pts > _TILE_MODE_THRESHOLD

        if use_tiled:
            progress_callback(
                f"Large dataset ({n_pts:,} pts) — tiled streaming mode "
                f"(tile={tile_size:.0f} m, workers={n_workers})",
                45,
            )
            grid_z = _tiled_parallel_query(
                mm_path, n_pts,
                grid_x, grid_y, max_dist, tile_size,
                n_workers, progress_callback, base_pct=46, range_pct=44,
            )
        else:
            progress_callback(
                f"Dataset ({n_pts:,} pts) — direct mode (workers={n_workers})",
                45,
            )
            grid_z = _direct_parallel_query(
                mm_path, n_pts, grid_x, grid_y, max_dist,
                n_workers, progress_callback, base_pct=46, range_pct=44,
            )

        n_zero = int(np.sum(grid_z == 0.0))
        if n_zero == n_grid:
            progress_callback(
                "WARNING: ALL grid points are beyond max_dist — "
                "check max_dist or verify coordinate system.", 90,
            )
        else:
            progress_callback(f"{n_zero:,} grid points set to Z=0 (>{max_dist} m).", 90)

        # ── Phase 4: Write CSV ───────────────────────────────────────────────
        progress_callback("Writing output CSV...", 91)
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
                chunk = data[i:i + wc]
                np.savetxt(f_out, chunk, delimiter=",", fmt="%.3f")
                done = i + len(chunk)
                progress_callback(
                    f"Writing CSV... {int(done/total_rows*100)}%",
                    91 + int(done / total_rows * 9),
                )

        mode_detail = (
            f"Rotation angle : {angle_deg}°\nGrid cell size : {grid_size} m\n"
            if grid_mode == "angle"
            else (
                f"Centreline length  : {total_length:.1f} m\n"
                f"Cross-sections     : {n_sections}\n"
            )
        )
        finish_callback(
            True,
            f"Grid generated successfully!\n\n"
            f"Mode             : {'Tiled' if use_tiled else 'Direct'} "
            f"({n_workers} worker(s))\n"
            f"LAZ tiles read   : {n_laz}\n"
            f"Total LAZ points : {n_pts:,}\n"
            + mode_detail +
            f"Grid points      : {total_rows:,}\n"
            f"Zero-elev points : {n_zero:,}\n\n"
            f"Saved to: {output_path}",
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        finish_callback(False, f"An error occurred:\n{str(e)}")

    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------

CLASS_NAMES = {
    0: "Never Classified", 1: "Unclassified",     2: "Ground",
    3: "Low Vegetation",   4: "Medium Vegetation", 5: "High Vegetation",
    6: "Building",         7: "Low Point",          9: "Water", 12: "Overlap",
}


def find_laz_files(folder):
    return sorted(
        glob.glob(os.path.join(folder, "*.laz")) +
        glob.glob(os.path.join(folder, "*.las"))
    )


class LazGridGenerator:
    def __init__(self, root):
        self.root = root
        self.root.title("LAZ Grid Generator v0.7")
        self.root.geometry("800x1020")
        self.root.resizable(False, False)

        # LAZ state
        self.laz_mode    = tk.StringVar(value="single")
        self.laz_display = tk.StringVar()
        self._laz_files  = []

        # Grid method
        self.grid_mode = tk.StringVar(value="dxf")

        # Angle-mode params
        self.grid_size = tk.DoubleVar(value=1.0)
        self.angle     = tk.DoubleVar(value=0.0)

        # DXF-mode state
        self.dxf_display    = tk.StringVar()
        self._dxf_polylines = []
        self._selected_poly = tk.IntVar(value=-1)

        # DXF-mode params
        self.chainage_spacing = tk.DoubleVar(value=5.0)
        self.cross_width      = tk.DoubleVar(value=10.0)
        self.cross_spacing    = tk.DoubleVar(value=1.0)

        # DXF UCS transform
        self.dxf_offset_x = tk.DoubleVar(value=0.0)
        self.dxf_offset_y = tk.DoubleVar(value=0.0)
        self.dxf_rotation = tk.DoubleVar(value=0.0)

        # Shared
        self.max_dist = tk.DoubleVar(value=2.0)
        self.status   = tk.StringVar(value="Ready")
        self.selected_classes = {}

        # Performance
        default_workers = max(1, min(4, (os.cpu_count() or 4) // 2))
        self.n_workers = tk.IntVar(value=default_workers)
        self.tile_size = tk.DoubleVar(value=500.0)

        self._warn_no_ezdxf()
        self._build_ui()

    def _warn_no_ezdxf(self):
        if not HAS_EZDXF:
            messagebox.showwarning(
                "Missing dependency",
                "ezdxf is not installed.\nDXF mode requires:  pip install ezdxf\n"
                "Angle-based mode works without it.",
            )

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
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
                       command=self._on_grid_mode_change).pack(side=tk.LEFT, padx=(0, 20))
        tk.Radiobutton(gmr, text="DXF Centreline  (chainage / cross-section)",
                       variable=self.grid_mode, value="dxf",
                       command=self._on_grid_mode_change).pack(side=tk.LEFT)

        self.angle_frame = tk.LabelFrame(main, text="Angle-based Parameters", padx=8, pady=5)
        self._build_angle_params(self.angle_frame)

        self.dxf_frame = tk.LabelFrame(main, text="DXF Centreline Parameters", padx=8, pady=5)
        self._build_dxf_params(self.dxf_frame)

        self._shared_frame = tk.Frame(main)
        self._shared_frame.pack(fill=tk.X, pady=(4, 4))
        tk.Label(self._shared_frame, text="Max Distance (m):").pack(side=tk.LEFT)
        tk.Entry(self._shared_frame, textvariable=self.max_dist, width=10).pack(
            side=tk.LEFT, padx=8
        )
        tk.Label(self._shared_frame,
                 text="Grid points further than this → Z=0",
                 fg="grey").pack(side=tk.LEFT)

        self._on_grid_mode_change()

        # ── Performance ───────────────────────────────────────────────────────
        perf_lf = tk.LabelFrame(main, text="Performance  (chunked + parallel processing)",
                                padx=8, pady=5)
        perf_lf.pack(fill=tk.X, pady=(0, 6))

        tk.Label(perf_lf, text="Workers:").grid(row=0, column=0, sticky="w")
        tk.Spinbox(perf_lf, from_=1, to=16, textvariable=self.n_workers,
                   width=5).grid(row=0, column=1, padx=8, sticky="w")
        tk.Label(perf_lf,
                 text=f"Parallel threads for KD-tree queries  "
                      f"(auto-default: {self.n_workers.get()})",
                 fg="grey").grid(row=0, column=2, sticky="w")

        tk.Label(perf_lf, text="Tile Size (m):").grid(row=1, column=0, sticky="w", pady=2)
        tk.Entry(perf_lf, textvariable=self.tile_size, width=10).grid(
            row=1, column=1, padx=8, sticky="w"
        )
        tk.Label(perf_lf,
                 text=f"Spatial tile size for large datasets (>{_TILE_MODE_THRESHOLD//1_000_000}M pts)",
                 fg="grey").grid(row=1, column=2, sticky="w")

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
        tk.Label(parent, text="DXF File:").grid(row=0, column=0, sticky="w", pady=2)
        dxf_ef = tk.Frame(parent)
        dxf_ef.grid(row=0, column=1, columnspan=2, sticky="ew", pady=2)
        tk.Entry(dxf_ef, textvariable=self.dxf_display, state="readonly", width=36).pack(
            side=tk.LEFT
        )
        tk.Button(dxf_ef, text="Browse", command=self.load_dxf).pack(side=tk.LEFT, padx=4)

        tk.Label(parent, text="Select Polyline:").grid(row=1, column=0, sticky="nw", pady=(4, 2))
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
            self.poly_inner, text="  Load a DXF file to see polylines.", fg="grey", bg="white"
        )
        self.poly_hint.pack(anchor="w", padx=4, pady=4)

        for i, (lbl, var, hint) in enumerate([
            ("Chainage Spacing (m):",      self.chainage_spacing, "Along-centreline interval"),
            ("Cross-Section Width (m):",   self.cross_width,      "Half-width ± from centreline"),
            ("Cross-Section Spacing (m):", self.cross_spacing,    "Point spacing within each cross-section"),
        ], start=2):
            tk.Label(parent, text=lbl).grid(row=i, column=0, sticky="w", pady=2)
            tk.Entry(parent, textvariable=var, width=10).grid(row=i, column=1, padx=8, sticky="w")
            tk.Label(parent, text=hint, fg="grey").grid(row=i, column=2, padx=5, sticky="w")

        sep = ttk.Separator(parent, orient="horizontal")
        sep.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(8, 4))

        tk.Label(
            parent,
            text="DXF Local UCS → World Transform  (leave at 0 if DXF is already in world coords)",
            fg="#555555", font=("Arial", 8, "italic"),
        ).grid(row=6, column=0, columnspan=3, sticky="w", pady=(0, 4))

        for i, (lbl, var, hint) in enumerate([
            ("Offset X (m):",     self.dxf_offset_x, "Add to every DXF X coordinate"),
            ("Offset Y (m):",     self.dxf_offset_y, "Add to every DXF Y coordinate"),
            ("UCS Rotation (°):", self.dxf_rotation, "CCW rotation of local UCS (applied before offset)"),
        ], start=7):
            tk.Label(parent, text=lbl).grid(row=i, column=0, sticky="w", pady=2)
            tk.Entry(parent, textvariable=var, width=14).grid(row=i, column=1, padx=8, sticky="w")
            tk.Label(parent, text=hint, fg="grey").grid(row=i, column=2, padx=5, sticky="w")

    # ── Mode switching ────────────────────────────────────────────────────────

    def _on_laz_mode_change(self):
        self.laz_display.set("")
        self.laz_info_var.set("")
        self._laz_files = []
        self._clear_classes()
        self._check_enable()

    def _on_grid_mode_change(self):
        if self.grid_mode.get() == "angle":
            self.dxf_frame.pack_forget()
            self.angle_frame.pack(fill=tk.X, pady=(0, 0), before=self._shared_frame)
        else:
            self.angle_frame.pack_forget()
            self.dxf_frame.pack(fill=tk.X, pady=(0, 0), before=self._shared_frame)
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
                text="  No LWPOLYLINE/POLYLINE entities found.",
                fg="red", bg="white",
            ).pack(anchor="w", padx=4, pady=4)
            self._update_progress("No polylines found in DXF.", 0)
            return

        self._dxf_polylines = polys
        longest_idx = max(range(len(polys)), key=lambda i: polys[i]["arc_length"])
        self._selected_poly.set(longest_idx)

        for i, pl in enumerate(polys):
            lbl = (
                f"  Layer: {pl['layer']:<20s}  "
                f"Length: {pl['arc_length']:>10.2f} m  "
                f"Vertices: {pl['n_vertices']:>4d}  [{pl['etype']}]"
            )
            tk.Radiobutton(
                self.poly_inner, text=lbl, variable=self._selected_poly, value=i,
                anchor="w", justify=tk.LEFT, bg="white", command=self._check_enable,
                font=("Courier", 9),
            ).pack(anchor="w", fill=tk.X, padx=2)

        self._update_progress(
            f"DXF loaded: {len(polys)} polyline(s). Select the road/track centreline.", 0
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
        if self.grid_mode.get() == "angle":
            ok = laz_ok
        else:
            ok = laz_ok and (self._selected_poly.get() >= 0)
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
            grid_s  = self.grid_size.get()
            ang     = self.angle.get()
            ch_sp   = self.chainage_spacing.get()
            cr_w    = self.cross_width.get()
            cr_sp   = self.cross_spacing.get()
            mx_d    = self.max_dist.get()
            workers = int(self.n_workers.get())
            ts      = self.tile_size.get()
        except tk.TclError:
            messagebox.showerror("Error", "Invalid numeric value in parameters.")
            return

        if mx_d <= 0:
            messagebox.showerror("Error", "Max Distance must be positive.")
            return
        if mode == "angle" and grid_s <= 0:
            messagebox.showerror("Error", "Grid Size must be positive.")
            return
        if mode == "dxf" and (ch_sp <= 0 or cr_w <= 0 or cr_sp <= 0):
            messagebox.showerror("Error", "Chainage/cross-section values must be positive.")
            return
        if workers < 1:
            messagebox.showerror("Error", "Workers must be at least 1.")
            return
        if ts <= 0:
            messagebox.showerror("Error", "Tile Size must be positive.")
            return

        output = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile="grid_output.csv",
        )
        if not output:
            return

        self.generate_btn["state"] = tk.DISABLED
        finish_cb = lambda s, m: self.root.after(0, lambda: self._on_finish(s, m))

        threading.Thread(
            target=process,
            args=(
                list(self._laz_files), mode,
                grid_s, ang,
                centerline_pts,
                ch_sp, cr_w, cr_sp,
                mx_d, selected,
                output,
                self._update_progress, finish_cb,
                workers, ts,
            ),
            daemon=True,
        ).start()


if __name__ == "__main__":
    root = tk.Tk()
    app = LazGridGenerator(root)
    root.mainloop()
