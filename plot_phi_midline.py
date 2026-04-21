#!/usr/bin/env python3
"""Plot phi along x at y = NY/2 for several VTK snapshots.

Expected input files are ASCII VTK files written by the solver in the form:
  data_000000.vtk, data_000050.vtk, ...

The script plots the midline phi profile for 5 snapshots by default:
  first, 25%, 50%, 75% and last.

Usage:
  python plot_phi_midline.py --dir <output_folder>
  python plot_phi_midline.py --dir <output_folder> --save phi_midline.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np


VTK_STEP_RE = re.compile(r"data_(\d+)\.vtk$")


def list_vtk_files(folder: Path) -> List[Path]:
    files = []
    for p in folder.glob("data_*.vtk"):
        if VTK_STEP_RE.search(p.name):
            files.append(p)
    files.sort(key=lambda p: int(VTK_STEP_RE.search(p.name).group(1)))
    return files


def parse_vtk_structured_points_phi(path: Path) -> Tuple[int, int, np.ndarray]:
    """Return NX, NY and phi as a flat array of length NX*NY.

    This matches the writer in post_process.cu, which stores phi as an ASCII
    SCALARS block in a STRUCTURED_POINTS dataset.
    """
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()

    nx = ny = None
    phi_start = None
    num_points = None

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith("DIMENSIONS"):
            parts = line.split()
            nx = int(parts[1])
            ny = int(parts[2])
        elif line.startswith("POINT_DATA"):
            num_points = int(line.split()[1])
        elif line.startswith("SCALARS phi"):
            # next line is LOOKUP_TABLE default, then phi values begin
            phi_start = i + 2
            break
        i += 1

    if nx is None or ny is None or phi_start is None or num_points is None:
        raise ValueError(f"Could not parse VTK header in {path}")

    values = []
    for line in lines[phi_start:]:
        s = line.strip()
        if not s:
            continue
        # Stop before the next field if present.
        if s.startswith("SCALARS ") or s.startswith("VECTORS "):
            break
        for v in s.split():
            try:
                val = float(v)
                if np.isnan(val) or np.isinf(val):
                    val = np.nan
                values.append(val)
            except ValueError:
                # trata casos tipo -nan(ind)
                values.append(np.nan)
        if len(values) >= num_points:
            break

    if len(values) < num_points:
        raise ValueError(
            f"Expected {num_points} phi values in {path}, got {len(values)}"
        )

    return nx, ny, np.asarray(values[:num_points], dtype=float)


def choose_snapshots(files: List[Path]) -> List[Path]:
    if not files:
        raise ValueError("No VTK files found.")

    n = len(files)
    # 5 snapshots: first, 1/4, 1/2, 3/4, last.
    idxs = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
    # Remove duplicates while preserving order.
    seen = set()
    chosen = []
    for idx in idxs:
        idx = max(0, min(n - 1, idx))
        if idx not in seen:
            seen.add(idx)
            chosen.append(files[idx])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Folder containing data_*.vtk files",
    )
    parser.add_argument(
        "--save",
        type=str,
        default="",
        help="Optional output image path. If omitted, the plot is shown on screen.",
    )
    parser.add_argument(
        "--y-index",
        type=int,
        default=None,
        help="Override the midline index. Default is NY//2.",
    )
    args = parser.parse_args()

    folder = Path(args.dir)
    files = list_vtk_files(folder)
    if not files:
        raise SystemExit(f"No data_*.vtk files found in: {folder}")

    chosen = choose_snapshots(files)

    plt.figure(figsize=(10, 6))

    for path in chosen:
        nx, ny, phi = parse_vtk_structured_points_phi(path)
        if args.y_index is None:
            y = ny // 2
        else:
            y = max(0, min(ny - 1, args.y_index))

        phi_2d = phi.reshape((ny, nx))
        phi_line = phi_2d[y, :]

        step_match = VTK_STEP_RE.search(path.name)
        step = int(step_match.group(1)) if step_match else -1
        label = f"t={step}"
        plt.plot(np.arange(nx), phi_line, linewidth=2, label=label)

    plt.xlabel("x")
    plt.ylabel("phi")
    plt.title(f"phi ao longo de x em y = NY/2 (ou y={args.y_index})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=200)
        print(f"Saved: {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
