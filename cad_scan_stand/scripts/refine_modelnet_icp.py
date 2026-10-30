#!/usr/bin/env python3
"""Refine an existing ModelNet registration; preserve all earlier results.

Run in Ubuntu's geotransformer environment. Diagnostics use the same fixed
source and CAD point samples as the ModelNet run, in original CAD units.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree


def apply_transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_run(inputs):
    runs = sorted(path for path in inputs.glob("modelnet_*")
                  if path.is_dir() and (path / "metrics.json").is_file()
                  and (path / "transform_scan_to_cad.npy").is_file())
    if not runs:
        raise FileNotFoundError("No completed ModelNet run found in " + str(inputs))
    return runs[-1]


def point_cloud(points, o3d):
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return cloud


def refine(original_scan, cad, initial_transform, thresholds, iterations, o3d):
    """ICP estimates a rigid correction of an already scale-matched source."""
    source = point_cloud(apply_transform(original_scan, initial_transform), o3d)
    target = point_cloud(cad, o3d)
    registration = o3d.pipelines.registration
    correction = np.eye(4)
    stages = []
    started = perf_counter()
    for threshold in thresholds:
        result = registration.registration_icp(
            source, target, float(threshold), correction,
            registration.TransformationEstimationPointToPoint(with_scaling=False),
            registration.ICPConvergenceCriteria(
                relative_fitness=1e-6, relative_rmse=1e-6, max_iteration=iterations
            ),
        )
        # Each result is cumulative relative to the SAME, unchanged source.
        correction = np.asarray(result.transformation, dtype=np.float64).copy()
        if correction.shape != (4, 4) or not np.isfinite(correction).all():
            raise ValueError("ICP returned an invalid transform")
        stages.append({"threshold": float(threshold), "fitness": float(result.fitness),
                       "inlier_rmse": float(result.inlier_rmse)})
    return correction @ initial_transform, correction, stages, perf_counter() - started


def summarize(points, tree):
    distances = tree.query(points)[0]
    return {"count": len(distances), "mean": float(np.mean(distances)),
            "p95": float(np.percentile(distances, 95)), "max": float(np.max(distances))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path,
                        default=Path("/mnt/c/Users/mihei/Desktop/vkr/registration_input"))
    parser.add_argument("--run", type=Path, help="Default: latest completed ModelNet run")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--show", action="store_true", help="Show only the result AFTER ICP")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    inputs = args.inputs.expanduser().resolve()
    run = args.run.expanduser().resolve() if args.run else latest_run(inputs)
    report = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    for name in ("src", "ref"):
        if file_hash(inputs / (name + ".npy")) != report["base_" + name + "_sha256"]:
            raise ValueError("Input changed since this ModelNet run: " + name)
    with np.load(inputs / "preparation.npz") as saved:
        scan_scale = float(saved["scan_to_cad_scale"])
        previous_scale = float(saved["network_scale"])
    if not np.isclose(scan_scale, report["scan_to_cad_scale"], rtol=1e-12, atol=0):
        raise ValueError("Scan calibration changed since this ModelNet run")
    source_base, reference_base = np.load(inputs / "src.npy"), np.load(inputs / "ref.npy")
    original_scan = source_base.astype(np.float64) / (scan_scale * previous_scale)
    cad = reference_base.astype(np.float64) / previous_scale
    initial_transform = np.load(run / "transform_scan_to_cad.npy")
    if initial_transform.shape != (4, 4) or not np.isfinite(initial_transform).all():
        raise ValueError("Invalid saved ModelNet transform")
    tree = cKDTree(cad)
    before = summarize(apply_transform(original_scan, initial_transform), tree)
    expected = report["metrics"]["ModelNet"]
    if not np.allclose([before["mean"], before["p95"]],
                       [expected["mean"], expected["p95"]], rtol=1e-6, atol=1e-10):
        raise ValueError("Saved ModelNet diagnostics do not match the current preparation")
    print("Using ModelNet run:", run, flush=True)
    print("Before ICP: mean={:.10g}, p95={:.10g}".format(before["mean"], before["p95"]), flush=True)

    import open3d as o3d
    extent = np.ptp(cad, axis=0)
    diagonal = float(np.linalg.norm(extent))
    if not np.isfinite(diagonal) or diagonal <= 0:
        raise ValueError("CAD sample has zero or invalid extent")
    thresholds = [0.10 * diagonal, 0.05 * diagonal, 0.02 * float(extent.max())]
    refined, correction, stages, seconds = refine(
        original_scan, cad, initial_transform, thresholds, args.iterations, o3d
    )
    after = summarize(apply_transform(original_scan, refined), tree)
    print("After ICP: mean={:.10g}, p95={:.10g}".format(after["mean"], after["p95"]), flush=True)

    baseline = report.get("metrics", {}).get("Manual + ICP")
    if baseline is not None:
        print("Manual + ICP: mean={:.10g}, p95={:.10g}".format(
            baseline["mean"], baseline["p95"]), flush=True)
    changes = {key: 100.0 * (after[key] / before[key] - 1.0) if before[key] > 0 else None
               for key in ("mean", "p95", "max")}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = run / ("icp_" + stamp)
    output.mkdir()
    np.save(output / "transform_scan_to_cad.npy", refined)
    np.save(output / "initial_transform_scan_to_cad.npy", initial_transform)
    np.save(output / "icp_correction.npy", correction)
    result_report = {
        "parent_modelnet_run": str(run), "parent_metrics_sha256": file_hash(run / "metrics.json"),
        "open3d": o3d.__version__, "method": "point_to_point, rigid correction",
        "thresholds": thresholds, "max_iterations_per_stage": args.iterations,
        "relative_fitness": 1e-6, "relative_rmse": 1e-6,
        "icp_seconds": seconds, "stages": stages,
        "source_point_count": len(original_scan), "reference_point_count": len(cad),
        "metric": "one-way nearest distance to the SAME fixed CAD point sample",
        "units": "original CAD coordinate units; physical unit not verified",
        "scan_to_cad_scale": scan_scale, "previous_network_scale": previous_scale,
        "before": before, "after": after, "manual_icp_saved_control": baseline,
        "change_percent_after_vs_before": changes,
        "interpretation": "Distance diagnostics alone do not establish correct orientation",
    }
    (output / "metrics.json").write_text(json.dumps(result_report, indent=2), encoding="utf-8")
    print("Saved:", output, flush=True)

    if args.show:
        target, source = point_cloud(cad, o3d), point_cloud(apply_transform(original_scan, refined), o3d)
        target.paint_uniform_color([1, 0.8, 0])
        source.paint_uniform_color([0.1, 0.4, 1])
        o3d.visualization.draw_geometries([target, source], window_name="ModelNet + ICP: AFTER refinement")


if __name__ == "__main__":
    main()
