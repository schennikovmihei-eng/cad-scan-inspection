#!/usr/bin/env python3
"""Test 24 proper orientations around CAD principal axes, then refine with ICP.

Place next to refine_modelnet_icp.py and run in the geotransformer environment.
The manual registration is a diagnostic control, never an orientation candidate
or a selection input. Ranking uses the same fixed source/CAD point samples.
"""

import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree
from refine_modelnet_icp import apply_transform, file_hash, latest_run, point_cloud, refine, summarize


def orientation_hypotheses(cad):
    values, basis = np.linalg.eigh(np.cov(cad, rowvar=False))
    basis = basis[:, ::-1].copy()
    if np.linalg.det(basis) < 0:
        basis[:, -1] *= -1
    rotations = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            local = np.eye(3)[:, permutation] @ np.diag(signs)
            if np.linalg.det(local) > 0:
                rotations.append(basis @ local @ basis.T)
    rotations.sort(key=lambda r: (round(float(np.linalg.norm(r - np.eye(3))), 10),
                                  tuple(np.round(r.ravel(), 10))))
    if len(rotations) != 24 or not np.allclose(rotations[0], np.eye(3), atol=1e-10):
        raise ValueError("Unexpected principal-axis rotation group")
    rotations[0] = np.eye(3)
    center = cad.mean(axis=0)
    hypotheses = []
    for rotation in rotations:
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = center - rotation @ center
        hypotheses.append(transform)
    return hypotheses, values[::-1]


def metric_key(metrics):
    return metrics["mean"], metrics["p95"]


def rotation_only(matrix):
    u, _, vt = np.linalg.svd(matrix[:3, :3])
    if np.linalg.det(u @ vt) < 0:
        u[:, -1] *= -1
    return u @ vt


def rotation_difference_degrees(first, second):
    relative = rotation_only(first) @ rotation_only(second).T
    return float(np.degrees(np.arccos(np.clip((np.trace(relative) - 1) / 2, -1, 1))))


def search(original_scan, cad, initial_transform, thresholds, iterations, o3d, log=True):
    tree = cKDTree(cad)
    hypotheses, variances = orientation_hypotheses(cad)
    records, transforms, starts = [], [], []
    for index, hypothesis in enumerate(hypotheses):
        start = hypothesis @ initial_transform
        initial_score = summarize(apply_transform(original_scan, start), tree)
        refined, _, stages, seconds = refine(original_scan, cad, start, thresholds, iterations, o3d)
        refined_score = summarize(apply_transform(original_scan, refined), tree)
        use_refined = metric_key(refined_score) < metric_key(initial_score)
        chosen, score = (refined, refined_score) if use_refined else (start, initial_score)
        starts.append(start)
        transforms.append(chosen)
        records.append({"candidate": index, "initial": initial_score, "after_icp": refined_score,
                        "selected": score, "used_icp_result": use_refined,
                        "icp_seconds": seconds, "stages": stages})
        if log:
            print("{:02d}/24: mean={:.10g}, p95={:.10g}".format(
                index + 1, score["mean"], score["p95"]), flush=True)
    best = min(range(len(records)), key=lambda i: metric_key(records[i]["selected"]))
    return best, records, np.stack(transforms), np.stack(starts), variances


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path,
                        default=Path("/mnt/c/Users/mihei/Desktop/vkr/registration_input"))
    parser.add_argument("--run", type=Path, help="ModelNet run; default: latest completed run")
    parser.add_argument("--icp-run", type=Path, help="Default: latest completed ICP refinement")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--show", action="store_true", help="Show only the best-ranked candidate")
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    inputs = args.inputs.expanduser().resolve()
    run = args.run.expanduser().resolve() if args.run else latest_run(inputs)
    parent = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    for name in ("src", "ref"):
        if file_hash(inputs / (name + ".npy")) != parent["base_" + name + "_sha256"]:
            raise ValueError("Input changed since the ModelNet run: " + name)
    if args.icp_run:
        icp_run = args.icp_run.expanduser().resolve()
    else:
        runs = sorted(path for path in run.glob("icp_*")
                      if (path / "metrics.json").is_file()
                      and (path / "transform_scan_to_cad.npy").is_file())
        if not runs:
            raise FileNotFoundError("Run refine_modelnet_icp.py first")
        icp_run = runs[-1]
    icp_report = json.loads((icp_run / "metrics.json").read_text(encoding="utf-8"))
    if file_hash(run / "metrics.json") != icp_report["parent_metrics_sha256"]:
        raise ValueError("ICP refinement does not match the selected ModelNet report")
    with np.load(inputs / "preparation.npz") as saved:
        scan_scale = float(saved["scan_to_cad_scale"])
        network_scale = float(saved["network_scale"])
    original = np.load(inputs / "src.npy").astype(np.float64) / (scan_scale * network_scale)
    cad = np.load(inputs / "ref.npy").astype(np.float64) / network_scale
    initial = np.load(icp_run / "transform_scan_to_cad.npy")
    if initial.shape != (4, 4) or not np.isfinite(initial).all():
        raise ValueError("Invalid initial transform")
    tree = cKDTree(cad)
    before = summarize(apply_transform(original, initial), tree)
    expected = icp_report["after"]
    if not np.allclose([before["mean"], before["p95"]], [expected["mean"], expected["p95"]],
                       rtol=1e-6, atol=1e-10):
        raise ValueError("Saved ICP diagnostics do not match the current inputs")
    import open3d as o3d
    extent = np.ptp(cad, axis=0)
    diagonal = float(np.linalg.norm(extent))
    thresholds = [0.10 * diagonal, 0.05 * diagonal, 0.02 * float(extent.max())]
    print("Using:", icp_run, flush=True)
    print("Current: mean={:.10g}, p95={:.10g}".format(before["mean"], before["p95"]), flush=True)
    started = perf_counter()
    best, records, transforms, starts, variances = search(
        original, cad, initial, thresholds, args.iterations, o3d
    )
    seconds = perf_counter() - started
    metrics = records[best]["selected"]
    print("Best candidate:", best + 1, flush=True)
    print("Best: mean={:.10g}, p95={:.10g}".format(metrics["mean"], metrics["p95"]), flush=True)

    # This comparison happens AFTER geometric selection and does not affect it.
    angle_to_manual = None
    alignment = inputs.parent / "aligment.npz"
    if alignment.exists():
        with np.load(alignment) as saved:
            manual_rotation_matrix = saved["icp_transform"] @ saved["manual_transform"]
        angle_to_manual = rotation_difference_degrees(transforms[best], manual_rotation_matrix)
        print("Best vs saved manual rotation (deg): {:.6f}".format(angle_to_manual), flush=True)
    baseline = parent.get("metrics", {}).get("Manual + ICP")
    if baseline:
        print("Manual + ICP: mean={:.10g}, p95={:.10g}".format(
            baseline["mean"], baseline["p95"]), flush=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = icp_run / ("orientations_" + stamp)
    output.mkdir()
    np.save(output / "transform_scan_to_cad.npy", transforms[best])
    np.savez(output / "candidates.npz", transforms=transforms, initial_transforms=starts,
             base_transform=initial, cad_principal_variances=variances)
    metadata = {
        "modelnet_run": str(run), "icp_run": str(icp_run),
        "parent_metrics_sha256": file_hash(icp_run / "metrics.json"),
        "base_src_sha256": file_hash(inputs / "src.npy"),
        "base_ref_sha256": file_hash(inputs / "ref.npy"),
        "open3d": o3d.__version__, "strategy": "24 proper axis permutations in CAD PCA frame + ICP",
        "rotation_center": cad.mean(axis=0).tolist(), "thresholds": thresholds,
        "max_iterations_per_stage": args.iterations, "seconds": seconds,
        "candidate_count": len(records), "best_candidate_zero_based": best,
        "selection": "minimum all-source-point mean distance; p95 breaks ties",
        "manual_pose_used_for_selection": False,
        "best_vs_saved_manual_rotation_degrees_diagnostic_only": angle_to_manual,
        "metric": "one-way nearest distance to the same fixed CAD POINT SAMPLE",
        "units": "original CAD coordinate units; physical unit not verified",
        "before": before, "best": metrics, "manual_icp_saved_control": baseline,
        "candidates": records,
        "interpretation": "Geometric ranking alone does not certify anatomical orientation",
    }
    (output / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Saved:", output, flush=True)
    if args.show:
        target = point_cloud(cad, o3d)
        target.paint_uniform_color([1, 0.8, 0])
        source = point_cloud(apply_transform(original, transforms[best]), o3d)
        source.paint_uniform_color([0.1, 0.4, 1])
        o3d.visualization.draw_geometries(
            [target, source], window_name="Best orientation candidate {} AFTER ICP".format(best + 1)
        )


if __name__ == "__main__":
    main()
