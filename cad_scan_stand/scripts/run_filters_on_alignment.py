#!/usr/bin/env python3
"""Compare SOR and Pointfilter on one fixed, validated alignment.

Run with the Windows .venv Python. Pointfilter runs in a separate .venv-ai
process, using the model and loader from the existing Pointfilter repository.
The input, old programs, old alignment caches and old results are preserved.
"""

import argparse
import csv
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np


FIELDS = ["method", "nb_neighbors", "std_ratio", "kept", "removed", "removed_percent",
          "mean", "median", "std", "p95", "max", "filter_seconds",
          "mean_change_percent", "p95_change_percent", "std_change_percent"]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_alignment(project):
    paths = sorted(path for path in (project / "registration_input").glob(
        "modelnet_*/icp_*/orientations_*/full_scan_*")
        if (path / "aligned_scan.npy").is_file() and (path / "metrics.json").is_file())
    if not paths:
        raise FileNotFoundError("Run evaluate_full_registration.py first")
    return paths[-1]


def distance_summary(distances):
    values = np.asarray(distances)
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("Expected a non-empty finite distance vector")
    return {"mean": float(np.mean(values)), "median": float(np.median(values)),
            "std": float(np.std(values)), "p95": float(np.percentile(values, 95)),
            "max": float(np.max(values))}


def comparison_row(name, distances, total, seconds, baseline, neighbors=None, ratio=None):
    stats = distance_summary(distances)
    row = {"method": name, "nb_neighbors": neighbors, "std_ratio": ratio,
           "kept": len(distances), "removed": total - len(distances),
           "removed_percent": 100 * (total - len(distances)) / total,
           "filter_seconds": float(seconds), **stats}
    for key in ("mean", "p95", "std"):
        row[key + "_change_percent"] = (
            100 * (stats[key] / baseline[key] - 1) if baseline[key] > 0 else None
        )
    return row


def save_table(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def as_cpu_array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def restore_prediction(prediction, inverse_rotation, center, radius):
    prediction = as_cpu_array(prediction).reshape(-1)
    if prediction.shape != (3,) or not np.isfinite(prediction).all():
        raise ValueError("Pointfilter prediction must contain three finite coordinates")
    restored = as_cpu_array(inverse_rotation) @ (prediction * radius) + as_cpu_array(center)
    if restored.shape != (3,) or not np.isfinite(restored).all():
        raise ValueError("Invalid restored point")
    return restored


def pointfilter_worker(args):
    repo = args.pointfilter_repo.expanduser().resolve()
    source = args.input_path.expanduser().resolve()
    destination = args.output_path.expanduser().resolve()
    weights = repo / "Summary/pre_train_model/model_full_ae.pth"
    sys.path.insert(0, str(repo))
    import torch
    from Pointfilter_Network_Architecture import pointfilternet
    from Pointfilter_DataLoader import PointcloudPatchDataset

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the Pointfilter environment")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    points = np.load(source).copy()
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Pointfilter requires finite Nx3 input")
    original_hash = file_hash(source)
    started = perf_counter()
    checkpoint = torch.load(str(weights), map_location="cpu", weights_only=True)
    model = pointfilternet()
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model = model.to("cuda").eval()
    del checkpoint
    torch.cuda.synchronize()
    model_load_seconds = perf_counter() - started

    started = perf_counter()
    dataset = PointcloudPatchDataset(
        root=str(source.parent), shape_name=source.stem, train_state="evaluation",
        patch_radius=0.05, points_per_patch=500, seed=args.seed
    )
    dataset_seconds = perf_counter() - started
    if len(dataset) != len(points):
        raise ValueError("Dataset point order/length does not match the input")
    radius = float(dataset.patch_radius_absolute[0])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Invalid Pointfilter patch radius")
    print("Pointfilter: {} points on {}".format(len(points), torch.cuda.get_device_name(0)), flush=True)
    skipped = 0
    started = perf_counter()
    for index in range(len(points)):
        sample = dataset[index]
        if sample is None:
            skipped += 1
        else:
            patch, inverse_rotation, center = sample
            batch = patch.T.unsqueeze(0).contiguous().float().to("cuda")
            with torch.inference_mode():
                prediction = model(batch)
            points[index] = restore_prediction(
                prediction.cpu().squeeze(0), inverse_rotation, center, radius
            )
        if (index + 1) % 1000 == 0 or index + 1 == len(points):
            print("Pointfilter: {}/{}".format(index + 1, len(points)), flush=True)
    torch.cuda.synchronize()
    inference_seconds = perf_counter() - started
    if file_hash(source) != original_hash:
        raise RuntimeError("Pointfilter loader modified its input file")
    np.save(destination, points)
    result = {
        "input_sha256": original_hash, "output_sha256": file_hash(destination),
        "points": len(points), "skipped_kept_original": skipped, "seed": args.seed,
        "patch_radius": 0.05, "patch_radius_absolute": radius, "points_per_patch": 500,
        "model_load_seconds": model_load_seconds, "dataset_seconds": dataset_seconds,
        "inference_loop_seconds": inference_seconds,
        "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0), "weights_sha256": file_hash(weights),
        "network_source_sha256": file_hash(repo / "Pointfilter_Network_Architecture.py"),
        "loader_source_sha256": file_hash(repo / "Pointfilter_DataLoader.py"),
    }
    args.worker_report.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Pointfilter skipped (kept original):", skipped, flush=True)
    print("Pointfilter saved:", destination, flush=True)


def run_host(args):
    import open3d as o3d
    project = args.project.expanduser().resolve()
    alignment = args.alignment_folder.expanduser().resolve() if args.alignment_folder else latest_alignment(project)
    previous = json.loads((alignment / "metrics.json").read_text(encoding="utf-8"))
    mesh_path = project / "shoe.stl"
    if file_hash(mesh_path) != previous["mesh_sha256"]:
        raise ValueError("CAD mesh changed since full-scan validation")
    source = np.load(alignment / "aligned_scan.npy")
    if source.ndim != 2 or source.shape[1] != 3 or not np.isfinite(source).all():
        raise ValueError("Invalid aligned scan")
    if len(source) != previous["source_point_count"]:
        raise ValueError("Full-scan point count does not match the validation")
    repo = (args.pointfilter_repo or project / "Pointfilter").expanduser().resolve()
    ai_python = (args.pointfilter_python or project / ".venv-ai/Scripts/python.exe").expanduser().absolute()
    if not args.sor_only:
        for needed in (ai_python, repo / "Pointfilter_DataLoader.py",
                       repo / "Pointfilter_Network_Architecture.py",
                       repo / "Summary/pre_train_model/model_full_ae.pth"):
            if not needed.is_file():
                raise FileNotFoundError("Pointfilter dependency is missing: " + str(needed))
    params_path = project / "filter_params.json"
    experiments = json.loads(params_path.read_text(encoding="utf-8")) if params_path.exists() else [
        {"nb_neighbors": 20, "std_ratio": 2.0}
    ]
    if not isinstance(experiments, list) or not experiments:
        raise ValueError("filter_params.json must contain a non-empty list")
    for params in experiments:
        k, ratio = params["nb_neighbors"], params["std_ratio"]
        if int(k) != k or not 1 < k < len(source) or not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("Invalid SOR parameters: " + str(params))

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError("CAD mesh did not load")
    scene = o3d.t.geometry.RaycastingScene(nthreads=0)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    def distances(points):
        return scene.compute_distance(o3d.core.Tensor(
            np.asarray(points, dtype=np.float32), dtype=o3d.core.Dtype.Float32
        ), nthreads=0).numpy()

    raw_distances = distances(source)
    raw_stats = distance_summary(raw_distances)
    if not np.allclose([raw_stats["mean"], raw_stats["p95"]],
                       [previous["selected_pipeline"]["mean"], previous["selected_pipeline"]["p95"]],
                       rtol=1e-5, atol=1e-9):
        raise ValueError("Aligned cloud no longer matches its validated distances")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = alignment / ("filters_" + stamp)
    output.mkdir()
    # Vendor-created dataset caches are confined to this NEW filtering run.
    worker_input = output / "scan_for_ai.npy"
    np.save(worker_input, source.astype(np.float32))
    np.save(output / "distances_raw.npy", raw_distances)
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(source)
    rows = [comparison_row("Raw aligned", raw_distances, len(source), 0, raw_stats)]
    print("Using aligned scan:", alignment, flush=True)
    print("Raw aligned: mean={:.10g}, p95={:.10g}".format(raw_stats["mean"], raw_stats["p95"]), flush=True)
    for index, params in enumerate(experiments, 1):
        k, ratio = int(params["nb_neighbors"]), float(params["std_ratio"])
        started = perf_counter()
        _, kept = cloud.remove_statistical_outlier(nb_neighbors=k, std_ratio=ratio)
        seconds = perf_counter() - started
        kept = np.asarray(kept, dtype=np.int64)
        name = "SOR(k={}, std={:g})".format(k, ratio)
        row = comparison_row(name, raw_distances[kept], len(source), seconds, raw_stats, k, ratio)
        rows.append(row)
        np.savez(output / ("sor_{:02d}.npz".format(index)), points=source[kept],
                 kept_indices=kept, distances=raw_distances[kept])
        print("{}: kept={}, mean={:.10g}, p95={:.10g}".format(
            name, row["kept"], row["mean"], row["p95"]), flush=True)
    save_table(output / "filter_comparison.csv", rows)

    worker_report, shifts = None, None
    process_seconds = None
    if not args.sor_only:
        ai_output = output / "scan_pointfilter.npy"
        worker_report_path = output / "pointfilter_run.json"
        command = [str(ai_python), str(Path(__file__).resolve()), "--pointfilter-worker",
                   "--pointfilter-repo", str(repo), "--input-path", str(worker_input),
                   "--output-path", str(ai_output), "--worker-report", str(worker_report_path),
                   "--seed", str(args.seed)]
        started = perf_counter()
        subprocess.run(command, check=True, cwd=str(repo))
        process_seconds = perf_counter() - started
        filtered = np.load(ai_output)
        if filtered.shape != source.shape or not np.isfinite(filtered).all():
            raise ValueError("Pointfilter returned invalid point correspondence/coordinates")
        worker_report = json.loads(worker_report_path.read_text(encoding="utf-8"))
        if worker_report["input_sha256"] != file_hash(worker_input):
            raise ValueError("Pointfilter input does not match this filtering run")
        ai_distances = distances(filtered)
        np.save(output / "distances_pointfilter.npy", ai_distances)
        row = comparison_row("Pointfilter", ai_distances, len(source),
                             worker_report["inference_loop_seconds"], raw_stats)
        rows.append(row)
        shifts_array = np.linalg.norm(filtered.astype(np.float64) - source, axis=1)
        shifts = {"mean": float(np.mean(shifts_array)), "p95": float(np.percentile(shifts_array, 95)),
                  "max": float(np.max(shifts_array))}
        np.save(output / "point_shifts.npy", shifts_array)
        print("Pointfilter: mean={:.10g}, p95={:.10g}".format(row["mean"], row["p95"]), flush=True)
        save_table(output / "filter_comparison.csv", rows)

    result = {
        "alignment_folder": str(alignment), "alignment_metrics_sha256": file_hash(alignment / "metrics.json"),
        "aligned_scan_sha256": file_hash(alignment / "aligned_scan.npy"),
        "mesh_sha256": file_hash(mesh_path), "open3d": o3d.__version__, "numpy": np.__version__,
        "python": platform.python_version(), "registration_frozen_for_all_branches": True,
        "branches_start_from_the_same_raw_aligned_cloud": True, "seed": args.seed,
        "metric": "unsigned point-to-triangle-surface distance",
        "units": "original CAD coordinate units; physical unit not verified",
        "percent_changes": "(filtered/raw - 1)*100; negative means a smaller distance statistic",
        "sor_interpretation": "statistics cover retained points; removed counts must also be considered",
        "pointfilter_interpretation": "same point count, coordinates change; no post-filter registration",
        "pointfilter_process_seconds": process_seconds, "pointfilter_run": worker_report,
        "pointfilter_displacements": shifts, "results": rows,
        "limitation": "lower CAD distances on a real scan alone do not prove noise removal or defect preservation",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved:", output / "filter_comparison.csv", flush=True)
    if args.show and not args.sor_only:
        before, after = o3d.geometry.PointCloud(), o3d.geometry.PointCloud()
        before.points = o3d.utility.Vector3dVector(source)
        after.points = o3d.utility.Vector3dVector(filtered)
        before.paint_uniform_color([0.1, 0.4, 1])
        after.paint_uniform_color([1, 0, 0.7])
        o3d.visualization.draw_geometries([before, after], window_name="Pointfilter: blue BEFORE, magenta AFTER")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("C:/Users/mihei/Desktop/vkr"))
    parser.add_argument("--alignment-folder", type=Path)
    parser.add_argument("--pointfilter-repo", type=Path)
    parser.add_argument("--pointfilter-python", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sor-only", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--pointfilter-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--output-path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-report", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.pointfilter_worker:
        for key in ("pointfilter_repo", "input_path", "output_path", "worker_report"):
            if getattr(args, key) is None:
                parser.error("Pointfilter worker requires --" + key.replace("_", "-"))
        pointfilter_worker(args)
    else:
        run_host(args)


if __name__ == "__main__":
    main()
