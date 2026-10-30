#!/usr/bin/env python3
"""Validate the selected registration on the full scan and CAD triangle surface.

Run with the Windows Python environment used by stand.py, which can read the
ASCII STL and provides Open3D RaycastingScene. Neural inference is not repeated.
Every evaluation is saved in a new directory under its orientation search run.
"""

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import numpy as np


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def apply_transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def translation(vector):
    matrix = np.eye(4)
    matrix[:3, 3] = vector
    return matrix


def manual_transform(saved):
    return (translation(saved["manual_c_s"]) @ saved["icp_transform"]
            @ saved["manual_transform"] @ translation(-saved["manual_c_ply"]))


def latest_selection(inputs):
    paths = sorted(path for path in inputs.glob("modelnet_*/icp_*/orientations_*")
                   if path.is_dir() and (path / "metrics.json").is_file()
                   and (path / "transform_scan_to_cad.npy").is_file())
    if not paths:
        raise FileNotFoundError("No completed orientation search found: " + str(inputs))
    return paths[-1]


def summary(distances):
    distances = np.asarray(distances)
    if distances.ndim != 1 or len(distances) == 0 or not np.isfinite(distances).all():
        raise ValueError("Invalid distance array")
    return {"count": len(distances), "mean": float(np.mean(distances)),
            "median": float(np.median(distances)), "std": float(np.std(distances)),
            "p95": float(np.percentile(distances, 95)), "max": float(np.max(distances))}


def mesh_distances(scene, points, o3d):
    query = o3d.core.Tensor(np.asarray(points, dtype=np.float32), dtype=o3d.core.Dtype.Float32)
    return scene.compute_distance(query, nthreads=0).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path("C:/Users/mihei/Desktop/vkr"))
    parser.add_argument("--selection", type=Path, help="Default: latest completed orientation search")
    parser.add_argument("--show", action="store_true", help="Show full aligned scan against CAD mesh")
    args = parser.parse_args()
    project = args.project.expanduser().resolve()
    inputs = project / "registration_input"
    selected = args.selection.expanduser().resolve() if args.selection else latest_selection(inputs)
    report = json.loads((selected / "metrics.json").read_text(encoding="utf-8"))
    for name in ("src", "ref"):
        if file_hash(inputs / (name + ".npy")) != report["base_" + name + "_sha256"]:
            raise ValueError("Prepared input changed: " + name)
    matrix = np.load(selected / "transform_scan_to_cad.npy")
    with np.load(selected / "candidates.npz") as saved:
        expected = saved["transforms"][int(report["best_candidate_zero_based"])]
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix, expected, rtol=1e-12, atol=1e-12)):
        raise ValueError("Selected transform does not match the saved winning candidate")
    with np.load(inputs / "preparation.npz") as saved:
        scan_scale = float(saved["scan_to_cad_scale"])
        network_scale = float(saved["network_scale"])
        indices = saved["source_indices"]

    import open3d as o3d
    print("Open3D:", o3d.__version__, flush=True)
    mesh_path, scan_path = project / "shoe.stl", project / "shoe_clean.ply"
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    scan = o3d.io.read_point_cloud(str(scan_path))
    if not mesh.has_triangles() or scan.is_empty():
        raise ValueError("CAD mesh or scan did not load")
    points = np.asarray(scan.points)
    if not np.isfinite(points).all():
        raise ValueError("Full scan contains non-finite points")
    prepared_source = np.load(inputs / "src.npy")
    if not np.allclose(points[indices] * scan_scale * network_scale,
                       prepared_source, rtol=1e-6, atol=1e-6):
        raise ValueError("Current PLY does not match the scan used for neural registration")
    started = perf_counter()
    scene = o3d.t.geometry.RaycastingScene(nthreads=0)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    scene_seconds = perf_counter() - started
    reference_sample = np.load(inputs / "ref.npy").astype(np.float64) / network_scale
    reference_distances = mesh_distances(scene, reference_sample, o3d)
    diagonal = float(np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()))
    if float(reference_distances.max()) > max(1e-7, 1e-5 * diagonal):
        raise ValueError("Current CAD mesh does not match the prepared reference sample")

    with np.load(project / "aligment.npz") as saved:
        control_matrix = manual_transform(saved)
    aligned = apply_transform(points, matrix)
    manual_aligned = apply_transform(points, control_matrix)
    started = perf_counter()
    distances = mesh_distances(scene, aligned, o3d)
    neural_seconds = perf_counter() - started
    started = perf_counter()
    manual_distances = mesh_distances(scene, manual_aligned, o3d)
    manual_seconds = perf_counter() - started
    neural_metrics, manual_metrics = summary(distances), summary(manual_distances)
    print("Full scan points:", len(points), flush=True)
    for name, metrics in (("Selected pipeline (mesh)", neural_metrics),
                          ("Manual + ICP (mesh)", manual_metrics)):
        print("{}: mean={:.10g}, p95={:.10g}".format(
            name, metrics["mean"], metrics["p95"]), flush=True)
    changes = {key: 100 * (neural_metrics[key] / manual_metrics[key] - 1)
               if manual_metrics[key] > 0 else None
               for key in ("mean", "median", "std", "p95", "max")}

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = selected / ("full_scan_" + stamp)
    output.mkdir()
    np.save(output / "aligned_scan.npy", aligned.astype(np.float32))
    np.save(output / "manual_control_aligned_scan.npy", manual_aligned.astype(np.float32))
    np.save(output / "transform_scan_to_cad.npy", matrix)
    np.save(output / "manual_control_transform.npy", control_matrix)
    np.save(output / "distances.npy", distances)
    np.save(output / "manual_control_distances.npy", manual_distances)
    np.savetxt(output / "deviations.csv", np.column_stack((aligned, distances)),
               delimiter=",", header="x,y,z,distance_to_cad_mesh", comments="")
    result = {
        "selected_orientation_run": str(selected),
        "selected_metrics_sha256": file_hash(selected / "metrics.json"),
        "mesh_sha256": file_hash(mesh_path), "scan_sha256": file_hash(scan_path),
        "manual_cache_sha256": file_hash(project / "aligment.npz"),
        "open3d": o3d.__version__, "numpy": np.__version__, "python": platform.python_version(),
        "source_point_count": len(points), "mesh_triangles": len(mesh.triangles),
        "metric": "unsigned point-to-TRIANGLE-SURFACE distance for every original scan point",
        "units": "original CAD coordinate units; physical unit not verified",
        "method": "GeoTransformer ModelNet + ICP + 24 geometric orientation hypotheses with ICP",
        "scale_origin": "previous manual landmark calibration",
        "selected_pipeline": neural_metrics, "manual_icp_control": manual_metrics,
        "change_percent_vs_manual": changes,
        "scene_build_seconds": scene_seconds,
        "distance_query_seconds_selected": neural_seconds,
        "distance_query_seconds_manual": manual_seconds,
        "reference_sample_mesh_distance_max": float(reference_distances.max()),
        "orientation_status": "requires visual verification; distance metrics are not a pose certificate",
        "aligned_scan_role": "candidate input for a NEW denoising run; old denoising results belong to their old alignment",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("Saved:", output, flush=True)
    if args.show:
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([1, 0.8, 0])
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(aligned)
        cloud.paint_uniform_color([0.1, 0.4, 1])
        o3d.visualization.draw_geometries(
            [mesh, cloud], window_name="Selected registration: FULL scan and CAD triangle mesh"
        )


if __name__ == "__main__":
    main()
