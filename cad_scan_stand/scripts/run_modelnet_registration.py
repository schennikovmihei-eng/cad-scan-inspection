#!/usr/bin/env python3
"""Run the official ModelNet GeoTransformer preset on the prepared CAD/scan pair.

Use inside the Ubuntu Conda environment `geotransformer`. The original model,
stand.py, old input arrays and earlier results are preserved. This runner creates
a reviewable inference-only model copy and a separate directory for each run.
Point-sample distances printed here are diagnostics, not distances to CAD faces.
"""

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from textwrap import indent
from time import perf_counter

import numpy as np


EXPERIMENT = "geotransformer.modelnet.rpmnet.stage4.gse.k3.max.oacl.stage2.sinkhorn"


def apply_transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patch_inference_source(source):
    """Keep the native model architecture; calculate GT targets only in training."""
    assignment = "        transform = data_dict['transform'].detach()\n"
    if source.count(assignment) != 1:
        raise ValueError("Unexpected model.py: GT assignment does not match")
    source = source.replace(assignment, "", 1)
    begin = source.index(
        "        gt_node_corr_indices, gt_node_corr_overlaps = get_node_correspondences("
    )
    marker = "        output_dict['gt_node_corr_overlaps'] = gt_node_corr_overlaps"
    end = source.index(marker, begin) + len(marker)
    source = (
        source[:begin]
        + "        if self.training:\n"
        + "            " + assignment.strip() + "\n"
        + indent(source[begin:end], "    ")
        + source[end:]
    )
    compile(source, "model_for_stand.py", "exec")
    return source


def prepare_pair(src_base, ref_base, scan_scale, previous_scale, normalize_fn, count, seed):
    """Return network inputs and exact maps from original scan/CAD coordinates."""
    for name, points in (("src", src_base), ("ref", ref_base)):
        if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
            raise ValueError("Invalid finite Nx3 array: " + name)
    if scan_scale <= 0 or previous_scale <= 0:
        raise ValueError("Preparation scales must be positive")
    if not 0 < count <= min(len(src_base), len(ref_base)):
        raise ValueError("Point count exceeds available input points")

    scaled_scan = np.asarray(src_base, dtype=np.float64) / previous_scale
    cad = np.asarray(ref_base, dtype=np.float64) / previous_scale
    normalized_cad = np.asarray(normalize_fn(cad.copy()), dtype=np.float64)

    # Recover the affine map from the repository's actual normalize_points().
    # This checks its behavior instead of assuming a normalization convention.
    cad_center = cad.mean(axis=0)
    normalized_center = normalized_cad.mean(axis=0)
    centered = cad - cad_center
    denominator = float(np.sum(centered * centered))
    if denominator <= 0:
        raise ValueError("CAD point sample has zero extent")
    scale = float(np.sum(centered * (normalized_cad - normalized_center)) / denominator)
    offset = normalized_center - scale * cad_center
    if not np.isfinite(scale) or scale <= 0 or not np.allclose(
        normalized_cad, cad * scale + offset, atol=1e-6, rtol=1e-6
    ):
        raise ValueError("normalize_points is not a positive uniform scale plus translation")

    reference_pre_transform = np.eye(4)
    reference_pre_transform[:3, :3] *= scale
    reference_pre_transform[:3, 3] = offset
    source_pre_transform = np.eye(4)
    source_pre_transform[:3, :3] *= scale * scan_scale
    source_pre_transform[:3, 3] = normalized_center - scale * scaled_scan.mean(axis=0)

    original_scan = scaled_scan / scan_scale
    normalized_scan = apply_transform(original_scan, source_pre_transform)
    rng = np.random.default_rng(seed)
    src_indices = rng.choice(len(src_base), count, replace=False)
    ref_indices = rng.choice(len(ref_base), count, replace=False)
    return {
        "src_points": normalized_scan[src_indices].astype(np.float32),
        "ref_points": normalized_cad[ref_indices].astype(np.float32),
        "source_pre_transform": source_pre_transform,
        "reference_pre_transform": reference_pre_transform,
        "source_indices_in_base": src_indices,
        "reference_indices_in_base": ref_indices,
        "original_scan": original_scan,
        "cad": cad,
        "normalization_scale": scale,
    }


def diagnostic_comparison(inputs, pair, raw_transform, scan_scale, previous_scale):
    from scipy.spatial import cKDTree

    tree = cKDTree(pair["cad"])
    original_scan = pair["original_scan"]
    candidates = {"ModelNet": apply_transform(original_scan, raw_transform)}
    previous = inputs / "geotransformer_transform.npy"
    if previous.exists():
        src_base = original_scan * scan_scale * previous_scale
        candidates["3DMatch"] = apply_transform(src_base, np.load(previous)) / previous_scale
    alignment = inputs.parent / "aligment.npz"
    if alignment.exists():
        with np.load(alignment) as saved:
            points = apply_transform(
                original_scan - saved["manual_c_ply"], saved["manual_transform"]
            )
            candidates["Manual + ICP"] = (
                apply_transform(points, saved["icp_transform"]) + saved["manual_c_s"]
            )
    metrics = {}
    for name, points in candidates.items():
        distances = tree.query(points)[0]
        metrics[name] = {
            "count": len(distances),
            "mean": float(np.mean(distances)),
            "p95": float(np.percentile(distances, 95)),
            "max": float(np.max(distances)),
        }
        print("{}: mean={:.10g}, p95={:.10g}".format(
            name, metrics[name]["mean"], metrics[name]["p95"]
        ), flush=True)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.home() / "GeoTransformer")
    parser.add_argument("--inputs", type=Path,
                        default=Path("/mnt/c/Users/mihei/Desktop/vkr/registration_input"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--points", type=int, help="Default: cfg.data.num_points (717)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--show", action="store_true", help="Show only the final alignment")
    args = parser.parse_args()
    repo, inputs = args.repo.expanduser().resolve(), args.inputs.expanduser().resolve()
    experiment = repo / "experiments" / EXPERIMENT
    weights = (args.weights or repo / "weights/geotransformer-modelnet.pth.tar").resolve()
    required = [experiment / "model.py", experiment / "config.py", weights,
                inputs / "src.npy", inputs / "ref.npy", inputs / "preparation.npz"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError("Required file is missing: " + str(path))

    model_source = (experiment / "model.py").read_text(encoding="utf-8")
    patched = patch_inference_source(model_source)
    adapter = experiment / "model_for_stand.py"
    if adapter.exists() and adapter.read_text(encoding="utf-8") != patched:
        raise RuntimeError("Existing adapter differs; please inspect it: " + str(adapter))
    if not adapter.exists():
        adapter.write_text(patched, encoding="utf-8")
    sys.path.insert(0, str(experiment))
    sys.path.insert(1, str(repo))

    import torch
    from config import make_cfg
    from model_for_stand import create_model
    from geotransformer.transforms.functional import normalize_points
    from geotransformer.utils.data import registration_collate_fn_stack_mode
    from geotransformer.utils.torch import to_cuda, release_cuda

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this Python environment")
    cfg = make_cfg()
    count = args.points if args.points is not None else int(cfg.data.num_points)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    started = perf_counter()
    src_base, ref_base = np.load(inputs / "src.npy"), np.load(inputs / "ref.npy")
    with np.load(inputs / "preparation.npz") as saved:
        scan_scale = float(saved["scan_to_cad_scale"])
        previous_scale = float(saved["network_scale"])
    pair = prepare_pair(src_base, ref_base, scan_scale, previous_scale,
                        normalize_points, count, args.seed)
    preparation_seconds = perf_counter() - started

    # For this small pair, use an upper bound that does not truncate neighbors.
    # There are at most 2*count input points across the two clouds.
    neighbor_limits = [2 * count] * int(cfg.backbone.num_stages)
    data = {key: pair[key] for key in ("ref_points", "src_points")}
    data["ref_feats"] = np.ones((count, 1), dtype=np.float32)
    data["src_feats"] = np.ones((count, 1), dtype=np.float32)
    started = perf_counter()
    data = registration_collate_fn_stack_mode(
        [data], cfg.backbone.num_stages, cfg.backbone.init_voxel_size,
        cfg.backbone.init_radius, neighbor_limits
    )
    collation_seconds = perf_counter() - started
    print("ModelNet input: {} points per cloud; {} stages".format(
        count, cfg.backbone.num_stages), flush=True)

    torch.cuda.reset_peak_memory_stats()
    started = perf_counter()
    checkpoint = torch.load(str(weights), map_location="cpu")
    model = create_model(cfg)
    model.load_state_dict(checkpoint["model"], strict=True)
    model = model.cuda().eval()
    del checkpoint
    torch.cuda.synchronize()
    model_load_seconds = perf_counter() - started
    print("Weights loaded", flush=True)

    torch.cuda.synchronize()
    started = perf_counter()
    data = to_cuda(data)
    with torch.inference_mode():
        output = model(data)
    torch.cuda.synchronize()
    inference_seconds = perf_counter() - started
    network_transform = np.asarray(
        release_cuda(output["estimated_transform"]), dtype=np.float64
    )
    if network_transform.shape != (4, 4) or not np.isfinite(network_transform).all():
        raise ValueError("Model returned an invalid 4x4 transform")
    if not np.allclose(network_transform[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Model transform is not homogeneous")
    raw_transform = (
        np.linalg.inv(pair["reference_pre_transform"])
        @ network_transform @ pair["source_pre_transform"]
    )

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    result_dir = inputs / ("modelnet_" + stamp)
    result_dir.mkdir()
    np.save(result_dir / "transform_network.npy", network_transform)
    np.save(result_dir / "transform_scan_to_cad.npy", raw_transform)
    np.save(result_dir / "src.npy", pair["src_points"])
    np.save(result_dir / "ref.npy", pair["ref_points"])
    np.savez(result_dir / "preparation.npz", **{
        key: pair[key] for key in (
            "source_pre_transform", "reference_pre_transform", "source_indices_in_base",
            "reference_indices_in_base", "normalization_scale"
        )
    }, scan_to_cad_scale=scan_scale, seed=args.seed)
    print("Saved:", result_dir, flush=True)
    print("Transform maps ORIGINAL full scan coordinates to ORIGINAL CAD coordinates.", flush=True)

    metrics = diagnostic_comparison(inputs, pair, raw_transform, scan_scale, previous_scale)
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=False).stdout.strip()
    report = {
        "preset": EXPERIMENT, "git_commit": commit, "seed": args.seed,
        "points_per_cloud": count, "base_evaluation_point_count": len(src_base),
        "gt_supplied": False, "scale_origin": "previous_manual_landmark_calibration",
        "normalization": "native normalize_points on CAD; shared scale and separate centering",
        "normalization_scale": pair["normalization_scale"],
        "scan_to_cad_scale": scan_scale, "num_stages": int(cfg.backbone.num_stages),
        "init_voxel_size": float(cfg.backbone.init_voxel_size),
        "init_radius": float(cfg.backbone.init_radius), "neighbor_limits": neighbor_limits,
        "weights_sha256": file_hash(weights),
        "configuration": dict(cfg),
        "config_sha256": file_hash(experiment / "config.py"),
        "backbone_sha256": file_hash(experiment / "backbone.py"),
        "native_model_sha256": hashlib.sha256(model_source.encode("utf-8")).hexdigest(),
        "adapter_sha256": hashlib.sha256(patched.encode("utf-8")).hexdigest(),
        "base_src_sha256": file_hash(inputs / "src.npy"),
        "base_ref_sha256": file_hash(inputs / "ref.npy"),
        "normalization_source_sha256": file_hash(repo / "geotransformer/transforms/functional.py"),
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "preparation_seconds": preparation_seconds, "collation_seconds": collation_seconds,
        "model_load_seconds": model_load_seconds,
        "inference_and_host_to_device_seconds": inference_seconds,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "metric": "one-way nearest distance to the SAME fixed input CAD point sample",
        "cad_evaluation_sample_count": len(ref_base),
        "units": "original CAD coordinate units; physical unit not verified",
        "metrics": metrics,
    }
    (result_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if args.show:
        import open3d as o3d
        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(pair["cad"])
        target.paint_uniform_color([1, 0.8, 0])
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(apply_transform(pair["original_scan"], raw_transform))
        source.paint_uniform_color([0.1, 0.4, 1])
        o3d.visualization.draw_geometries([target, source], window_name="ModelNet: AFTER registration")


if __name__ == "__main__":
    main()
