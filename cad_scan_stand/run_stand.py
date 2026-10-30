#!/usr/bin/env python3
"""One Windows entry point for the existing three-environment CAD/scan stand."""

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path, PureWindowsPath
from time import perf_counter


SCRIPTS = Path(__file__).resolve().parent / "scripts"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def project_path(project, value):
    path = Path(value).expanduser()
    return path.absolute() if path.is_absolute() else (project / path).absolute()


def wsl_path(path):
    path = PureWindowsPath(str(path))
    if len(path.drive) != 2 or path.drive[1] != ":" or not path.is_absolute():
        raise ValueError("A local Windows drive path is required: " + str(path))
    return "/mnt/" + path.drive[0].lower() + "/" + "/".join(path.parts[1:])


def wsl_command(cfg, arguments):
    linux = cfg["wsl"]
    return ["wsl.exe", "--distribution", linux["distribution"], "--user", linux["user"],
            "--exec", linux["conda"], "run", "--no-capture-output", "--cwd", linux["repo"],
            "-n", linux["environment"], "--", "python", "-u"] + list(arguments)


def child_environment():
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"
    # Prevent a parent Python/Python venv from injecting its modules into children.
    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    return env


def stream_command(command, log_path):
    with Path(log_path).open("w", encoding="utf-8") as log:
        log.write(json.dumps(command, ensure_ascii=False) + "\n")
        log.flush()
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace", bufsize=1,
                                   env=child_environment())
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    if code:
        raise subprocess.CalledProcessError(code, command)


def one_result(parent, pattern, required):
    candidates = [path for path in Path(parent).glob(pattern)
                  if all((path / name).is_file() for name in required)]
    if len(candidates) != 1:
        raise RuntimeError("Expected one new result in {} ({}), found {}".format(
            parent, pattern, len(candidates)))
    return candidates[0]


def direct_file_hashes(folder):
    return {str(path): sha256(path) for path in Path(folder).iterdir() if path.is_file()}


def preflight(cfg, project, output):
    """Check the actual existing environments; do not install or change them."""
    windows_python = project_path(project, cfg["windows_python"])
    ai_python = project_path(project, cfg["pointfilter_python"])
    repo = project_path(project, cfg["pointfilter_repo"])
    required = [windows_python, ai_python, project_path(project, cfg["mesh"]),
                project_path(project, cfg["scan"]), project_path(project, cfg["manual_calibration"]),
                repo / "Summary/pre_train_model/model_full_ae.pth",
                repo / "Pointfilter_DataLoader.py", repo / "Pointfilter_Network_Architecture.py"]
    if cfg["preparation"]["mode"] == "reuse_fixed_sample":
        base = project_path(project, cfg["prepared_inputs"])
        required += [base / name for name in ("src.npy", "ref.npy", "preparation.npz")]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(str(path))
    common = ("import sys,json,platform,subprocess; import numpy as np; "
              "r={'python':platform.python_version(),'executable':sys.executable,'numpy':np.__version__}; ")
    freeze = ("r['pip_freeze']=subprocess.check_output([sys.executable,'-m','pip','freeze'],"
              "text=True).splitlines(); print('STAND_PROBE:'+json.dumps(r))")
    host = common + ("import open3d as o; assert hasattr(o.t.geometry,'RaycastingScene'); "
                     "r['open3d']=o.__version__; ") + freeze
    gpu = common + ("import torch; assert torch.cuda.is_available(),'CUDA unavailable'; "
                    "assert torch.isfinite(torch.ones(1,device='cuda')).all(); "
                    "r.update(torch=torch.__version__,cuda=torch.version.cuda,"
                    "gpu=torch.cuda.get_device_name(0)); ")
    neural = gpu + ("import open3d; import geotransformer.ext; from pathlib import Path; "
                    "assert Path('weights/geotransformer-modelnet.pth.tar').is_file(); "
                    "r['open3d']=open3d.__version__; "
                    "r['git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(); ") + freeze
    commands = {"windows_open3d": [str(windows_python), "-c", host],
                "windows_pointfilter": [str(ai_python), "-c", gpu + freeze],
                "wsl_geotransformer": wsl_command(cfg, ["-c", neural])}
    probes = {}
    for name, command in commands.items():
        print("Checking environment:", name, flush=True)
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   env=child_environment(), timeout=300)
        (output / (name + ".txt")).write_text(completed.stdout, encoding="utf-8")
        if completed.returncode:
            raise RuntimeError("Environment check failed: {}. See {}".format(name, output / (name + ".txt")))
        lines = [line[len("STAND_PROBE:"):] for line in completed.stdout.splitlines()
                 if line.startswith("STAND_PROBE:")]
        if len(lines) != 1:
            raise ValueError("Environment check returned no version record: " + name)
        probes[name] = json.loads(lines[0])
    if probes["wsl_geotransformer"]["git_commit"] != cfg["wsl"]["expected_commit"]:
        raise ValueError("GeoTransformer commit differs from the tested version in config.json")
    write_json(output / "environments.json", probes)


def prepare(cfg, project, workspace):
    import numpy as np
    import open3d as o3d
    workspace.mkdir()
    inputs = workspace / "registration_input"
    inputs.mkdir()
    origins = {}
    for key, target in (("mesh", "shoe.stl"), ("scan", "shoe_clean.ply"),
                        ("manual_calibration", "aligment.npz")):
        source = project_path(project, cfg[key])
        shutil.copy2(source, workspace / target)
        origins[key] = {"original": str(source), "sha256": sha256(source)}
    write_json(workspace / "filter_params.json", cfg["sor"])
    with np.load(workspace / "aligment.npz") as saved:
        matrix = saved["manual_transform"]
        for key in ("manual_c_ply", "manual_c_s", "icp_transform"):
            if not np.isfinite(saved[key]).all():
                raise ValueError("Invalid manual control cache")
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Invalid similarity calibration")
    singular = np.linalg.svd(matrix[:3, :3], compute_uv=False)
    scale = float(singular.mean())
    if scale <= 0 or not np.allclose(singular, scale, rtol=1e-5, atol=0):
        raise ValueError("Manual calibration must contain uniform scale")
    mesh = o3d.io.read_triangle_mesh(str(workspace / "shoe.stl"))
    scan = o3d.io.read_point_cloud(str(workspace / "shoe_clean.ply"))
    if not mesh.has_triangles() or scan.is_empty():
        raise ValueError("Input geometry is empty")
    points = np.asarray(scan.points)
    if not np.isfinite(points).all():
        raise ValueError("Non-finite scan coordinates")
    for params in cfg["sor"]:
        k, ratio = params["nb_neighbors"], params["std_ratio"]
        if int(k) != k or not 1 < k < len(points) or not np.isfinite(ratio) or ratio <= 0:
            raise ValueError("Invalid SOR parameters: " + str(params))
    mode = cfg["preparation"]["mode"]
    if mode == "reuse_fixed_sample":
        existing = project_path(project, cfg["prepared_inputs"])
        for name in ("src.npy", "ref.npy", "preparation.npz"):
            shutil.copy2(existing / name, inputs / name)
            origins[name] = {"original": str(existing / name), "sha256": sha256(existing / name)}
        with np.load(inputs / "preparation.npz") as saved:
            used_scale = float(saved["scan_to_cad_scale"])
            network_scale = float(saved["network_scale"])
            indices = saved["source_indices"]
        if not np.isclose(scale, used_scale, rtol=1e-8, atol=0):
            raise ValueError("Prepared scale differs from the manual calibration")
    elif mode == "resample":
        count = cfg["preparation"]["sample_points"]
        indices = np.random.default_rng(cfg["seed"]).choice(len(points), count, replace=False)
        diagonal = float(np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()))
        if not np.isfinite(diagonal) or diagonal <= 0:
            raise ValueError("Invalid CAD extent")
        network_scale = cfg["preparation"]["network_diagonal"] / diagonal
        o3d.utility.random.seed(cfg["seed"])
        reference = np.asarray(mesh.sample_points_uniformly(number_of_points=count).points)
        np.save(inputs / "src.npy", (points[indices] * scale * network_scale).astype(np.float32))
        np.save(inputs / "ref.npy", (reference * network_scale).astype(np.float32))
        np.savez(inputs / "preparation.npz", scan_to_cad_scale=scale, network_scale=network_scale,
                 source_indices=indices, scale_origin="manual_landmarks")
    else:
        raise ValueError("Unknown preparation mode: " + str(mode))
    source, reference = np.load(inputs / "src.npy"), np.load(inputs / "ref.npy")
    for array in (source, reference):
        if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
            raise ValueError("Prepared input must be finite Nx3")
        if len(array) < cfg["modelnet_points"]:
            raise ValueError("Prepared sample is smaller than ModelNet point count")
    if (not np.isfinite(network_scale) or network_scale <= 0
            or indices.dtype.kind not in "iu" or indices.shape != (len(source),)
            or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(points))):
        raise ValueError("Invalid prepared source indices or normalization scale")
    if not np.allclose(source, points[indices] * scale * network_scale, rtol=1e-6, atol=1e-6):
        raise ValueError("Prepared points do not match the original scan and scale")
    scene = o3d.t.geometry.RaycastingScene(nthreads=0)
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    distances = scene.compute_distance(o3d.core.Tensor(
        (reference / network_scale).astype(np.float32), dtype=o3d.core.Dtype.Float32)).numpy()
    diagonal = float(np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()))
    if float(np.max(distances)) > max(1e-7, 1e-5 * diagonal):
        raise ValueError("Prepared reference is not on this CAD mesh")
    write_json(inputs / "preparation_report.json", {
        "mode": mode, "scan_to_cad_scale": scale, "network_scale": network_scale,
        "scale_origin": "previous manual landmark calibration; rotation/translation excluded from neural input",
        "source_points": len(points), "sample_points": len(source), "origins": origins,
        "reference_surface_distance_max": float(np.max(distances))})
    print("Scale validated:", scale, "; preparation:", mode, flush=True)
    return inputs


def final_report(run, full, filters):
    geometry = read_json(full / "metrics.json")
    filtering = read_json(filters / "metrics.json")
    with (filters / "filter_comparison.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    shutil.copy2(filters / "filter_comparison.csv", run / "filter_comparison.csv")
    comparison = {"registration": geometry, "filtering": filtering,
                  "full_alignment_folder": str(full), "filtering_folder": str(filters)}
    write_json(run / "summary.json", comparison)
    before, selected = geometry["manual_icp_control"], geometry["selected_pipeline"]
    lines = ["# Результат прогона стенда", "", "## Совмещение", "",
             "| Метод | Среднее до CAD | P95 до CAD |", "|---|---:|---:|",
             "| Ручное + ICP (контроль) | {:.9g} | {:.9g} |".format(before["mean"], before["p95"]),
             "| GeoTransformer + ICP + поиск ориентации | {:.9g} | {:.9g} |".format(selected["mean"], selected["p95"]),
             "", "## Фильтрация при фиксированном совмещении", "",
             "| Метод | Точек | Удалено, % | Среднее | P95 | Изменение среднего, % | Изменение P95, % | Время фильтра, с |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        values = [row["method"], str(int(row["kept"]))]
        values += ["{:.6g}".format(float(row[key])) if row[key] else "—" for key in (
            "removed_percent", "mean", "p95", "mean_change_percent", "p95_change_percent", "filter_seconds")]
        lines.append("| " + " | ".join(values) + " |")
    lines += ["", "Расстояния без знака до треугольной поверхности, в исходных единицах CAD; физическая единица не подтверждена.",
              "Изменения рассчитаны относительно облака до фильтрации; отрицательный процент означает уменьшение показателя.",
              "SOR удаляет точки: его статистики относятся к оставшимся точкам. Pointfilter изменяет координаты с сохранением числа точек.",
              "Время SOR — вызов фильтра; время Pointfilter — полный цикл обработки патчей, включая подготовку и переносы в этом цикле. Загрузка модели и окружения учитывается отдельно в JSON.",
              "Меньшее расстояние само по себе не доказывает удаление шума или сохранение дефектов. Проверка ориентации выполняется визуально.",
              "Масштаб взят из сохранённой ручной калибровки. Ручная поза используется только как контроль; она не участвует в выборе кандидата.",
              "", "Полные статистики, настройки, версии, хеши, время этапов и матрицы сохранены в JSON и подпапках этого прогона.", ""]
    (run / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_pipeline(cfg, config_path, resume=None, checker=preflight, preparer=prepare, execute=stream_command):
    project = project_path(Path(config_path).parent, cfg["project_dir"])
    if resume:
        run = Path(resume).absolute()
        manifest = read_json(run / "run.json")
        if manifest["config"] != cfg or manifest["project"] != str(project):
            raise ValueError("Resume requires the same config and project path")
    else:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
        run = project_path(project, cfg["results_dir"]) / ("run_" + stamp)
        run.mkdir(parents=True)
        manifest = {"status": "running", "config": cfg, "project": str(project), "stages": {},
                    "launcher_python": platform.python_version(), "created_utc": stamp}
        write_json(run / "config.json", cfg)
    (run / "logs").mkdir(exist_ok=True)
    state_path = run / "run.json"
    scripts_hashes = {path.name: sha256(path) for path in SCRIPTS.glob("*.py")}
    scripts_hashes["run_stand.py"] = sha256(__file__)
    if resume and manifest["scripts_sha256"] != scripts_hashes:
        raise ValueError("Launcher or bundled scripts changed since the interrupted run")
    manifest["scripts_sha256"] = scripts_hashes
    manifest["status"] = "running"
    write_json(state_path, manifest)
    host_python = str(project_path(project, cfg["windows_python"]))
    workspace = run / "data"

    def stage(name, action):
        old = manifest["stages"].get(name)
        if old and old["status"] == "completed":
            if any(not Path(path).is_file() or sha256(path) != value
                   for path, value in old["artifact_hashes"].items()):
                raise ValueError("Saved stage output changed: " + name)
            print("Reusing completed stage:", name, flush=True)
            return Path(old["output"])
        print("\n=== " + name + " ===", flush=True)
        if old:
            # Keep interrupted work for inspection; all new stages get a fresh run.
            raise RuntimeError("Stage {} was interrupted. Start a new run; earlier results are preserved.".format(name))
        manifest["stages"][name] = {"status": "running"}
        write_json(state_path, manifest)
        started = perf_counter()
        output = Path(action())
        hashes = direct_file_hashes(output)
        if name == "01_prepare":
            hashes.update(direct_file_hashes(workspace))
        manifest["stages"][name] = {"status": "completed", "output": str(output),
                                    "seconds": perf_counter() - started,
                                    "artifact_hashes": hashes}
        write_json(state_path, manifest)
        return output

    def command_result(name, command, parent, pattern, files):
        execute(command, run / "logs" / (name + ".log"))
        return one_result(parent, pattern, files)

    try:
        checker(cfg, project, run / "logs")
        inputs = stage("01_prepare", lambda: preparer(cfg, project, workspace))
        modelnet = stage("02_neural_registration", lambda: command_result("02_neural_registration",
            wsl_command(cfg, [wsl_path(SCRIPTS / "run_modelnet_registration.py"), "--repo", cfg["wsl"]["repo"],
                             "--inputs", wsl_path(inputs), "--points", str(cfg["modelnet_points"]),
                             "--seed", str(cfg["seed"])]), inputs, "modelnet_*", ["metrics.json", "transform_scan_to_cad.npy"]))
        icp = stage("03_icp", lambda: command_result("03_icp",
            wsl_command(cfg, [wsl_path(SCRIPTS / "refine_modelnet_icp.py"), "--inputs", wsl_path(inputs),
                             "--run", wsl_path(modelnet), "--iterations", str(cfg["icp_iterations"])]),
            modelnet, "icp_*", ["metrics.json", "transform_scan_to_cad.npy"]))
        orientation = stage("04_orientation_search", lambda: command_result("04_orientation_search",
            wsl_command(cfg, [wsl_path(SCRIPTS / "try_modelnet_orientations.py"), "--inputs", wsl_path(inputs),
                             "--run", wsl_path(modelnet), "--icp-run", wsl_path(icp),
                             "--iterations", str(cfg["icp_iterations"])]),
            icp, "orientations_*", ["metrics.json", "candidates.npz", "transform_scan_to_cad.npy"]))
        full = stage("05_surface_evaluation", lambda: command_result("05_surface_evaluation",
            [host_python, "-u", str(SCRIPTS / "evaluate_full_registration.py"),
             "--project", str(workspace), "--selection", str(orientation)],
            orientation, "full_scan_*", ["metrics.json", "aligned_scan.npy"]))
        filters = stage("06_filtering", lambda: command_result("06_filtering",
            [host_python, "-u", str(SCRIPTS / "run_filters_on_alignment.py"), "--project", str(workspace),
             "--alignment-folder", str(full), "--pointfilter-repo", str(project_path(project, cfg["pointfilter_repo"])),
             "--pointfilter-python", str(project_path(project, cfg["pointfilter_python"])), "--seed", str(cfg["seed"])],
            full, "filters_*", ["metrics.json", "filter_comparison.csv", "scan_pointfilter.npy"]))
        final_report(run, full, filters)
        manifest["status"] = "completed"
        manifest["report"] = str(run / "report.md")
        manifest["full_alignment"] = str(full)
        manifest["filters"] = str(filters)
        manifest.pop("error", None)
        write_json(state_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = type(error).__name__ + ": " + str(error)
        for record in manifest["stages"].values():
            if record["status"] == "running":
                record["status"] = "failed"
                record["error"] = manifest["error"]
        write_json(state_path, manifest)
        print("Run preserved:", run, flush=True)
        raise
    print("\nCOMPLETED:", run / "report.md", flush=True)
    print("TABLE:", run / "filter_comparison.csv", flush=True)
    return run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--resume", type=Path, help="Rebuild report or reuse completed stages; interrupted stages need a fresh run")
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("Run with the Windows .venv Python; this launcher starts WSL itself")
    config_path = args.config.expanduser().absolute()
    cfg = read_json(config_path)
    if cfg["preparation"]["mode"] not in ("reuse_fixed_sample", "resample"):
        parser.error("Unknown preparation mode")
    if cfg["icp_iterations"] <= 0 or not 3 <= cfg["modelnet_points"] <= 717:
        parser.error("Invalid ICP iterations or ModelNet point count (3..717)")
    if not cfg["sor"]:
        parser.error("At least one SOR configuration is required")
    run_pipeline(cfg, config_path, args.resume)


if __name__ == "__main__":
    main()
