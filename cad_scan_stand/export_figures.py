#!/usr/bin/env python3
"""Export report figures from a completed run; no registration or AI inference."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


UNITS = "в исходных единицах CAD"
COLORS = ["#2864b7", "#d28017", "#008b79"]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def latest_run(config_path):
    cfg = read_json(config_path)
    project = (Path(config_path).parent / cfg["project_dir"]).resolve()
    results = Path(cfg["results_dir"])
    if not results.is_absolute():
        results = project / results
    runs = sorted(path for path in results.glob("run_*")
                  if (path / "summary.json").is_file() and (path / "run.json").is_file()
                  and read_json(path / "run.json").get("status") == "completed")
    if not runs:
        raise FileNotFoundError("No completed stand run found in " + str(results))
    return runs[-1]


def plot_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "pdf.fonttype": 42,
        "axes.titleweight": "semibold", "axes.spines.top": False,
        "axes.spines.right": False, "axes.unicode_minus": True,
    })


def save_figure(fig, output, name, dpi):
    for suffix in ("png", "pdf"):
        fig.savefig(output / (name + "." + suffix), dpi=dpi,
                    facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("Saved:", output / (name + ".png"), flush=True)


def metrics_plot(summary, output, dpi):
    rows = summary["filtering"]["results"][1:]
    count = len(rows)
    labels = [("SOR {}/{}".format(row["nb_neighbors"], "{:g}".format(row["std_ratio"]))
               if row["nb_neighbors"] is not None else row["method"]) for row in rows]
    y = np.arange(count)
    fig, (axis, removed) = plt.subplots(1, 2, figsize=(11.3, max(4.5, 0.49 * count + 1.7)),
                                      sharey=True, gridspec_kw={"width_ratios": [3.7, 1.25]})
    keys = [("mean", "Среднее"), ("p95", "P95"), ("std", "Станд. отклонение")]
    for offset, ((key, label), color) in enumerate(zip(keys, COLORS)):
        values = [-float(row[key + "_change_percent"])
                  if row[key + "_change_percent"] is not None else np.nan for row in rows]
        bars = axis.barh(y + (offset - 1) * 0.23, values, height=0.22, color=color, label=label)
        axis.bar_label(bars, labels=["{:.1f}".format(v) if np.isfinite(v) else "н/д" for v in values],
                       padding=3, fontsize=8)
    axis.set_yticks(y, labels)
    axis.invert_yaxis()
    axis.axvline(0, color="#666666", linewidth=0.7)
    axis.set_xlabel("Уменьшение показателя расстояний до CAD, %")
    axis.set_title("Изменение геометрических показателей", loc="left", pad=32)
    axis.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=3, fontsize=8.5, frameon=False)
    axis.grid(axis="x", alpha=0.17)
    axis.set_axisbelow(True)
    axis.margins(x=0.18)
    values = [row["removed_percent"] for row in rows]
    bars = removed.barh(y, values, height=0.62, color="#a1acb8")
    removed.bar_label(bars, labels=["{:.2f}%".format(value) for value in values], padding=4, fontsize=8.5)
    removed.set_xlim(0, max(1.0, max(values)) * 1.5)
    removed.set_xlabel("Удалено точек, %")
    removed.set_title("Сохранность выборки", loc="left", pad=32)
    removed.tick_params(axis="y", left=False)
    removed.grid(axis="x", alpha=0.17)
    removed.set_axisbelow(True)
    fig.suptitle("SOR и Pointfilter при фиксированном совмещении", x=0.04, ha="left", fontsize=14, y=0.99)
    fig.text(0.04, 0.025,
             "Сравнение с облаком до фильтрации. Отрицательное уменьшение означает рост показателя.\n"
             "SOR: статистики оставшихся точек. Pointfilter: координаты изменяются, число точек сохраняется.",
             fontsize=8.5, color="#444444")
    fig.subplots_adjust(left=0.16, right=0.97, top=0.80, bottom=0.16, wspace=0.19)
    save_figure(fig, output, "03_filter_metrics", dpi)


def pca_frame(vertices):
    centered = vertices - vertices.mean(axis=0)
    _, vectors = np.linalg.eigh(centered.T @ centered / len(vertices))
    basis = vectors[:, ::-1].copy()
    for column in range(3):
        if basis[np.argmax(abs(basis[:, column])), column] < 0:
            basis[:, column] *= -1
    if np.linalg.det(basis) < 0:
        basis[:, 2] *= -1
    return vertices.mean(axis=0), basis


def check_saved_file(path, manifest, stage, sources):
    expected = [value for name, value in manifest["stages"][stage]["artifact_hashes"].items()
                if str(name).replace("\\", "/").split("/")[-1] == path.name]
    if len(expected) != 1 or file_hash(path) != expected[0]:
        raise ValueError("Saved result hash mismatch: " + str(path))
    sources[str(path)] = expected[0]


def checked_array(path, manifest, stage, sources):
    check_saved_file(path, manifest, stage, sources)
    array = np.load(path)
    if not np.isfinite(array).all():
        raise ValueError("Non-finite coordinates/distances: " + str(path))
    return array


def validate_stats(distances, row):
    if distances.ndim != 1 or len(distances) != row["kept"]:
        raise ValueError("Distance array count differs from its summary")
    got = [np.mean(distances), np.std(distances), np.percentile(distances, 95), np.max(distances)]
    expected = [row[key] for key in ("mean", "std", "p95", "max")]
    if not np.allclose(got, expected, rtol=1e-5, atol=1e-9):
        raise ValueError("Distance array differs from the saved statistics")


def load_geometry_data(run, manifest, summary, neighbors, ratio, mesh_faces):
    import open3d as o3d
    sources = {}
    full = Path(manifest["full_alignment"])
    filters = Path(manifest["filters"])
    mesh_path = run / "data/shoe.stl"
    if file_hash(mesh_path) != summary["registration"]["mesh_sha256"]:
        raise ValueError("The saved CAD mesh has changed")
    sources[str(mesh_path)] = file_hash(mesh_path)
    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if not mesh.has_triangles():
        raise ValueError("CAD mesh is empty")
    center, basis = pca_frame(np.asarray(mesh.vertices))
    original_faces = len(mesh.triangles)
    if mesh_faces > 0 and original_faces > mesh_faces:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=mesh_faces)
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    raw = checked_array(full / "aligned_scan.npy", manifest, "05_surface_evaluation", sources)
    control = checked_array(full / "manual_control_aligned_scan.npy", manifest, "05_surface_evaluation", sources)
    ai = checked_array(filters / "scan_pointfilter.npy", manifest, "06_filtering", sources)
    before = checked_array(filters / "distances_raw.npy", manifest, "06_filtering", sources)
    after = checked_array(filters / "distances_pointfilter.npy", manifest, "06_filtering", sources)
    rows = summary["filtering"]["results"]
    selected = [(index, row) for index, row in enumerate(rows[1:], 1)
                if row["nb_neighbors"] == neighbors and row["std_ratio"] == ratio]
    if len(selected) != 1:
        raise ValueError("Selected SOR parameters are absent from this run")
    index, sor_row = selected[0]
    sor_file = filters / ("sor_{:02d}.npz".format(index))
    check_saved_file(sor_file, manifest, "06_filtering", sources)
    with np.load(sor_file) as saved:
        sor, indices, sor_distances = saved["points"], saved["kept_indices"], saved["distances"]
    if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(raw))):
        raise ValueError("Invalid retained-point indices")
    if (sor.shape != (len(indices), 3) or not np.isfinite(sor).all()
            or not np.allclose(sor, raw[indices], rtol=1e-6, atol=1e-9)
            or not np.allclose(sor_distances, before[indices], rtol=1e-6, atol=1e-9)):
        raise ValueError("SOR input does not match the common aligned cloud")
    n = summary["registration"]["source_point_count"]
    if raw.shape != (n, 3) or control.shape != raw.shape or ai.shape != raw.shape:
        raise ValueError("Cloud shapes do not match this run")
    ai_row = next(row for row in rows if row["method"] == "Pointfilter")
    for distances, row in ((before, rows[0]), (after, ai_row), (sor_distances, sor_row)):
        validate_stats(distances, row)
    transform = lambda points: (points - center) @ basis
    raw, control, ai, sor, vertices = [transform(p) for p in (raw, control, ai, sor, vertices)]
    all_points = np.vstack((vertices, raw, control, ai))
    lower, upper = all_points.min(axis=0), all_points.max(axis=0)
    padding = np.maximum((upper - lower) * 0.05, 1e-9)
    return dict(vertices=vertices, triangles=triangles, raw=raw, control=control, ai=ai, sor=sor,
                before=before, after=after, sor_distances=sor_distances, sor_label="SOR {}/{:g}".format(neighbors, ratio),
                bounds=(lower - padding, upper + padding), source_hashes=sources,
                pca_origin=center.tolist(), pca_basis=basis.tolist(),
                original_mesh_triangles=original_faces, display_mesh_triangles=len(triangles))


def setup_3d(axis, bounds, elevation, azimuth, roll=180):
    lower, upper = bounds
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    axis.set_zlim(lower[2], upper[2])
    axis.set_box_aspect(upper - lower, zoom=1.33)
    axis.set_proj_type("ortho")
    # Rotate only the camera around its viewing axis. Data and labels stay unchanged.
    axis.view_init(elev=elevation, azim=azimuth, roll=roll)
    axis.set_axis_off()


def registration_plot(data, output, dpi, elevation, azimuth, roll=180):
    faces = data["vertices"][data["triangles"]]
    normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-15)
    light = np.array([-0.5, -0.7, 1.0])
    light /= np.linalg.norm(light)
    shades = 0.65 + 0.35 * abs(normals @ light)
    facecolors = shades[:, None] * np.array([0.94, 0.73, 0.28])[None, :]
    fig = plt.figure(figsize=(11.8, 4.6))
    for index, (name, cloud) in enumerate((("Ручное + ICP — контроль", data["control"]),
                                            ("GeoTransformer + ICP + поиск ориентации", data["raw"])), 1):
        axis = fig.add_subplot(1, 2, index, projection="3d")
        surface = Poly3DCollection(faces, facecolors=facecolors, edgecolors="none", alpha=0.30,
                                  linewidths=0, rasterized=True)
        axis.add_collection3d(surface)
        axis.scatter(*cloud.T, s=1.5, c="#1552ac", alpha=0.92, linewidths=0,
                     depthshade=False, rasterized=True)
        setup_3d(axis, data["bounds"], elevation, azimuth, roll)
        axis.set_title(name, fontsize=10.5, pad=0)
    fig.suptitle("Совмещение полного скана с эталоном", fontsize=14, y=0.99)
    fig.legend(handles=[Patch(facecolor="#dfbd65", alpha=0.7, label="Поверхность CAD"),
                        Line2D([], [], color="#1552ac", marker=".", linestyle="none", label="Точки скана")],
               loc="lower center", bbox_to_anchor=(0.5, 0.07), ncol=2, frameon=False)
    note = "Одинаковые ракурс и масштаб; ручной вариант — контроль, а не истинная поза."
    if data["display_mesh_triangles"] < data["original_mesh_triangles"]:
        note += "\nCAD упрощён только для отрисовки; численные результаты рассчитаны по исходной сетке."
    fig.text(0.5, 0.015, note, ha="center", fontsize=8.2, color="#444444")
    fig.subplots_adjust(left=0.0, right=1.0, top=0.90, bottom=0.16, wspace=0)
    save_figure(fig, output, "01_registration", dpi)


def deviation_plot(data, output, dpi, elevation, azimuth, roll=180):
    sets = [("До фильтрации", data["raw"], data["before"]),
            (data["sor_label"], data["sor"], data["sor_distances"]),
            ("Pointfilter", data["ai"], data["after"])]
    vmax = max(float(np.max(distances)) for _, _, distances in sets)
    norm = Normalize(vmin=0, vmax=vmax if vmax > 0 else 1e-12)
    fig = plt.figure(figsize=(13.0, 4.3))
    for index, (name, cloud, distances) in enumerate(sets, 1):
        axis = fig.add_subplot(1, 3, index, projection="3d")
        axis.scatter(*cloud.T, c=distances, cmap="turbo", norm=norm, s=2,
                     linewidths=0, depthshade=False, rasterized=True)
        setup_3d(axis, data["bounds"], elevation, azimuth, roll)
        axis.set_title(name + "\n{} точек".format(len(cloud)), fontsize=10.5, pad=0)
    fig.subplots_adjust(left=0, right=0.89, top=0.84, bottom=0.12, wspace=0)
    bar_axis = fig.add_axes([0.90, 0.23, 0.014, 0.52])
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="turbo"), cax=bar_axis)
    bar.set_label("Расстояние до CAD, исх. ед.", fontsize=9)
    bar.ax.tick_params(labelsize=8)
    bar.formatter.set_powerlimits((-3, -3))
    bar.update_ticks()
    fig.suptitle("Отклонения от CAD до и после фильтрации", fontsize=14, y=0.99)
    fig.text(0.46, 0.025, "Единая цветовая шкала и ракурс. Синий — меньшее расстояние, красный — большее.\n"
             "Цвет показывает расстояние до CAD, а не подтверждённый шум или дефект.",
             ha="center", fontsize=8.4, color="#444444")
    save_figure(fig, output, "02_deviation_maps", dpi)


def distribution_plot(data, output, dpi):
    fig, axis = plt.subplots(figsize=(8.4, 5.3))
    sets = [("До фильтрации", data["before"]), (data["sor_label"], data["sor_distances"]),
            ("Pointfilter", data["after"])]
    for (label, distances), color in zip(sets, COLORS):
        sorted_values = np.sort(distances)
        axis.step(np.r_[0, sorted_values], np.r_[0, np.arange(1, len(sorted_values) + 1) / len(sorted_values)],
                  where="post", label="{} (n={})".format(label, len(sorted_values)), color=color, linewidth=1.7)
    axis.axhline(0.95, color="#777777", linestyle="--", linewidth=0.9, label="Уровень P95")
    axis.set_xlim(left=0)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Расстояние до поверхности CAD, " + UNITS)
    axis.set_ylabel("Доля точек с расстоянием не больше указанного")
    axis.set_title("Распределение расстояний до эталона", loc="left", pad=15)
    axis.grid(alpha=0.18)
    axis.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    fig.text(0.12, 0.025, "SOR: распределение оставшихся точек. Pointfilter: все исходные точки с изменёнными координатами.",
             fontsize=8.1, color="#444444")
    fig.subplots_adjust(bottom=0.20, left=0.12, right=0.97, top=0.89)
    save_figure(fig, output, "04_distance_distribution", dpi)


def export(args):
    plot_style()
    run = args.run.expanduser().resolve() if args.run else latest_run(args.config.expanduser().resolve())
    manifest, summary = read_json(run / "run.json"), read_json(run / "summary.json")
    if manifest.get("status") != "completed":
        raise ValueError("The selected run is not completed")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    output = run / ("figures_" + stamp)
    output.mkdir()
    info = {"source_run": str(run), "run_sha256": file_hash(run / "run.json"),
            "summary_sha256": file_hash(run / "summary.json"), "dpi": args.dpi,
            "matplotlib": matplotlib.__version__, "numpy": np.__version__,
            "units": "original CAD coordinate units; physical unit not verified",
            "neural_inference_repeated": False, "plot_data": "saved results only",
            "metrics_only": args.metrics_only,
            "figure_formats": ["png", "pdf"], "white_background": True}
    captions = ["03_filter_metrics: изменение среднего, P95 и стандартного отклонения расстояний до CAD, а также доля удалённых точек; проценты относительно исходного совмещённого облака."]
    try:
        metrics_plot(summary, output, args.dpi)
        if not args.metrics_only:
            data = load_geometry_data(run, manifest, summary, args.sor_neighbors, args.sor_std, args.mesh_faces)
            info.update({key: data[key] for key in ("source_hashes", "pca_origin", "pca_basis",
                                                    "original_mesh_triangles", "display_mesh_triangles")})
            info.update(elevation=args.elev, azimuth=args.azim, camera_roll_degrees=args.roll,
                        common_color_limits=[0, max(float(d.max()) for d in (data["before"], data["after"], data["sor_distances"]))])
            registration_plot(data, output, args.dpi, args.elev, args.azim, args.roll)
            deviation_plot(data, output, args.dpi, args.elev, args.azim, args.roll)
            distribution_plot(data, output, args.dpi)
            captions += [
                "01_registration: совмещение полного скана с поверхностью CAD: слева ручное + ICP как контроль, справа GeoTransformer + ICP + геометрический поиск ориентации. Ракурс и масштаб одинаковы; отображение CAD может быть упрощено, метрики рассчитаны по исходной сетке.",
                "02_deviation_maps: расстояния до исходной треугольной поверхности CAD до фильтрации, после SOR и после Pointfilter. Общая цветовая шкала и одинаковый ракурс; SOR показан только по сохранённым точкам.",
                "04_distance_distribution: эмпирические функции распределения расстояний до поверхности CAD. Для SOR используются оставшиеся точки, для Pointfilter — все перемещённые точки."]
        info["status"] = "completed"
        (output / "captions.txt").write_text("\n\n".join(sorted(captions)) + "\n", encoding="utf-8")
    except BaseException as error:
        info["status"] = "failed"
        info["error"] = type(error).__name__ + ": " + str(error)
        raise
    finally:
        (output / "figures.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("FIGURES READY:", output, flush=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.json"))
    parser.add_argument("--run", type=Path, help="Default: latest completed run from config.json")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--elev", type=float, default=20, help="View elevation in degrees")
    parser.add_argument("--azim", type=float, default=-65, help="View azimuth in degrees")
    parser.add_argument("--roll", type=float, default=180, help="Camera roll in degrees; 0 restores the original view")
    parser.add_argument("--sor-neighbors", type=int, default=20)
    parser.add_argument("--sor-std", type=float, default=2.0)
    parser.add_argument("--mesh-faces", type=int, default=30000, help="Display only; 0 uses all original triangles")
    parser.add_argument("--metrics-only", action="store_true", help="Make the summary chart without geometry files or Open3D")
    args = parser.parse_args()
    if not 100 <= args.dpi <= 600 or args.mesh_faces < 0:
        parser.error("DPI must be 100..600; mesh-faces must be nonnegative")
    export(args)


if __name__ == "__main__":
    main()
