import open3d as o3d
import numpy as np
import matplotlib as mpl
from pathlib import Path
from time import perf_counter
import json

folder = Path(__file__).resolve().parent

stl = o3d.io.read_triangle_mesh("shoe.stl")

ply = o3d.io.read_point_cloud("shoe_clean.ply")

aligment_path = Path(__file__).resolve().parent / "aligment.npz"

if aligment_path.exists():
    with np.load(aligment_path) as saved:
        manual_matrix = saved["manual_transform"]
        icp_transform = saved["icp_transform"]
        manual_c_ply = saved["manual_c_ply"]
        manual_c_s = saved["manual_c_s"]
    
    ply.paint_uniform_color([0, 1, 0])
    stl.paint_uniform_color([1, 0, 0])
    
    ply.translate(-manual_c_ply)
    stl.translate(-manual_c_s)
    reference_pcd = stl.sample_points_uniformly(number_of_points=50000)
    reference_pcd.paint_uniform_color([0, 0, 1])

    ply.transform(manual_matrix)
    o3d.visualization.draw_geometries([ply, stl])

    ply.transform(icp_transform)
    o3d.visualization.draw_geometries([ply, stl])



else:
    def pick_points(pcd):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        picked = vis.get_picked_points()
        vis.destroy_window()
        return picked 

    count_stl_vertices = len(stl.vertices)

    count_stl_triangles = len(stl.triangles)

    count_ply_points = len(ply.points)

    ply.paint_uniform_color([0, 1, 0])
    stl.paint_uniform_color([1, 0, 0])

    center_ply = ply.get_center()
    ply.translate(-center_ply)

    center_stl = stl.get_center()
    stl.translate(-center_stl)

    reference_pcd = stl.sample_points_uniformly(number_of_points=50000)
    reference_pcd.paint_uniform_color([0, 0, 1])
    count_rf = len(reference_pcd.points)

    print(stl.get_min_bound())
    print(stl.get_max_bound())

    print(ply.get_min_bound())
    print(ply.get_max_bound())

    print("Количество элементов:", count_stl_vertices)
    print("Количество элементов:", count_stl_triangles)
    print("Количество элементов:", count_ply_points)
    print("Количество элементов:", count_rf)

    picked_ply = pick_points(ply)
    picked_reference = pick_points(reference_pcd)

    pairs = o3d.utility.Vector2iVector(
        list(zip(picked_ply, picked_reference))
    )

    estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=True
    )

    matrix = estimator.compute_transformation(ply, reference_pcd, pairs)

    ply.transform(matrix)

    o3d.visualization.draw_geometries([ply, stl])

    size = reference_pcd.get_max_bound() - reference_pcd.get_min_bound()
    treshold = 0.02 * max(size)

    before = o3d.pipelines.registration.evaluate_registration(
        ply,reference_pcd, treshold
    )
    print("До ICP:", before)

    result = o3d.pipelines.registration.registration_icp(ply, reference_pcd, treshold)
    print("result:", result)
    ply.transform(result.transformation)

    o3d.visualization.draw_geometries([ply, stl])

    np.savez(
        aligment_path, 
        manual_transform=matrix, 
        icp_transform=result.transformation,
        manual_c_ply = center_ply, 
        manual_c_s = center_stl
        )


scan_points = np.asarray(ply.points, dtype=np.float32)
scan_path = Path(__file__).resolve().parent / "scan_for_ai.npy"
np.save(scan_path, scan_points)

print("Вход для модели:", scan_points.shape)

distance_start = perf_counter()
scene = o3d.t.geometry.RaycastingScene(nthreads=0)
mesh = o3d.t.geometry.TriangleMesh.from_legacy(stl)
g_id = scene.add_triangles(mesh)

points = o3d.core.Tensor(
     np.asarray(ply.points),
     dtype=o3d.core.Dtype.Float32
) 

distances = scene.compute_distance(points, nthreads=0)
distance_seconds = perf_counter() - distance_start
print(f"Время расчёта отклонений: {distance_seconds:.3f}c")

print("Расстояний:", distances.shape[0])
print("Точек скана:", len(ply.points))

distances_np = distances.numpy()

mean = np.mean(distances_np)
median = np.median(distances_np)
max_distance = np.max(distances_np)

normalized = distances_np / np.max(distances_np)

colors = mpl.colormaps["turbo"](normalized)
ply.colors = o3d.utility.Vector3dVector(colors[:,:3])
o3d.visualization.draw_geometries([ply])

ply_p = np.asarray(ply.points)
fin = np.column_stack((ply_p, distances_np))

output_path = Path(__file__).resolve().parent / "deviations.csv"

np.savetxt(
     output_path,
     fin,
     delimiter=",",
     header="x,y,z,distance",
     comments=""
)

summary_path = Path(__file__).resolve().parent / "summary.txt"

with open(summary_path, "w", encoding="utf-8") as report:
    report.write(f"Время расчета отклонений: {distance_seconds:.3f} c\n")
    report.write(f"Среднее расстояние: {mean}\n")


params_path = Path(__file__).resolve().parent / "filter_params.json"

with open(params_path, "r", encoding="utf-8") as file:
    experiments = json.load(file)

print(mean, median, max_distance)

rows = []

filter_comparison_path = Path(__file__).resolve().parent / "filter_comparison.csv"

for params in experiments:
    filter_distance_start = perf_counter()
    filtered_ply, kept_indicies = ply.remove_statistical_outlier(
        nb_neighbors=params["nb_neighbors"], std_ratio = params["std_ratio"]
        )
    filter_seconds = perf_counter() - filter_distance_start
    
    removed_ply = ply.select_by_index(kept_indicies, invert=True)

    filtered_ply.paint_uniform_color([1, 1, 0])
    removed_ply.paint_uniform_color([0, 1, 1])

    o3d.visualization.draw_geometries([filtered_ply,removed_ply])

    filtered_distance = distances_np[kept_indicies]

    fdmax = np.max(filtered_distance)
    fdmedin = np.median(filtered_distance)
    fdmean = np.mean(filtered_distance)

    rows.append([
    params["nb_neighbors"],
    params["std_ratio"],
    len(filtered_ply.points),
    len(removed_ply.points),
    fdmean,
    fdmedin,
    fdmax,
    filter_seconds
    ])

np.savetxt(
    filter_comparison_path,
    rows,
    delimiter=",",
    header="nb_neighbors,std_ratio,kept,removed,mean,median,max,filter_seconds",
    fmt=["%d", "%.1f", "%d", "%d", "%.8f", "%.8f", "%.8f", "%.6f"],
    comments=""
    )
        
filter_ai = np.load(folder / "Pointfilter" / "scan_pointfilter.npy")

print(np.isfinite(filter_ai).all())

points_ai_filter = o3d.core.Tensor(
     np.asarray(filter_ai),
     dtype=o3d.core.Dtype.Float32
) 

distances_ai_np = scene.compute_distance(points_ai_filter, nthreads=0).numpy()

mean_ai = np.mean(distances_ai_np)
median_ai = np.median(distances_ai_np)
max_distance_ai = np.max(distances_ai_np)

print(mean_ai, median_ai, max_distance_ai)

ai_cloud = o3d.geometry.PointCloud()
ai_cloud.points = o3d.utility.Vector3dVector(filter_ai)
ai_cloud.paint_uniform_color([1, 0, 1])
o3d.visualization.draw_geometries([ply, ai_cloud])