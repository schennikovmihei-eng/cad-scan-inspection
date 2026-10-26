from pathlib import Path
import torch
import numpy as np
from Pointfilter_Network_Architecture import pointfilternet
from Pointfilter_DataLoader import PointcloudPatchDataset
from time import perf_counter

folder = Path(__file__).resolve().parent
weights_path = folder / "Summary" / "pre_train_model" / "model_full_ae.pth"

model = pointfilternet()

chekpoint = torch.load(
    weights_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(chekpoint["state_dict"])
model.to("cuda")
model.eval()

print("Модель загружена на", next(model.parameters()).device)

np.random.seed(42)

dataset = PointcloudPatchDataset(
    root=str(folder.parent),
    shape_name="scan_for_ai",
    train_state="evaluation",
    patch_radius=0.05,
    points_per_patch=500,
    seed=42
)

for point_index in range(len(dataset)):
    sample = dataset[point_index]
    if sample is not None:
        break
    else:
        raise RuntimeError("Подходящая окрестность не найдена")

patch, inverse_rotation, center = sample

batch = patch.T.unsqueeze(0).contiguous().float().to("cuda")

with torch.inference_mode():
    prediction = model(batch)

print("Вход", batch.shape)
print("Выход", prediction.shape)
print("Числа конечные", torch.isfinite(prediction).all().item())

result_pr = prediction.cpu().squeeze(dim=0)
prediction_np = result_pr.numpy()
prediction_np *= dataset.patch_radius_absolute[0]
pred_rot = inverse_rotation.numpy()@prediction_np
pred_rot += center

res_fin = np.linalg.norm(pred_rot - center)
print(res_fin, pred_rot, center)

points_copy = np.load(folder.parent / "scan_for_ai.npy").copy()

distance_start = perf_counter()
count_pr = 0
for point_index in range(len(points_copy)):
    sample = dataset[point_index]
    if sample is None:
        count_pr += 1
        continue
    else:
        patch, inverse_rotation, center = sample

        batch = patch.T.unsqueeze(0).contiguous().float().to("cuda")

        with torch.inference_mode():
            prediction = model(batch)

        result_pr = prediction.cpu().squeeze(dim=0)
        prediction_np = result_pr.numpy()
        prediction_np *= dataset.patch_radius_absolute[0]
        pred_rot = inverse_rotation.numpy()@prediction_np
        pred_rot += center
    points_copy[point_index] = pred_rot

distance_seconds = perf_counter() - distance_start
print(f"Время расчёта отклонений: {distance_seconds:.3f}c")

print(points_copy.shape, count_pr)

scan_pointfilter = Path(__file__).resolve().parent / "scan_pointfilter.npy"
np.save(scan_pointfilter, points_copy)