import os

import h5py
import numpy as np

EMBODIED_PATH = os.path.dirname(os.path.abspath(__file__))
REPO_PATH = os.path.dirname(EMBODIED_PATH)
LIBERO_REPO_PATH = os.path.join(REPO_PATH, "LIBERO")

dataset_a = "libero_spatial"
dataset_b = "libero_spatial2"

path_a = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset_a}"
path_b = f"{LIBERO_REPO_PATH}/libero/datasets_with_logits/{dataset_b}"

task_files_a = sorted([f for f in os.listdir(path_a) if f.endswith(".hdf5")])
task_files_b = sorted([f for f in os.listdir(path_b) if f.endswith(".hdf5")])

assert task_files_a == task_files_b, "❌ Dataset files do not match!"

print(f"Found {len(task_files_a)} matching tasks\n")

for task_file in task_files_a:
    print(f"Checking {task_file}")

    file_a = os.path.join(path_a, task_file)
    file_b = os.path.join(path_b, task_file)

    with h5py.File(file_a, "r") as fa, h5py.File(file_b, "r") as fb:
        demo_a = fa["data"]["demo_0"]
        demo_b = fb["data"]["demo_0"]

        # ---- Load observations ----
        obs_a = demo_a["obs"]["agentview_rgb"][:]
        obs_b = demo_b["obs"]["agentview_rgb"][:]

        if obs_a.shape != obs_b.shape:
            print("  ❌ Observation shape mismatch:", obs_a.shape, obs_b.shape)
            continue

        # ---- Load actions ----
        actions_a = demo_a["actions"][:]
        actions_b = demo_b["actions"][:]

        if actions_a.shape != actions_b.shape:
            print("  ❌ Action shape mismatch:", actions_a.shape, actions_b.shape)
            continue

        # ---- Compare actions ----
        if np.array_equal(actions_a, actions_b):
            print("  ✅ Actions match exactly")
        else:
            # fallback for floating-point noise
            if np.allclose(actions_a, actions_b, atol=1e-6):
                print("  ⚠️ Actions match within tolerance")
            else:
                diff = np.abs(actions_a - actions_b).max()
                print(f"  ❌ Actions differ! max |Δ| = {diff}")

    print("--------------------------------------")
