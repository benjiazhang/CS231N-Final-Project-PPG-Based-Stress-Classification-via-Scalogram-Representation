import modal

app = modal.App("moment-baseline")

volume = modal.Volume.from_name("stress-data")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("setuptools", "wheel")
    .pip_install(
        "torch",
        "torchvision",
        "momentfm",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "huggingface_hub",
    )
    .run_commands(
        # Pre-download MOMENT weights at build time to avoid DNS failures mid-run
        "python -c \"from momentfm import MOMENTPipeline; "
        "MOMENTPipeline.from_pretrained('AutonLab/MOMENT-1-base', "
        "model_kwargs={'task_name': 'classification', 'n_channels': 1, 'num_class': 2})\""
    )
    .add_local_python_source("moment_baseline")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 24,
    volumes={"/data": volume},
)
def run(stages: list = ["cv", "test", "loso"]):
    import moment_baseline as mb
    import numpy as np, json, os

    volume.reload()

    print(f"Device: {mb.DEVICE}")
    print(f"Stages: {stages}")

    print("\nLoading WESAD...")
    Xw, yw, sw = mb.load_wesad("/data/data/WESAD")

    print("\nLoading UBFC-Phys...")
    Xu, yu, su = mb.load_ubfc("/data/data/ubfc_baselines.npz")

    X = np.concatenate([Xw, Xu])
    y = np.concatenate([yw, yu])
    subjects = np.concatenate([sw, su])
    print(f"\nPooled: {len(y)} windows, {len(np.unique(subjects))} subjects")

    splits = mb.subject_split(X, y, subjects)
    Xall = np.concatenate([splits["train"][0], splits["val"][0], splits["test"][0]])
    yall = np.concatenate([splits["train"][1], splits["val"][1], splits["test"][1]])
    sall = np.concatenate([splits["train"][2], splits["val"][2], splits["test"][2]])

    out_root = "/data/results/baselines/moment"
    os.makedirs(out_root, exist_ok=True)

    bp_path = os.path.join(out_root, "best_params.json")
    if "cv" in stages:
        Xtv = np.concatenate([splits["train"][0], splits["val"][0]])
        ytv = np.concatenate([splits["train"][1], splits["val"][1]])
        stv = np.concatenate([splits["train"][2], splits["val"][2]])
        best_params = mb.cv_sweep(Xtv, ytv, stv, out_root)
        volume.commit()
    elif os.path.exists(bp_path):
        best_params = json.load(open(bp_path))
    else:
        best_params = {k: v[0] for k, v in mb.GRID.items()}

    summary = {"best_params": best_params}
    if "test" in stages:
        summary["test"] = mb.final_test(best_params, splits, out_root)
        volume.commit()
    if "loso" in stages:
        summary["loso"] = mb.loso(best_params, Xall, yall, sall, out_root)
        volume.commit()

    with open(os.path.join(out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    volume.commit()
    print(f"\nDone. Results in {out_root}")


@app.local_entrypoint()
def main():
    run.remote(stages=["cv", "test", "loso"])
