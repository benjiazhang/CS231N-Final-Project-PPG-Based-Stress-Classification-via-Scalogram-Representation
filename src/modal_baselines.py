import modal

app = modal.App("baselines-pipeline")

volume = modal.Volume.from_name("stress-data")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "scipy",
    )
    .add_local_python_source("baselines")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 24,
    volumes={"/data": volume},
)
def run(models: list = ["lstm", "cnn1d"], stages: list = ["cv", "test", "loso"]):
    import baselines as bl

    bl.main.__wrapped__ = None  # bypass argparse — call directly
    # Load data
    print(f"Device: {bl.DEVICE}")

    print("\nLoading WESAD...")
    Xw, yw, sw = bl.load_wesad("/data/WESAD")

    print("\nLoading UBFC-Phys...")
    Xu, yu, su = bl.load_ubfc("/data/ubfc_baselines.npz")  # or "/data/bvp_ubfc" for raw CSVs

    import numpy as np
    X = np.concatenate([Xw, Xu])
    y = np.concatenate([yw, yu])
    subjects = np.concatenate([sw, su])
    print(f"\nPooled: {len(y)} windows, {len(np.unique(subjects))} subjects")

    splits = bl.subject_split(X, y, subjects)
    Xall = np.concatenate([splits["train"][0], splits["val"][0], splits["test"][0]])
    yall = np.concatenate([splits["train"][1], splits["val"][1], splits["test"][1]])
    sall = np.concatenate([splits["train"][2], splits["val"][2], splits["test"][2]])

    import json, os
    out_root = "/data/results/baselines"
    run_cv   = "cv"   in stages
    run_test = "test" in stages
    run_loso = "loso" in stages

    overall = {}
    for model_key in models:
        out_dir = os.path.join(out_root, model_key)
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n\n########## {model_key.upper()} ##########")

        bp_path = os.path.join(out_dir, "best_params.json")
        if run_cv:
            Xtv = np.concatenate([splits["train"][0], splits["val"][0]])
            ytv = np.concatenate([splits["train"][1], splits["val"][1]])
            stv = np.concatenate([splits["train"][2], splits["val"][2]])
            best_params = bl.cv_sweep(model_key, Xtv, ytv, stv, out_dir)
        elif os.path.exists(bp_path):
            best_params = json.load(open(bp_path))
        else:
            best_params = {k: v[0] for k, v in bl.GRID[model_key].items()}

        model_summary = {"best_params": best_params}
        if run_test:
            model_summary["test"] = bl.final_test(model_key, best_params, splits, out_dir)
        if run_loso:
            model_summary["loso"] = bl.loso(model_key, best_params, Xall, yall, sall, out_dir)

        overall[model_key] = model_summary
        with open(os.path.join(out_dir, "summary.json"), "w") as f:
            json.dump(model_summary, f, indent=2)

    with open(os.path.join(out_root, "baselines_summary.json"), "w") as f:
        json.dump(overall, f, indent=2)

    volume.commit()
    print(f"\nDone. Results in {out_root}")


@app.local_entrypoint()
def main():
    run.remote(models=["lstm", "cnn1d"], stages=["cv", "test", "loso"])
