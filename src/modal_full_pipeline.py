import modal

# Build the image with all dependencies
image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch", "torchvision",
        "transformers",   # for CvT
        "timm",           # for AudioMAE
        "scikit-learn",
        "pandas",
        "matplotlib",
        "numpy",
    )
    .add_local_python_source("full_pipeline_all_models")
)

# Connect to your persistent volume (create it first if needed)
volume = modal.Volume.from_name("stress_data", create_if_missing=True)

app = modal.App("full_pipeline_all_models", image=image)

@app.function(
    gpu="A10G",               # or "T4", "A100", etc.
    timeout=60 * 60 * 6,      # 6 hours
    volumes={"/stress_data": volume},
)
def run_training(models=None,
    stages=None,
    cv_folds=None,
    phase2_blocks=None,
    quick=None,
    out_root=None,
    train_path=None,
    val_path=None,
    test_path=None,
    ):
    import full_pipeline_all_models
    full_pipeline_all_models.main(
        models=models,
        stages=stages,
        cv_folds=cv_folds,
        phase2_blocks=phase2_blocks,
        quick=quick,
        train_path=train_path,
        val_path=val_path,
        test_path=test_path,
        out_root=out_root,
    )

@app.local_entrypoint()
def main(
    models: str = "efficientnet",
    stages: str = "test",
    cv_folds: int = 5,
    phase2_blocks: int = 3,
    quick: bool = False,
    gpu: str = "A10G",
):
    run_training.remote(
        models=models.split(","),
        stages=stages.split(","),
        cv_folds=cv_folds,
        phase2_blocks=phase2_blocks,
        quick=(quick or None),
        out_root="/stress_data/results/all_models",
        train_path="/stress_data/data/trainSINGLE.npz",
        val_path="/stress_data/data/valSINGLE.npz",
        test_path="/stress_data/data/testSINGLE.npz",
    )