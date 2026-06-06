import modal

app = modal.App("dinov2-pipeline")

volume = modal.Volume.from_name("stress-data")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch",
        "torchvision",
        "transformers",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "scipy",
        "Pillow",
    )
    .add_local_python_source("full_pipeline_all_models")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 24,
    volumes={"/data": volume},
)
def run(phase2_blocks: int = 0, full_finetune: bool = False, skip_cv: bool = False):
    import full_pipeline_all_models as pipeline

    pipeline.main(
        models=["dinov2"],
        stages=["test", "loso"] if skip_cv else ["cv", "test", "loso"],
        phase2_blocks=phase2_blocks,
        full_finetune=full_finetune if full_finetune else None,
        out_root="/data/results/all_models",
        phase1_out_root="/data/results/all_models",
        train_path="/data/data/trainSINGLE.npz",
        val_path="/data/data/valSINGLE.npz",
        test_path="/data/data/testSINGLE.npz",
        commit_fn=volume.commit,
    )
    volume.commit()   # final commit


@app.local_entrypoint()
def main(phase2: bool = False, full_finetune: bool = False, skip_cv: bool = False):
    run.remote(
        phase2_blocks=2 if phase2 else 0,
        full_finetune=full_finetune,
        skip_cv=skip_cv,
    )
