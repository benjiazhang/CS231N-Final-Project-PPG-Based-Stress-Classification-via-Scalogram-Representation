import modal

app = modal.App("efficientnet")

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
    )
    .add_local_python_source("efficientnet")
)

@app.function(
    image=image,
    gpu="A10",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
)
def run():

    import efficientnet
    efficientnet.main()
    
@app.local_entrypoint()
def main():
    run.remote()