import modal

app = modal.App("crossvalEN")

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
    .add_local_python_source("crossvalEN")
)

@app.function(
    image=image,
    gpu="A10G",
    timeout=60 * 60 * 12,
    volumes={"/data": volume},
)
def run():
#     import os
#     os.chdir("/root/project")

    import crossvalEN
    crossvalEN.main()

# @app.function(
#     image=image,
#     gpu="A10G",
#     volumes={"/data": volume},
# )
# def run():
#     import os

#     print("=== ROOT ===")
#     print(os.listdir("/"))

#     print("=== /data ===")
#     print(os.listdir("/data"))

#     try:
#         print("=== /data/data ===")
#         print(os.listdir("/data/data"))
#     except Exception as e:
#         print(e)


@app.local_entrypoint()
def main():
    run.remote()