"""MLCube handler file"""
import typer
from train import run_train
from infer import run_inference
from metrics import run_metrics

app = typer.Typer()


@app.command("train")
def train(
    data_dir: str = typer.Option(..., "--data-dir"),
    config_file: str = typer.Option(..., "--config_file"),
    ckpt_folder: str = typer.Option(..., "--ckpt_folder")
    # Provide additional parameters as described in the mlcube.yaml file
    # e.g. model weights:
    # weights: str = typer.Option(..., "--weights"),
):
    run_train(data_dir, ckpt_folder, config_file)


@app.command("infer")
def infer(
    data_path: str = typer.Option(..., '--data_path'),
    parameters_file: str = typer.Option(..., '--parameters_file'),
    weights: str = typer.Option(..., '--weights'),
    output_path: str = typer.Option(..., '--output_path')
    # Provide additional parameters as described in the mlcube.yaml file
    # e.g. model weights:
    # weights: str = typer.Option(..., "--weights"),
):
    # Modify the infer command as needed
    run_inference(data_path, parameters_file, weights, output_path)


@app.command("metrics")
def metrics(
    data_path: str = typer.Option(..., '--data_path'),
    weights: str = typer.Option(..., '--weights'),
    parameters_file: str = typer.Option(..., '--parameters_file'),
    output_path: str = typer.Option(..., '--output_path')
    # Provide additional parameters as described in the mlcube.yaml file
    # e.g. model weights:
    # weights: str = typer.Option(..., "--weights"),
):
    # Modify the infer command as needed
    run_metrics(data_path, weights, parameters_file, output_path)


@app.command("hotfix")
def hotfix():
    # NOOP command for typer to behave correctly. DO NOT REMOVE OR MODIFY
    pass


if __name__ == "__main__":
    app()
