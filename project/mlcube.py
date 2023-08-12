"""MLCube handler file"""
import typer
from train import run_train
from infer import run_inference
from metrics import run_metrics

app = typer.Typer()


@app.command("train")
def train(
    config_file: str = typer.Option(..., "--config_file")
    # Provide additional parameters as described in the mlcube.yaml file
    # e.g. model weights:
    # weights: str = typer.Option(..., "--weights"),
):
    run_train(config_file)


@app.command("infer")
def infer(
    parameters: str = typer.Option(..., '--parameters')
    # Provide additional parameters as described in the mlcube.yaml file
    # e.g. model weights:
    # weights: str = typer.Option(..., "--weights"),
):
    # Modify the infer command as needed
    run_inference(parameters)


@app.command("metrics")
def metrics(
    parameters: str = typer.Option(..., '--parameters')
    # Provide additional parameters as described in the mlcube.yaml file
    # e.g. model weights:
    # weights: str = typer.Option(..., "--weights"),
):
    # Modify the infer command as needed
    run_metrics(parameters)


@app.command("hotfix")
def hotfix():
    # NOOP command for typer to behave correctly. DO NOT REMOVE OR MODIFY
    pass


if __name__ == "__main__":
    app()
