"""MLCube handler file"""
import typer
from train import run_train

app = typer.Typer()


@app.command("train")
def train(
    data_path: str = typer.Option(..., "--data_path"),
    config_file: str = typer.Option(..., "--config_file"),
    checkpoint_path: str = typer.Options(..., "--checkpoint_path")
    # Provide additional parameters as described in the mlcube.yaml file
    # e.g. model weights:
    # weights: str = typer.Option(..., "--weights"),
):
    run_train(data_path, config_file, checkpoint_path)


@app.command("infer")
def infer(
    data_path: str = typer.Option(..., "--data_path"),
    parameters_file: str = typer.Option(..., "--parameters_file"),
    output_path: str = typer.Option(..., "--output_path"),
    # Provide additional parameters as described in the mlcube.yaml file
    # e.g. model weights:
    # weights: str = typer.Option(..., "--weights"),
):
    # Modify the infer command as needed
    raise NotImplementedError("The infer method is not yet implemented")


@app.command("hotfix")
def hotfix():
    # NOOP command for typer to behave correctly. DO NOT REMOVE OR MODIFY
    pass


if __name__ == "__main__":
    app()
