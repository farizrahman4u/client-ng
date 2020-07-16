import wandb
from wandb.cli import cli
import traceback
import platform
import os


def test_artifact_download(runner, git_repo, mock_server):
    result = runner.invoke(cli.artifact, ["get", "test/mnist:v0"])
    print(result.output)
    print(result.exception)
    print(traceback.print_tb(result.exc_info[2]))
    assert result.exit_code == 0
    assert "Downloading dataset artifact" in result.output
    path = os.path.join(".", "artifacts", "mnist:v0")
    if platform.system() == "Windows":
        path = path.replace(":", "-")
    assert "Artifact downloaded to %s" % path in result.output
    assert os.path.exists(path)


def test_artifact_upload(runner, git_repo, mock_server):
    with open("artifact.txt", "w") as f:
        f.write("My Artifact")
    os.environ["WANDB_MODE"] = "mock"
    wandb._IS_INTERNAL_PROCESS = False # TODO: this is rather unfortunate and shouldn't be needed
    result = runner.invoke(cli.artifact, ["put", "artifact.txt", "-n", "test/simple"])
    print(result.output)
    print(result.exception)
    print(traceback.print_tb(result.exc_info[2]))
    assert result.exit_code == 0
    assert "Uploading file artifact.txt to:" in result.output
    assert 'artifact = run.use_artifact("vanpelt/test/simple:v0")' in result.output


def test_artifact_ls(runner, git_repo, mock_server):
    result = runner.invoke(cli.artifact, ["ls", "test"])
    print(result.output)
    print(result.exception)
    print(traceback.print_tb(result.exc_info[2]))
    assert result.exit_code == 0
    assert "9KB" in result.output
    assert "mnist:v2" in result.output