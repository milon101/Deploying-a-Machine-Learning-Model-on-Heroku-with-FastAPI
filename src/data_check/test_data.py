import pytest
import pandas as pd
import wandb

def pytest_addoption(parser):
    parser.addoption("--csv", action="store")

@pytest.fixture(scope='session')
def data(request):
    run = wandb.init(job_type="data_tests", resume=True)

    # Download input artifact. This will also note that this script is using this
    # particular version of the artifact
    data_path = run.use_artifact(request.config.option.csv).file()

    if data_path is None:
        pytest.fail("You must provide the --csv option on the command line")

    df = pd.read_csv(data_path)

    return df

def test_null(data):
    """
    Check Data for no null values
    """
    assert data.shape == data.dropna().shape

def test_question_mark(data):
    """
    Check data for no question marks
    """
    assert '?' not in data.values