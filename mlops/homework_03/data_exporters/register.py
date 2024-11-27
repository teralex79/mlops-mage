import pickle
import mlflow

EXPERIMENT_NAME = "nyc-yellow-taxi-lin-reg-exp"

mlflow.set_tracking_uri('http://localhost:5001')
mlflow.set_experiment(EXPERIMENT_NAME)

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def dump_pickle(filename: str):
    with open(filename, "wb") as f_out:
        return pickle.dump(f_out)


def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    dv, lr = data
    with mlflow.start_run():
        dump_pickle(dv, 'dv.pkl')
        mlflow.log_artifact('dv.pkl')
        mlflow.log_model(lr, 'model')



