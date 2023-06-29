import argparse
import os
import pandas as pd
from src.utils.common import read_config
from src.utils.data_mgmt import get_data
from src.utils.model import create_model, save_model
from src.utils.plot import save_plot
from predict import predict
from src.utils.callback import callback


def training(config_path):
    # Reading Config Files
    config = read_config(config_path)
    validation_datasize = config["params"]["validation_data_size"]
    LOSS_FUNCTION = config["params"]["loss_function"]
    OPTIMIZER = config["params"]["optimizer"]
    METRICS = config["params"]["metrics"]
    NUM_CLASSES = config["params"]["no_classes"]
    EPOCHS = config["params"]["epochs"]
    artifact_dir = config["artifacts"]["artifact_dir"]
    model_dir = config["artifacts"]["model_dir"]
    model_name = config["artifacts"]["model_name"]
    plots_dir = config["artifacts"]["plots_dir"]

    # Data Gathering
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = get_data(validation_datasize)
    VALIDATION_SET = (X_valid, y_valid)

    # Creating Model
    model = create_model(LOSS_FUNCTION, OPTIMIZER, METRICS, NUM_CLASSES)

    # Creating Callback/logs
    CALL_BACK = callback(config, X_train)
    # Training data to model
    history = model.fit(X_train, y_train, epochs=EPOCHS,
                        validation_data=VALIDATION_SET, callbacks=CALL_BACK)
    # Saving Model
    model_dir_path = os.path.join(artifact_dir, model_dir)
    os.makedirs(model_dir_path, exist_ok=True)
    file_name_model = save_model(model, model_name, model_dir_path)

    # Predicting on test data
    predict(os.path.join(model_dir_path, file_name_model), X_test, y_test)

    # Plotting Training Graph
    plot_dir = os.path.join(artifact_dir, plots_dir)
    os.makedirs(plot_dir, exist_ok=True)
    plot_dir = os.path.join(plot_dir, file_name_model)
    save_plot(pd.DataFrame(history.history), plot_dir)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="C:\\Users\\Rupam\\Desktop\\Work\\Ineuron\\DL\\config.yaml")

    parsed_arg = args.parse_args()
    training(config_path=parsed_arg.config)
