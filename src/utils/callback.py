import os
import numpy as np
import tensorflow as tf
from src.utils.model import get_unique_filename


def callback(config, X_train):
    logs_dir = os.path.join(config["artifacts"]["artifact_dir"], config["artifacts"]["logs_dir"],
                            get_unique_filename(config["artifacts"]["model_name"]))
    os.makedirs(logs_dir, exist_ok=True)
    file_writer = tf.summary.create_file_writer(logdir=logs_dir)
    with file_writer.as_default():
        images = np.reshape(X_train[10:30], (-1, 28, 28, 1))
        tf.summary.image("20 handwritten digit samples", images, max_outputs=25, step=0)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=logs_dir)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=config["params"]["patience"],
                                                         restore_best_weights=config["params"]["restore_best_weights"])

    checkpoint_path = os.path.join(config["artifacts"]["artifact_dir"], config["artifacts"]["checkpoints_dir"])
    os.makedirs(checkpoint_path, exist_ok=True)
    CKPT_model_name = os.path.join(checkpoint_path, config["artifacts"]["model_name"])
    checkpointing_cb = tf.keras.callbacks.ModelCheckpoint(CKPT_model_name,
                                                          save_best_only=config["params"]["save_best_only"])

    return [tensorboard_cb, early_stopping_cb, checkpointing_cb]
