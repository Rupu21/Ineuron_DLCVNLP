import tensorflow as tf


def create_model(LOSS_FUNCTION, OPTIMIZER, METRICS ,NUM_CLASSES):
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28, 28], name="inputLayer"),
              tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
              tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
              tf.keras.layers.Dense(NUM_CLASSES, activation="softmax", name="outputLayer")]

    model = tf.keras.models.Sequential(LAYERS)
    print(model.summary())
    model.compile(loss=LOSS_FUNCTION,
                  optimizer=OPTIMIZER,
                  metrics=METRICS)

    return model
