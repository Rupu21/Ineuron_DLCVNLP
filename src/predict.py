import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


def predict(model_path, X_test, y_test):
    model = tf.keras.models.load_model(model_path)
    y_pred = np.argmax(model.predict(X_test), axis=-1)
    print(f"Accuracy :  {accuracy_score(y_test, y_pred) * 100}")
    # for data, pred, actual in zip(X_test, y_pred, y_test):
    #     plt.imshow(data, cmap="binary")
    #     plt.title(f"Predicted: {pred}, Actual: {actual}")
    #     plt.axis('off')
    #     plt.show()
    #     print("---" * 20)
