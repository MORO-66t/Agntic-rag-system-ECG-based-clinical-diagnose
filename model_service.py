# model_service.py

import numpy as np
import tensorflow as tf


class ECGModelService:

    CLASS_MAP = {
        0: "N",
        1: "S",
        2: "V",
        3: "F",
        4: "Q"
    }

    def __init__(self, model_path):

        print("Loading ECG model...")

        self.model = tf.keras.models.load_model(
            model_path,
            compile=False,
            safe_mode=False
        )

        print("ECG model loaded.")

        print(
            "Input Shape:",
            self.model.input_shape
        )

        print(
            "Output Shape:",
            self.model.output_shape
        )

    def predict_single(
        self,
        signal
    ):

        signal = np.asarray(
            signal,
            dtype=np.float32
        )

        signal = signal.reshape(
            1,
            187,
            1
        )

        probs = self.model.predict(
            signal,
            verbose=0
        )[0]

        label_idx = int(
            np.argmax(probs)
        )

        confidence = float(
            np.max(probs)
        )

        return {

            "predicted_label":
                label_idx,

            "predicted_class":
                self.CLASS_MAP[label_idx],

            "prediction_confidence":
                confidence,

            "probabilities":
                probs.tolist()
        }

    def predict_batch(
        self,
        batch_signals,
        batch_metadata=None
    ):

        if len(batch_signals) == 0:

            return []

        X_batch = np.asarray(
            batch_signals,
            dtype=np.float32
        )

        X_batch = np.expand_dims(
            X_batch,
            axis=-1
        )

        predictions = self.model.predict(
            X_batch,
            verbose=0
        )

        results = []

        for i in range(
            len(predictions)
        ):

            probs = predictions[i]

            label_idx = int(
                np.argmax(probs)
            )

            confidence = float(
                np.max(probs)
            )

            result = {

                "predicted_label":
                    label_idx,

                "predicted_class":
                    self.CLASS_MAP[label_idx],

                "prediction_confidence":
                    confidence,

                "probabilities":
                    probs.tolist()
            }

            if batch_metadata:

                result.update(
                    batch_metadata[i]
                )

            results.append(
                result
            )

        return results