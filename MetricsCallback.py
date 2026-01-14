import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    ConfusionMatrixDisplay
)
import tensorflow as tf

class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self, val_data, class_names, output_dir="metrics"):
        super().__init__()
        self.val_data = val_data
        self.class_names = class_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.history = []

    def on_epoch_end(self, epoch, logs=None):
        y_true = []
        y_pred = []

        # Prédiction sur tout le dataset de validation
        for images, labels in self.val_data:
            preds = self.model.predict(images, verbose=0)
            y_true.extend(labels.numpy())
            y_pred.extend(np.argmax(preds, axis=1))

        # Métriques globales (macro = toutes les classes comptent pareil)
        precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
        recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Sauvegarde métriques
        self.history.append({
            "epoch": epoch + 1,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1
        })

        pd.DataFrame(self.history).to_csv(
            os.path.join(self.output_dir, "metrics_per_epoch.csv"),
            index=False
        )

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(10, 10))
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=self.class_names
        )
        disp.plot(ax=ax, xticks_rotation=90, cmap="Blues", colorbar=False)
        plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                self.output_dir,
                f"confusion_matrix_epoch_{epoch + 1}.png"
            )
        )
        plt.close()
