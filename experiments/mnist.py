import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.manifold import TSNE
from tqdm.auto import tqdm

from XDCCA.algorithms.clustering import kmeans_clustering_acc
from XDCCA.algorithms.losses_metrics import (
    MetricDict,
    MovingMetric,
)
from XDCCA.experiments.template import (
    ConvDeepCCAExperiment,
    Experiment,
)


class MNISTExperiment(Experiment):
    def get_moving_metrics(self):
        cor_movmetr = {
            "cor_"
            + str(num): MovingMetric(
                window_length=5, history_length=10, fun=tf.math.reduce_mean
            )
            for num in range(self.shared_dim)
        }

        acc_movmetr = {
            "acc_v0": MovingMetric(
                window_length=5, history_length=10, fun=tf.math.reduce_mean
            ),
            "acc_v1": MovingMetric(
                window_length=5, history_length=10, fun=tf.math.reduce_mean
            ),
        }

        return {**cor_movmetr, **acc_movmetr}

    def log_metrics(self):
        self.watchdog.decrease_counter()

        if self.epoch % 10 == 0:
            # Compute correlation values on training data
            training_outp = self.predict(self.dataprovider.training_data)
            ccor = training_outp["ccor"]

            l2 = self.architecture.get_l2()

            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (ccor[i], "Correlations/" + str(i)) for i in range(self.shared_dim)
                ]
                + [(l2, "Regularization/L2")],
            )

        if self.epoch % self.eval_epochs == 0:
            acc_v0 = self.compute_clustering_accuracy(view="view0", split="eval")
            self.moving_metrics["acc_v0"].update_window(acc_v0)
            smoothed_acc_v0 = self.moving_metrics["acc_v0"].get_metric()

            acc_v1 = self.compute_clustering_accuracy(view="view1", split="eval")
            self.moving_metrics["acc_v1"].update_window(acc_v1)
            smoothed_acc_v1 = self.moving_metrics["acc_v1"].get_metric()

            acc_avg = (acc_v0 + acc_v1) / 2
            smoothed_acc_avg = (smoothed_acc_v0 + smoothed_acc_v1) / 2

            if smoothed_acc_v0 > self.best_val_view0:
                self.best_val_view0 = smoothed_acc_v0
                self.save_weights(subdir="view0")

            if smoothed_acc_v1 > self.best_val_view1:
                self.best_val_view1 = smoothed_acc_v1
                self.save_weights(subdir="view1")

            if smoothed_acc_avg > self.best_val_avg:
                self.best_val_avg = smoothed_acc_avg
                self.save_weights(subdir="avg")

            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (acc_v0, "Accuracy/View0"),
                    (acc_v1, "Accuracy/View1"),
                    (acc_avg, "Accuracy/Average"),
                    (smoothed_acc_v0, "AccuracySmoothed/View0"),
                    (smoothed_acc_v1, "AccuracySmoothed/View1"),
                    (smoothed_acc_avg, "AccuracySmoothed/Average"),
                ],
            )

    def compute_clustering_accuracy(self, view="view0", split="eval"):
        assert view in ["view0", "view1"]
        assert split in ["eval", "test"]
        if split == "eval":
            data_for_acc = self.dataprovider.eval_data
        else:
            data_for_acc = self.dataprovider.test_data

        outputs_met, labels_met = MetricDict(), MetricDict()
        for data in data_for_acc:
            outputs_met.update(self.architecture(inputs=data, training=False))
            labels_met.update(dict(labels=data["labels"].numpy()))

        netw_output = outputs_met.output()
        labels = labels_met.output()["labels"]

        if view == "view0":
            latent_repr = netw_output["cca_view_0"]
        elif view == "view1":
            latent_repr = netw_output["cca_view_1"]

        return kmeans_clustering_acc(
            data_points=latent_repr,
            labels=labels,
            num_classes=self.dataprovider.num_classes,
        )


class MNISTConvDeepCCAExperiment(ConvDeepCCAExperiment, MNISTExperiment):
    pass
