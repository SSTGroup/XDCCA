import os
import pickle as pkl
from abc import ABC

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize
from tqdm.auto import tqdm

from XDCCA.algorithms.losses_metrics import (
    EmptyWatchdog,
    MetricDict,
)
from XDCCA.algorithms.tf_summary import TensorboardWriter
from XDCCA.algorithms.utils import logdir_update_from_params
from XDCCA.architecture.encoder import (
    MVEncoder,
    ConvMVEncoder,
)


class Experiment(ABC):
    """
    Experiment meta class
    """

    def __init__(
        self,
        architecture,
        dataprovider,
        shared_dim,
        optimizer,
        log_dir,
        summary_writer,
        eval_epochs,
        watchdog,
        val_default_value=0.0,
    ):
        self.architecture = architecture
        self.dataprovider = dataprovider
        self.optimizer = optimizer
        self.summary_writer = self.create_summary_writer(summary_writer, log_dir)
        self.log_dir = self.summary_writer.dir
        self.shared_dim = shared_dim
        self.watchdog = watchdog
        self.moving_metrics = self.get_moving_metrics()
        self.eval_epochs = eval_epochs

        self.epoch = 1
        self.continue_training = True
        self.best_val = val_default_value
        self.best_val_view0 = val_default_value
        self.best_val_view1 = val_default_value
        self.best_val_avg = val_default_value
        # Convergence criteria
        self.loss_threshold = 0.0
        self.prev_loss = 1e5
        self.prev_epoch = 0

    def get_moving_metrics(self):
        raise NotImplementedError

    def create_summary_writer(self, summary_writer, log_dir):
        # Define writer for tensorboard summary
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        return summary_writer(log_dir)

    def train_multiple_epochs(self, num_epochs):
        # Load training data once

        # Iterate over epochs
        for epoch in tqdm(range(num_epochs), desc="Epochs"):
            # Train one epoch
            self.train_single_epoch()

            if not self.continue_training:
                break

        self.save_weights("latest")

    def save_weights(self, subdir=None):
        if subdir is not None:
            save_path = os.path.join(self.log_dir, subdir)
            if not os.path.exists(save_path):
                os.mkdir(save_path)
        else:
            save_path = self.log_dir

        self.architecture.save_weights(filepath=save_path)

    def load_weights_from_log(self, subdir=None):
        if subdir is not None:
            save_path = os.path.join(self.log_dir, subdir)
        else:
            save_path = self.log_dir

        self.architecture.load_weights(filepath=save_path)

    def load_best(self):
        self.architecture.load_weights(filepath=self.log_dir)

    def train_single_epoch(self):
        # for data in tqdm(training_data, desc='Batch', leave=False):
        for data in self.dataprovider.training_data:
            with tf.GradientTape() as tape:
                # Feed forward
                network_output = self.architecture(inputs=data, training=True)
                # Compute loss
                loss = self.compute_loss(network_output, data)

            # Compute gradients
            gradients = tape.gradient(loss, self.architecture.trainable_variables)
            # Apply gradients
            self.optimizer.apply_gradients(
                zip(gradients, self.architecture.trainable_variables)
            )

        # Write metric summary
        self.log_metrics()

        # Increase epoch counter
        self.epoch += 1

        if self.epoch % 100 == 0:
            self.prev_loss = self.prev_loss + 1e5
            next_loss = loss + 1e5
            if tf.abs(self.prev_loss - next_loss) < self.loss_threshold:
                self.continue_training = False
            self.prev_loss = loss
            self.prev_epoch = self.epoch

    def predict(self, data_to_predict):
        outputs = MetricDict()
        for data in data_to_predict:
            network_output = self.architecture(inputs=data, training=True)
            outputs.update(network_output)

        return outputs.output()

    def save(self):
        self.architecture.save(self.log_dir)

    def compute_loss(self, network_output, data):
        raise NotImplementedError

    def log_metrics(self):
        raise NotImplementedError


class DeepCCAExperiment(Experiment):
    """
    Experiment class for DeepCCA
    """

    def __init__(
        self,
        log_dir,
        encoder_config_v1,
        encoder_config_v2,
        dataprovider,
        shared_dim,
        lambda_l2=0,
        cca_reg=0,
        eval_epochs=10,
        optimizer=None,
        val_default_value=0.0,
        watchdog=None,
    ):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()

        if watchdog is None:
            watchdog = EmptyWatchdog()

        architecture = MVEncoder(
            encoder_config_v1=encoder_config_v1,
            encoder_config_v2=encoder_config_v2,
            cca_reg=cca_reg,
            num_shared_dim=shared_dim,
        )

        log_dir = logdir_update_from_params(
            log_dir=log_dir,
            shared_dim=shared_dim,
            num_neurons=encoder_config_v1[0][0],
            lambda_l1=0.0,
            lambda_l2=lambda_l2,
        )

        super(DeepCCAExperiment, self).__init__(
            architecture=architecture,
            dataprovider=dataprovider,
            shared_dim=shared_dim,
            optimizer=optimizer,
            log_dir=log_dir,
            summary_writer=TensorboardWriter,
            eval_epochs=eval_epochs,
            watchdog=watchdog,
            val_default_value=val_default_value,
        )

        # Dimensions and lambdas
        self.shared_dim = shared_dim
        self.lambda_l2 = lambda_l2

    def compute_loss(self, network_output, data):
        # Compute CCA loss
        ccor = network_output["ccor"]
        cca_loss = -1 * tf.reduce_sum(ccor) / len(ccor)

        l2_loss = self.architecture.get_l2()
        l2_loss *= self.lambda_l2

        loss = cca_loss + l2_loss

        if self.epoch % 10 == 0:
            self.summary_writer.write_scalar_summary(
                epoch=self.epoch,
                list_of_tuples=[
                    (loss, "Loss/Total"),
                    (cca_loss, "Loss/CCA"),
                    (l2_loss, "Loss/L2"),
                    (self.shared_dim, "MovingMean/Dimensions"),
                ],
            )
        return loss


class ConvDeepCCAExperiment(DeepCCAExperiment):
    """
    Experiment class for DeepCCAE
    """

    def __init__(
        self,
        log_dir,
        encoder_config_v1,
        encoder_config_v2,
        dataprovider,
        shared_dim,
        lambda_l2=0,
        cca_reg=0,
        eval_epochs=10,
        optimizer=None,
        val_default_value=0.0,
    ):
        if optimizer is None:
            optimizer = tf.keras.optimizers.Adam()

        architecture = ConvMVEncoder(
            encoder_config_v1=encoder_config_v1,
            encoder_config_v2=encoder_config_v2,
            cca_reg=cca_reg,
            num_shared_dim=shared_dim,
        )

        log_dir = logdir_update_from_params(
            log_dir=log_dir,
            shared_dim=shared_dim,
            num_neurons=-1,
            lambda_l1=0.0,
            lambda_l2=lambda_l2,
        )

        super(DeepCCAExperiment, self).__init__(
            architecture=architecture,
            dataprovider=dataprovider,
            shared_dim=shared_dim,
            optimizer=optimizer,
            log_dir=log_dir,
            summary_writer=TensorboardWriter,
            eval_epochs=eval_epochs,
            watchdog=EmptyWatchdog(),
            val_default_value=val_default_value,
        )

        # Dimensions and lambdas
        self.shared_dim = shared_dim
        self.lambda_l2 = lambda_l2
