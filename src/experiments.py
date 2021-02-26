import tensorflow as tf
import numpy as np
from network_archs import JointEncoder, AutoEncoder, MLP
from PIL import Image
from os import makedirs
from welford import Welford


class Experiment:
    def __init__(self):
        self.writer = tf.summary.create_file_writer("./summaries")
        self.reconstructions_path = './reconstructions'
        makedirs(self.reconstructions_path)

    def log_proprioception_reconstruction_loss(self, loss_1, step):
        position_slice = slice(0, 7)
        velocity_slice = slice(7, 14)
        tf.summary.scalar("jointencoder/loss_proprioception", tf.reduce_mean(loss_1), step=step)
        tf.summary.scalar("jointencoder/loss_position", tf.reduce_mean(loss_1[position_slice]), step=step)
        tf.summary.scalar("jointencoder/loss_velocity", tf.reduce_mean(loss_1[velocity_slice]), step=step)

    def log_readout_loss(self, loss, step):
        arm0_end_eff_slice = slice(0, 3)
        arm0_positions_slice = slice(3, 10)
        arm0_velocities_slice = slice(10, 17)
        arm1_end_eff_slice = slice(17, 20)
        arm1_positions_slice = slice(20, 27)
        arm1_velocities_slice = slice(27, 34)
        tf.summary.scalar("readout/loss", tf.reduce_mean(loss), step=step)
        tf.summary.scalar("readout/arm0_end_eff", tf.reduce_mean(loss[arm0_end_eff_slice]), step=step)
        tf.summary.scalar("readout/arm0_positions", tf.reduce_mean(loss[arm0_positions_slice]), step=step)
        tf.summary.scalar("readout/arm0_velocities", tf.reduce_mean(loss[arm0_velocities_slice]), step=step)
        tf.summary.scalar("readout/arm1_end_eff", tf.reduce_mean(loss[arm1_end_eff_slice]), step=step)
        tf.summary.scalar("readout/arm1_positions", tf.reduce_mean(loss[arm1_positions_slice]), step=step)
        tf.summary.scalar("readout/arm1_velocities", tf.reduce_mean(loss[arm1_velocities_slice]), step=step)

    def print_readout_loss(self, loss):
        arm0_end_eff_slice = slice(0, 3)
        arm0_positions_slice = slice(3, 10)
        arm0_velocities_slice = slice(10, 17)
        arm1_end_eff_slice = slice(17, 20)
        arm1_positions_slice = slice(20, 27)
        arm1_velocities_slice = slice(27, 34)
        print("readout loss: {:.4f}    eef {:.4f}  pos {:.4f}  vel {:.4f}  eef {:.4f}  pos {:.4f}  vel {:.4f}".format(
            tf.reduce_mean(loss).numpy(),
            tf.reduce_mean(loss[arm0_end_eff_slice]).numpy(),
            tf.reduce_mean(loss[arm0_positions_slice]).numpy(),
            tf.reduce_mean(loss[arm0_velocities_slice]).numpy(),
            tf.reduce_mean(loss[arm1_end_eff_slice]).numpy(),
            tf.reduce_mean(loss[arm1_positions_slice]).numpy(),
            tf.reduce_mean(loss[arm1_velocities_slice]).numpy(),
        ))

    def log_image_autoencoder_loss(self, loss_0, step):
        stop = loss_0.shape[1] // 2
        tf.summary.scalar("autoencoder/loss_image", tf.reduce_mean(loss_0), step=step)
        tf.summary.scalar("autoencoder/loss_image_left", tf.reduce_mean(loss_0[:, :stop]), step=step)
        tf.summary.scalar("autoencoder/loss_image_right", tf.reduce_mean(loss_0[:, stop:]), step=step)

    def print_image_autoencoder_loss(self, loss_0):
        stop = loss_0.shape[1] // 2
        print("autoencoder loss: {:.4f} ({:.4f}/{:.4f})".format(
            tf.reduce_mean(loss_0).numpy(),
            tf.reduce_mean(loss_0[:, :stop]).numpy(),
            tf.reduce_mean(loss_0[:, stop:]).numpy(),
        ))

    def print_image_proprioception_losses(self, loss_0, loss_1):
        position_slice = slice(0, 7)
        velocity_slice = slice(7, 14)
        stop = loss_0.shape[1] // 2
        print("image loss: {:.4f} ({:.4f}/{:.4f})    proprioception loss: {:.4f} ({:.4f}/{:.4f})".format(
            tf.reduce_mean(loss_0).numpy(),
            tf.reduce_mean(loss_0[:, :stop]).numpy(),
            tf.reduce_mean(loss_0[:, stop:]).numpy(),
            tf.reduce_mean(loss_1).numpy(),
            tf.reduce_mean(loss_1[position_slice]).numpy(),
            tf.reduce_mean(loss_1[velocity_slice]).numpy(),
        ))

    def print_cross_modality_losses(self, loss_0, loss_1):
        print("loss 0 -> 1 : {:.4f}   loss 1 -> 0 : {:.4f}".format(tf.reduce_mean(loss_0).numpy(), tf.reduce_mean(loss_1).numpy()))

    def train_jointencoder(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_jointencoder)
        with self.writer.as_default():
            for inp_0, inp_1 in dataset:
                loss_0, loss_1 = self.train_jointencoder_batch(inp_0, inp_1)
                self.log_image_reconstruction_loss(loss_0, step)
                self.log_proprioception_reconstruction_loss(loss_1, step)
                self.print_image_proprioception_losses(loss_0, loss_1)
                step += 1

    def train_readout(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_readout)
        with self.writer.as_default():
            for inp_0, inp_1, target in dataset:
                loss = self.train_readout_batch(inp_0, inp_1, target)
                self.log_readout_loss(loss, step)
                self.print_readout_loss(loss)
                step += 1


    def train_autoencoder(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_autoencoder)
        with self.writer.as_default():
            for inp in dataset:
                loss = self.train_autoencoder_batch(inp)
                self.log_image_autoencoder_loss(loss, step)
                self.print_image_autoencoder_loss(loss)
                step += 1

    def train_cross_modality(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_cross_modality)
        with self.writer.as_default():
            for inp_0, inp_1 in dataset:
                loss_0, loss_1 = self.train_cross_modality_batch(inp_0, inp_1)
                self.log_cross_modality_losses(loss_0, loss_1, step)
                self.print_cross_modality_losses(loss_0, loss_1)
                step += 1

    def z_score_encoding(self, dataset):
        welford = Welford(shape=(self.autoencoder.encoding_size,))
        for inp in dataset:
            encoding = self.autoencoder(inp, what=['encoding'])['encoding']
            welford(encoding)
        self.encoding_mean = welford.mean
        self.encoding_std = welford.std

    def get_image_and_reconstructions(self, dataset):
        return np.array([(
                inp_0,
                self.get_image_reconstructions(
                    inp_0,
                    inp_1
                )
            )
            for inp_0, inp_1 in dataset
        ])

    def save_image_reconstructions(self, dataset):
        image_reconstructions = self.get_image_and_reconstructions(dataset) # shape = [M, 2, N, height, width, 3]
        batch_size = image_reconstructions.shape[2]
        ratio = 320 // image_reconstructions.shape[3]
        size = (image_reconstructions.shape[4] * ratio, image_reconstructions.shape[3] * ratio)
        for j, batch in enumerate(image_reconstructions):
            for i in range(batch_size):
                frame = np.concatenate([batch[0, i], batch[1, i]], axis=0)
                frame = np.clip(frame * 127.5 + 127.5, 0, 255).astype(np.uint8)
                Image.fromarray(frame).resize(size).save(self.reconstructions_path + '/{:04d}_{:04d}.jpg'.format(j, i))
        error_map = np.mean((image_reconstructions[:, 0] - image_reconstructions[:, 1]) ** 2, axis=(0, 1, -1)) # [height, width]
        mini, maxi = error_map.min(), error_map.max()
        error_map = (error_map - mini) / (maxi - mini)
        error_map = (error_map * 255).astype(np.uint8)
        Image.fromarray(error_map).resize(size).save(self.reconstructions_path + '/error_map_{:.4f}_{:.4f}.jpg'.format(mini, maxi))


class ExperimentOption1(Experiment):
    def log_image_reconstruction_loss(self, loss_0, step):
        stop = loss_0.shape[1] // 2
        tf.summary.scalar("jointencoder/loss_image", tf.reduce_mean(loss_0), step=step)
        tf.summary.scalar("jointencoder/loss_image_left", tf.reduce_mean(loss_0[:, :stop]), step=step)
        tf.summary.scalar("jointencoder/loss_image_right", tf.reduce_mean(loss_0[:, stop:]), step=step)

    def print_image_proprioception_losses(self, loss_0, loss_1):
        stop = loss_0.shape[1] // 2
        position_slice = slice(0, 7)
        velocity_slice = slice(7, 14)
        print("image loss: {:.4f} ({:.4f}/{:.4f})    proprioception loss: {:.4f} ({:.4f}/{:.4f})".format(
            tf.reduce_mean(loss_0).numpy(),
            tf.reduce_mean(loss_0[:, :stop]).numpy(),
            tf.reduce_mean(loss_0[:, stop:]).numpy(),
            tf.reduce_mean(loss_1).numpy(),
            tf.reduce_mean(loss_1[position_slice]).numpy(),
            tf.reduce_mean(loss_1[velocity_slice]).numpy(),
        ))

    def log_cross_modality_losses(self, loss_0, loss_1, step):
        stop = loss_1.shape[1] // 2
        tf.summary.scalar("cross_modality/loss_image", tf.reduce_mean(loss_1), step=step)
        tf.summary.scalar("cross_modality/loss_image_left", tf.reduce_mean(loss_1[:, :stop]), step=step)
        tf.summary.scalar("cross_modality/loss_image_right", tf.reduce_mean(loss_1[:, stop:]), step=step)
        position_slice = slice(0, 7)
        velocity_slice = slice(7, 14)
        tf.summary.scalar("cross_modality/loss_proprioception", tf.reduce_mean(loss_0), step=step)
        tf.summary.scalar("cross_modality/loss_position", tf.reduce_mean(loss_0[position_slice]), step=step)
        tf.summary.scalar("cross_modality/loss_velocity", tf.reduce_mean(loss_0[velocity_slice]), step=step)


class ExperimentOption2(Experiment):
    def log_image_reconstruction_loss(self, loss_0, step):
        tf.summary.scalar("jointencoder/loss_image", tf.reduce_mean(loss_0), step=step)

    def print_image_proprioception_losses(self, loss_0, loss_1):
        position_slice = slice(0, 7)
        velocity_slice = slice(7, 14)
        print("image loss: {:.4f}    proprioception loss: {:.4f} ({:.4f}/{:.4f})".format(
            tf.reduce_mean(loss_0).numpy(),
            tf.reduce_mean(loss_1).numpy(),
            tf.reduce_mean(loss_1[position_slice]).numpy(),
            tf.reduce_mean(loss_1[velocity_slice]).numpy(),
        ))

    def log_cross_modality_losses(self, loss_0, loss_1, step):
        tf.summary.scalar("cross_modality/loss_image", tf.reduce_mean(loss_1), step=step)
        position_slice = slice(0, 7)
        velocity_slice = slice(7, 14)
        tf.summary.scalar("cross_modality/loss_proprioception", tf.reduce_mean(loss_0), step=step)
        tf.summary.scalar("cross_modality/loss_position", tf.reduce_mean(loss_0[position_slice]), step=step)
        tf.summary.scalar("cross_modality/loss_velocity", tf.reduce_mean(loss_0[velocity_slice]), step=step)


class JointEncodingOption1(ExperimentOption1):
    def __init__(self, config):
        super().__init__()
        self.jointencoder = JointEncoder(config.jointencoder)
        self.readout = MLP(config.readout)
        self.n_epoch_jointencoder = config.jointencoder.n_epochs
        self.n_epoch_readout = config.readout.n_epochs

    def train_jointencoder_batch(self, inp_0, inp_1):
        return self.jointencoder.train(inp_0, inp_1)

    def train_readout_batch(self, inp_0, inp_1, target):
        encoding = self.jointencoder(inp_0, inp_1, what=['encoding'])['encoding']
        return self.readout.train(encoding, target)

    def get_image_reconstructions(self, inp_0, inp_1):
        return self.jointencoder(
            inp_0,
            inp_1,
            what=['reconstructions']
        )['reconstruction_0']


class JointEncodingOption2(ExperimentOption2):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = AutoEncoder(config.autoencoder)
        self.jointencoder = JointEncoder(config.jointencoder)
        self.readout = MLP(config.readout)
        self.n_epoch_autoencoder = config.autoencoder.n_epochs
        self.n_epoch_jointencoder = config.jointencoder.n_epochs
        self.n_epoch_readout = config.readout.n_epochs

    def train_autoencoder_batch(self, inp):
        return self.autoencoder.train(inp)

    def train_jointencoder_batch(self, inp_0, inp_1):
        encoding = (self.autoencoder(inp_0, what=['encoding'])['encoding'] - self.encoding_mean) / self.encoding_std
        return self.jointencoder.train(encoding, inp_1)

    def train_readout_batch(self, inp_0, inp_1, target):
        encoding_a = self.autoencoder(inp_0, what=['encoding'])['encoding']
        encoding_a = (self.autoencoder(inp_0, what=['encoding'])['encoding'] - self.encoding_mean) / self.encoding_std
        encoding_b = self.jointencoder(encoding_a, inp_1, what=['encoding'])['encoding']
        return self.readout.train(encoding_b, target)

    def get_image_reconstructions(self, inp_0, inp_1):
        return self.autoencoder.get_reconstruction(
            self.jointencoder(
                (self.autoencoder(
                    inp_0,
                    what=['encoding']
                )['encoding'] - self.encoding_mean) / self.encoding_std,
                inp_1,
                what=['reconstructions']
            )['reconstruction_0'] * self.encoding_std + self.encoding_mean
        )


class CrossModalityOption1(ExperimentOption1):
    def __init__(self, config):
        super().__init__()
        self.mod_0_to_1 = MLP(config.mod_0_to_1)
        self.mod_1_to_0 = MLP(config.mod_1_to_0)
        self.jointencoder = JointEncoder(config.jointencoder)
        self.readout = MLP(config.readout)
        self.n_epoch_cross_modality = config.cross_modality.n_epochs
        self.n_epoch_jointencoder = config.jointencoder.n_epochs
        self.n_epoch_readout = config.readout.n_epochs

    def train_cross_modality_batch(self, inp_0, inp_1):
        loss_0 = self.mod_0_to_1.train(inp_0, inp_1)
        loss_1 = self.mod_1_to_0.train(inp_1, inp_0)
        return loss_0, loss_1

    def train_jointencoder_batch(self, inp_0, inp_1):
        prediction_0 = self.mod_0_to_1(inp_0, what=['output'])['output']
        prediction_1 = self.mod_1_to_0(inp_1, what=['output'])['output']
        return self.jointencoder.train(prediction_1, prediction_0)

    def train_readout_batch(self, inp_0, inp_1, target):
        prediction_0 = self.mod_0_to_1(inp_0, what=['output'])['output']
        prediction_1 = self.mod_1_to_0(inp_1, what=['output'])['output']
        encoding = self.jointencoder(prediction_1, prediction_0, what=['encoding'])['encoding']
        return self.readout.train(encoding, target)

    def get_image_reconstructions(self, inp_0, inp_1):
        return self.jointencoder(
            self.mod_0_to_1(inp_0, what=['output'])['output'],
            self.mod_1_to_0(inp_1, what=['output'])['output'],
            what=['reconstructions']
        )['reconstruction_1']


class CrossModalityOption2(ExperimentOption2):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = AutoEncoder(config.autoencoder)
        self.mod_0_to_1 = MLP(config.mod_0_to_1)
        self.mod_1_to_0 = MLP(config.mod_1_to_0)
        self.jointencoder = JointEncoder(config.jointencoder)
        self.readout = MLP(config.readout)
        self.n_epoch_autoencoder = config.autoencoder.n_epochs
        self.n_epoch_cross_modality = config.cross_modality.n_epochs
        self.n_epoch_jointencoder = config.jointencoder.n_epochs
        self.n_epoch_readout = config.readout.n_epochs

    def train_autoencoder_batch(self, inp):
        return self.autoencoder.train(inp)

    def train_cross_modality_batch(self, inp_0, inp_1):
        encoding = self.autoencoder(inp_0, what=['encoding'])['encoding']
        loss_0 = self.mod_0_to_1.train(encoding, inp_1)
        loss_1 = self.mod_1_to_0.train(inp_1, encoding)
        return loss_0, loss_1

    def train_jointencoder_batch(self, inp_0, inp_1):
        encoding = self.autoencoder(inp_0, what=['encoding'])['encoding']
        prediction_0 = self.mod_0_to_1(encoding, what=['output'])['output']
        prediction_1 = self.mod_1_to_0(inp_1, what=['output'])['output']
        return self.jointencoder.train(prediction_1, prediction_0)

    def train_readout_batch(self, inp_0, inp_1, target):
        encoding_a = self.autoencoder(inp_0, what=['encoding'])['encoding']
        prediction_0 = self.mod_0_to_1(encoding_a, what=['output'])['output']
        prediction_1 = self.mod_1_to_0(inp_1, what=['output'])['output']
        encoding_b = self.jointencoder(prediction_1, prediction_0, what=['encoding'])['encoding']
        return self.readout.train(encoding_b, target)

    def get_image_reconstructions(self, inp_0, inp_1):
        return self.autoencoder.get_reconstruction(
            self.jointencoder(
                self.mod_0_to_1(
                    (
                        self.autoencoder(
                            inp_0,
                            what=['encoding']
                        )['encoding'] - self.encoding_mean
                    ) / self.encoding_std),
                self.mod_1_to_0(inp_1),
                what=['reconstructions']
            )['reconstruction_1'] * self.encoding_std + self.encoding_mean
        )
