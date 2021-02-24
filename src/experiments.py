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


class JointEncodingOption1(Experiment):
    def __init__(self, config):
        super().__init__()
        self.jointencoder = JointEncoder(config.jointencoder)
        self.readout = MLP(config.readout)
        self.n_epoch_jointencoder = config.jointencoder.n_epochs
        self.n_epoch_readout = config.readout.n_epochs

    def train_jointencoder(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_jointencoder)
        with self.writer.as_default():
            for inp_0, inp_1 in dataset:
                loss_0, loss_1 = self.jointencoder.train(inp_0, inp_1)
                print("image loss: {:.4f}    proprioception loss: {:.4f}".format(loss_0.numpy(), loss_1.numpy()))
                tf.summary.scalar("jointencoder_loss_0", loss_0, step=step)
                tf.summary.scalar("jointencoder_loss_1", loss_1, step=step)
                step += 1

    def train_readout(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_readout)
        with self.writer.as_default():
            for inp_0, inp_1, target in dataset:
                encoding = self.jointencoder(inp_0, inp_1, what=['encoding'])['encoding']
                loss = self.readout.train(encoding, target)
                print("readout_loss", loss.numpy())
                tf.summary.scalar("readout_loss", loss, step=step)
                step += 1

    def get_image_reconstructions(self, inp_0, inp_1):
        return self.jointencoder(
            inp_0,
            inp_1,
            what=['reconstructions']
        )['reconstruction_0']


class JointEncodingOption2(Experiment):
    def __init__(self, config):
        super().__init__()
        self.autoencoder = AutoEncoder(config.autoencoder)
        self.jointencoder = JointEncoder(config.jointencoder)
        self.readout = MLP(config.readout)
        self.n_epoch_autoencoder = config.autoencoder.n_epochs
        self.n_epoch_jointencoder = config.jointencoder.n_epochs
        self.n_epoch_readout = config.readout.n_epochs

    def train_autoencoder(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_autoencoder)
        with self.writer.as_default():
            for inp in dataset:
                loss = self.autoencoder.train(inp)
                print("autoencoder loss: {:.4f}".format(loss.numpy()))
                tf.summary.scalar("autoencoder_loss", loss, step=step)
                step += 1

    def z_score_encoding(self, dataset):
        welford = Welford(shape=(self.autoencoder.encoding_size,))
        for inp in dataset:
            encoding = self.autoencoder(inp, what=['encoding'])['encoding']
            welford(encoding)
        self.encoding_mean = welford.mean
        self.encoding_std = welford.std

    def train_jointencoder(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_jointencoder)
        with self.writer.as_default():
            for inp_0, inp_1 in dataset:
                encoding = (self.autoencoder(inp_0, what=['encoding'])['encoding'] - self.encoding_mean) / self.encoding_std
                loss_0, loss_1 = self.jointencoder.train(encoding, inp_1)
                print("image loss: {:.4f}    proprioception loss: {:.4f}".format(loss_0.numpy(), loss_1.numpy()))
                tf.summary.scalar("jointencoder_loss_0", loss_0, step=step)
                tf.summary.scalar("jointencoder_loss_1", loss_1, step=step)
                step += 1

    def train_readout(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_readout)
        with self.writer.as_default():
            for inp_0, inp_1, target in dataset:
                encoding_a = self.autoencoder(inp_0, what=['encoding'])['encoding']
                encoding_a = (self.autoencoder(inp_0, what=['encoding'])['encoding'] - self.encoding_mean) / self.encoding_std
                encoding_b = self.jointencoder(encoding_a, inp_1, what=['encoding'])['encoding']
                loss = self.readout.train(encoding_b, target)
                print("readout_loss", loss.numpy())
                tf.summary.scalar("readout_loss", loss, step=step)
                step += 1

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


class CrossModalityOption1(Experiment):
    def __init__(self, config):
        super().__init__()
        self.mod_0_to_1 = MLP(config.mod_0_to_1)
        self.mod_1_to_0 = MLP(config.mod_1_to_0)
        self.jointencoder = JointEncoder(config.jointencoder)
        self.readout = MLP(config.readout)
        self.n_epoch_cross_modality = config.cross_modality.n_epochs
        self.n_epoch_jointencoder = config.jointencoder.n_epochs
        self.n_epoch_readout = config.readout.n_epochs

    def train_cross_modality(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_cross_modality)
        with self.writer.as_default():
            for inp_0, inp_1 in dataset:
                loss_0 = self.mod_0_to_1.train(inp_0, inp_1)
                loss_1 = self.mod_1_to_0.train(inp_1, inp_0)
                print("loss 0 -> 1 : {:.4f}   loss 1 -> 0 : {:.4f}".format(loss_0.numpy(), loss_1.numpy()))
                tf.summary.scalar("loss_0_to_1", loss_0, step=step)
                tf.summary.scalar("loss_1_to_0", loss_1, step=step)
                step += 1

    def train_jointencoder(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_jointencoder)
        with self.writer.as_default():
            for inp_0, inp_1 in dataset:
                prediction_0 = self.mod_0_to_1(inp_0, what=['output'])['output']
                prediction_1 = self.mod_1_to_0(inp_1, what=['output'])['output']
                loss_0, loss_1 = self.jointencoder.train(prediction_0, prediction_1)
                print("image loss: {:.4f}    proprioception loss: {:.4f}".format(loss_0.numpy(), loss_1.numpy()))
                tf.summary.scalar("jointencoder_loss_0", loss_0, step=step)
                tf.summary.scalar("jointencoder_loss_1", loss_1, step=step)
                step += 1

    def train_readout(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_readout)
        with self.writer.as_default():
            for inp_0, inp_1, target in dataset:
                prediction_0 = self.mod_0_to_1(inp_0, what=['output'])['output']
                prediction_1 = self.mod_1_to_0(inp_1, what=['output'])['output']
                encoding = self.jointencoder(prediction_0, prediction_1, what=['encoding'])['encoding']
                loss = self.readout.train(encoding, target)
                print("readout_loss", loss.numpy())
                tf.summary.scalar("readout_loss", loss, step=step)
                step += 1

    def get_image_reconstructions(self, inp_0, inp_1):
        return self.jointencoder(
            self.mod_0_to_1(inp_0, what=['output'])['output'],
            self.mod_1_to_0(inp_1, what=['output'])['output'],
            what=['reconstructions']
        )['reconstruction_1']


class CrossModalityOption2(Experiment):
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

    def train_autoencoder(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_autoencoder)
        with self.writer.as_default():
            for inp in dataset:
                loss = self.autoencoder.train(inp)
                print("autoencoder loss: {:.4f}".format(loss.numpy()))
                tf.summary.scalar("autoencoder_loss", loss, step=step)
                step += 1

    def train_cross_modality(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_cross_modality)
        with self.writer.as_default():
            for inp_0, inp_1 in dataset:
                encoding = self.autoencoder(inp_0, what=['encoding'])['encoding']
                loss_0 = self.mod_0_to_1.train(encoding, inp_1)
                loss_1 = self.mod_1_to_0.train(inp_1, encoding)
                print("loss 0 -> 1 : {:.4f}   loss 1 -> 0 : {:.4f}".format(loss_0.numpy(), loss_1.numpy()))
                tf.summary.scalar("loss_0_to_1", loss_0, step=step)
                tf.summary.scalar("loss_1_to_0", loss_1, step=step)
                step += 1

    def train_jointencoder(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_jointencoder)
        with self.writer.as_default():
            for inp_0, inp_1 in dataset:
                encoding = self.autoencoder(inp_0, what=['encoding'])['encoding']
                prediction_0 = self.mod_0_to_1(encoding, what=['output'])['output']
                prediction_1 = self.mod_1_to_0(inp_1, what=['output'])['output']
                loss_0, loss_1 = self.jointencoder.train(prediction_0, prediction_1)
                print("image loss: {:.4f}    proprioception loss: {:.4f}".format(loss_0.numpy(), loss_1.numpy()))
                tf.summary.scalar("jointencoder_loss_0", loss_0, step=step)
                tf.summary.scalar("jointencoder_loss_1", loss_1, step=step)
                step += 1

    def train_readout(self, dataset):
        step = 0
        dataset = dataset.repeat(self.n_epoch_readout)
        with self.writer.as_default():
            for inp_0, inp_1, target in dataset:
                encoding_a = self.autoencoder(inp_0, what=['encoding'])['encoding']
                prediction_0 = self.mod_0_to_1(encoding_a, what=['output'])['output']
                prediction_1 = self.mod_1_to_0(inp_1, what=['output'])['output']
                encoding_b = self.jointencoder(prediction_0, prediction_1, what=['encoding'])['encoding']
                loss = self.readout.train(encoding_b, target)
                print("readout_loss", loss.numpy())
                tf.summary.scalar("readout_loss", loss, step=step)
                step += 1

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
