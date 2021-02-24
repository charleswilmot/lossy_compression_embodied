import tensorflow as tf
import numpy as np
from network_archs import JointEncoder, MLP
from PIL import Image
from os import makedirs


class JointEncodingOption1:
    def __init__(self, config):
        self.jointencoder = JointEncoder(config.jointencoder)
        self.readout = MLP(config.readout)
        self.n_epoch_jointencoder = config.jointencoder.n_epochs
        self.n_epoch_readout = config.readout.n_epochs
        self.writer = tf.summary.create_file_writer("./summaries")
        self.reconstructions_path = './reconstructions'
        makedirs(self.reconstructions_path)

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

    def get_image_reconstructions(self, dataset):
        return np.array([(
                inp_0,
                self.jointencoder(
                    inp_0,
                    inp_1,
                    what=['reconstructions']
                )['reconstruction_0']
            )
            for inp_0, inp_1 in dataset
        ]) # shape = [M, 2, N, height, width, 3]

    def save_image_reconstructions(self, dataset):
        image_reconstructions = self.get_image_reconstructions(dataset) # shape = [M, 2, N, height, width, 3]
        batch_size = image_reconstructions.shape[2]
        ratio = 320 // image_reconstructions.shape[3]
        size = (image_reconstructions.shape[4] * ratio, image_reconstructions.shape[3] * ratio)
        for batch in image_reconstructions:
            for i in range(batch_size):
                frame = np.concatenate([batch[0, i], batch[1, i]], axis=0)
                frame = np.clip(frame * 127.5 + 127.5, 0, 255).astype(np.uint8)
                Image.fromarray(frame).resize(size).save(self.reconstructions_path + '/{:04d}.jpg'.format(i))
        error_map = np.mean((image_reconstructions[:, 0] - image_reconstructions[:, 1]) ** 2, axis=(0, 1, -1)) # [height, width]
        mini, maxi = error_map.min(), error_map.max()
        error_map = (error_map - mini) / (maxi - mini)
        error_map = (error_map * 255).astype(np.uint8)
        Image.fromarray(error_map).resize(size).save(self.reconstructions_path + '/error_map_{:.4f}_{:.4f}.jpg'.format(mini, maxi))


class JointEncodingOption2:
    def __init__(self, config):
        self.autoencoder = AutoEncoder(config.autoencoder)
        self.jointencoder = JointEncoder(config.jointencoder)
        self.readout = MLP(config.readout)
        self.n_epoch_autoencoder = config.autoencoder.n_epochs
        self.n_epoch_jointencoder = config.jointencoder.n_epochs
        self.n_epoch_readout = config.readout.n_epochs

    def train_autoencoder(self, dataset):
        dataset = dataset.repeat(self.n_epoch_autoencoder)
        for inp in dataset:
            self.autoencoder.train(inp)

    def train_jointencoder(self, dataset):
        dataset = dataset.repeat(self.n_epoch_jointencoder)
        for inp_0, inp_1 in dataset:
            encoding = self.autoencoder(inp_0, what=['encoding'])['encoding']
            self.jointencoder.train(encoding, inp_1)

    def train_readout(self, dataset):
        dataset = dataset.repeat(self.n_epoch_readout)
        for inp_0, inp_1, target in dataset:
            encoding_a = self.autoencoder(inp_0, what=['encoding'])['encoding']
            encoding_b = self.jointencoder(encoding_a, inp_1, what=['encoding'])['encoding']
            self.readout.train(encoding_b, target)


class CrossModalityOption1:
    def __init__(self, config):
        self.mod_0_to_1 = MLP(config.mod_0_to_1)
        self.mod_1_to_0 = MLP(config.mod_1_to_0)
        self.jointencoder = JointEncoder(config.jointencoder)
        self.readout = MLP(config.readout)
        self.n_epoch_cross_modality = config.cross_modality.n_epochs
        self.n_epoch_jointencoder = config.jointencoder.n_epochs
        self.n_epoch_readout = config.readout.n_epochs

    def train_cross_modality(self, dataset):
        dataset = dataset.repeat(self.n_epoch_cross_modality)
        for inp_0, inp_1 in dataset:
            self.mod_0_to_1.train(inp_0, inp_1)
            self.mod_1_to_0.train(inp_1, inp_0)

    def train_jointencoder(self, dataset):
        dataset = dataset.repeat(self.n_epoch_jointencoder)
        for inp_0, inp_1 in dataset:
            prediction_0 = self.mod_0_to_1(inp_0)
            prediction_1 = self.mod_1_to_0(inp_1)
            self.jointencoder.train(prediction_0, prediction_1)

    def train_readout(self, dataset):
        dataset = dataset.repeat(self.n_epoch_readout)
        for inp_0, inp_1, target in dataset:
            prediction_0 = self.mod_0_to_1(inp_0)
            prediction_1 = self.mod_1_to_0(inp_1)
            encoding = self.jointencoder(prediction_0, prediction_1, what=['encoding'])['encoding']
            self.readout.train(encoding, target)


class CrossModalityOption2:
    def __init__(self, config):
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
        dataset = dataset.repeat(self.n_epoch_autoencoder)
        for inp in dataset:
            self.autoencoder.train(inp)

    def train_cross_modality(self, dataset):
        dataset = dataset.repeat(self.n_epoch_cross_modality)
        for inp_0, inp_1 in dataset:
            encoding = self.autoencoder(inp_0, what=['encoding'])['encoding']
            self.mod_0_to_1.train(encoding, inp_1)
            self.mod_1_to_0.train(inp_1, encoding)

    def train_jointencoder(self, dataset):
        dataset = dataset.repeat(self.n_epoch_jointencoder)
        for inp_0, inp_1 in dataset:
            encoding = self.autoencoder(inp_0, what=['encoding'])['encoding']
            prediction_0 = self.mod_0_to_1(encoding)
            prediction_1 = self.mod_1_to_0(inp_1)
            self.jointencoder.train(prediction_0, prediction_1)

    def train_readout(self, dataset):
        dataset = dataset.repeat(self.n_epoch_readout)
        for inp_0, inp_1, target in dataset:
            encoding_a = self.autoencoder(inp_0, what=['encoding'])['encoding']
            prediction_0 = self.mod_0_to_1(encoding_a)
            prediction_1 = self.mod_1_to_0(inp_1)
            encoding_b = self.jointencoder(prediction_0, prediction_1, what=['encoding'])['encoding']
            self.readout.train(encoding_b, target)
