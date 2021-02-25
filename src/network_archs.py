import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from omegaconf import OmegaConf
from custom_objects import custom_objects


def to_model(cfg):
    return models.model_from_yaml(OmegaConf.to_yaml(cfg, resolve=True), custom_objects=custom_objects)


class JointEncoder:
    def __init__(self, config):
        self.pre_encoding_model_0 = to_model(config.pre_encoding_model_0)
        self.pre_encoding_model_1 = to_model(config.pre_encoding_model_1)
        self.encoding_model = to_model(config.encoding_model)
        self.pre_decoding_model = to_model(config.pre_decoding_model)
        self.decoding_model_0 = to_model(config.decoding_model_0)
        self.decoding_model_1 = to_model(config.decoding_model_1)
        self.learning_rate = config.learning_rate
        self.pre_reconstruction_0_size = config.pre_reconstruction_0_size
        self.optimizer = keras.optimizers.Adam(self.learning_rate)
        self.reduction_type = 'mean'

    @tf.function
    def get_pre_encoding_0(self, inp):
        return self.pre_encoding_model_0(inp)

    @tf.function
    def get_pre_encoding_1(self, inp):
        return self.pre_encoding_model_1(inp)

    @tf.function
    def get_encoding(self, pre_encoding_0, pre_encoding_1):
        pre_encoding = tf.concat([pre_encoding_0, pre_encoding_1], axis=-1)
        return self.encoding_model(pre_encoding)

    @tf.function
    def get_pre_reconstructions(self, encoding):
        pre_reconstructions = self.pre_decoding_model(encoding)
        pre_reconstruction_0 = pre_reconstructions[..., :self.pre_reconstruction_0_size]
        pre_reconstruction_1 = pre_reconstructions[..., self.pre_reconstruction_0_size:]
        return pre_reconstruction_0, pre_reconstruction_1

    @tf.function
    def get_reconstruction_0(self, pre_reconstruction_0):
        return self.decoding_model_0(pre_reconstruction_0)

    @tf.function
    def get_reconstruction_1(self, pre_reconstruction_1):
        return self.decoding_model_1(pre_reconstruction_1)

    @tf.function
    def get_loss(self, inp, reconstruction):
        error = (reconstruction - inp) ** 2
        if self.reduction_type == 'mean':
            return tf.reduce_mean(error, axis=0)
        else:
            return tf.reduce_sum(error, axis=0)

    @tf.function
    def __call__(self, inp_0, inp_1, what):
        _to_number = {
            'pre_encodings': 0,
            'encoding': 1,
            'pre_reconstructions': 2,
            'reconstructions': 3,
            'losses': 4,
        }
        stop = max(_to_number[w] for w in what)
        ret = {}
        pre_encoding_0 = self.get_pre_encoding_0(inp_0)
        pre_encoding_1 = self.get_pre_encoding_1(inp_1)
        if stop > 0:
            encoding = self.get_encoding(pre_encoding_0, pre_encoding_1)
        if stop > 1:
            pre_reconstruction_0, pre_reconstruction_1 = self.get_pre_reconstructions(encoding)
        if stop > 2:
            reconstruction_0 = self.get_reconstruction_0(pre_reconstruction_0)
            reconstruction_1 = self.get_reconstruction_1(pre_reconstruction_1)
        if stop > 3:
            loss_0 = self.get_loss(inp_0, reconstruction_0)
            loss_1 = self.get_loss(inp_1, reconstruction_1)
        if 'pre_encodings' in what:
            ret['pre_encoding_0'] = pre_encoding_0
            ret['pre_encoding_1'] = pre_encoding_1
        if 'encoding' in what:
            ret['encoding'] = encoding
        if 'pre_reconstructions' in what:
            ret['pre_reconstruction_0'] = pre_reconstruction_0
            ret['pre_reconstruction_1'] = pre_reconstruction_1
        if 'reconstructions' in what:
            ret['reconstruction_0'] = reconstruction_0
            ret['reconstruction_1'] = reconstruction_1
        if 'losses' in what:
            ret['loss_0'] = loss_0
            ret['loss_1'] = loss_1
        return ret

    @tf.function
    def train(self, inp_0, inp_1):
        with tf.GradientTape() as tape:
            ret = self(inp_0, inp_1, what=['losses'])
            loss_0, loss_1 = ret['loss_0'], ret['loss_1']
            mean_loss_0 = tf.reduce_mean(loss_0)
            mean_loss_1 = tf.reduce_mean(loss_1)
            loss = (mean_loss_0 + mean_loss_1) / 2.0
            vars = \
                self.pre_encoding_model_0.trainable_variables + \
                self.pre_encoding_model_1.trainable_variables + \
                self.encoding_model.trainable_variables + \
                self.pre_decoding_model.trainable_variables + \
                self.decoding_model_0.trainable_variables + \
                self.decoding_model_1.trainable_variables
            grads = tape.gradient(loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))
        return loss_0, loss_1


class AutoEncoder:
    def __init__(self, config):
        self.encoding_model = to_model(config.encoding_model)
        self.decoding_model = to_model(config.decoding_model)
        self.encoding_size = self.encoding_model.layers[-1].units
        self.learning_rate = config.learning_rate
        self.optimizer = keras.optimizers.Adam(self.learning_rate)
        self.reduction_type = 'mean'

    @tf.function
    def get_encoding(self, inp):
        return self.encoding_model(inp)

    @tf.function
    def get_reconstruction(self, encoding):
        return self.decoding_model(encoding)

    @tf.function
    def get_loss(self, inp, reconstruction):
        error = (reconstruction - inp) ** 2
        if self.reduction_type == 'mean':
            return tf.reduce_mean(error, axis=0)
        else:
            return tf.reduce_sum(error, axis=0)

    @tf.function
    def __call__(self, inp, what):
        _to_number = {
            'encoding': 0,
            'reconstructions': 1,
            'loss': 2,
        }
        stop = max(_to_number[w] for w in what)
        ret = {}
        encoding = self.get_encoding(inp)
        if stop > 0:
            reconstruction = self.get_reconstruction(encoding)
        if stop > 1:
            loss = self.get_loss(inp, reconstruction)
        if 'encoding' in what:
            ret['encoding'] = encoding
        if 'reconstructions' in what:
            ret['reconstruction'] = reconstruction
        if 'loss' in what:
            ret['loss'] = loss
        return ret

    @tf.function
    def train(self, inp):
        with tf.GradientTape() as tape:
            loss = self(inp, what=['loss'])['loss']
            mean_loss = tf.reduce_mean(loss)
            vars = \
                self.encoding_model.trainable_variables + \
                self.decoding_model.trainable_variables
            grads = tape.gradient(mean_loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))
        return loss


class MLP:
    def __init__(self, config):
        self.model = to_model(config.model)
        self.learning_rate = config.learning_rate
        self.optimizer = keras.optimizers.Adam(self.learning_rate)
        self.reduction_type = 'mean'

    @tf.function
    def get_output(self, inp):
        return self.model(inp)

    @tf.function
    def get_loss(self, out, target):
        error = (out - target) ** 2
        if self.reduction_type == 'mean':
            return tf.reduce_mean(error, axis=0)
        else:
            return tf.reduce_sum(error, axis=0)

    @tf.function
    def __call__(self, inp, target=None, what=['output']):
        _to_number = {
            'output': 0,
            'loss': 1,
        }
        stop = max(_to_number[w] for w in what)
        out = self.get_output(inp)
        if stop > 0:
            loss = self.get_loss(out, target)
        ret = {}
        if 'output' in what:
            ret['output'] = out
        if 'loss' in what:
            ret['loss'] = loss
        return ret

    @tf.function
    def train(self, inp, target):
        with tf.GradientTape() as tape:
            loss = self(inp, target, what=['loss'])['loss']
            mean_loss = tf.reduce_mean(loss)
            vars = self.model.trainable_variables
            grads = tape.gradient(mean_loss, vars)
            self.optimizer.apply_gradients(zip(grads, vars))
        return loss
