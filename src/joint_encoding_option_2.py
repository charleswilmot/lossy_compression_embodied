import tensorflow as tf
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from experiments import JointEncodingOption1, JointEncodingOption2
from dataset import get_batched_dataset


@hydra.main(config_path="../conf/training/", config_name="joint_encoding_option_2.yaml")
def joint_encoding(config):
    dataset = get_batched_dataset(
        get_original_cwd() + '/' + config.dataset.path,
        config.dataset.batch_size,
        n_epochs=1,
        z_score_frames=False
    )
    exp = JointEncodingOption2(config)
    exp.train_autoencoder(frame_only(dataset))
    exp.z_score_encoding(frame_only(dataset).take(10))
    exp.train_jointencoder(frame_and_proprioception(dataset))
    exp.save_image_reconstructions(frame_and_proprioception(dataset).take(10))
    exp.train_readout(frame_proprioception_and_readout_target(dataset))


if __name__ == '__main__':
    joint_encoding()
