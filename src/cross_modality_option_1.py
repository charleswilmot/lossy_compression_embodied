import tensorflow as tf
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from experiments import CrossModalityOption1, CrossModalityOption2
from dataset import get_batched_dataset


@hydra.main(config_path="../conf/training/", config_name="cross_modality_option_1.yaml")
def cross_modality(config):
    dataset = get_batched_dataset(
        get_original_cwd() + '/' + config.dataset.path,
        config.dataset.batch_size,
        n_epochs=1,
        z_score_frames=False
    )
    exp = CrossModalityOption1(config)
    exp.train_cross_modality(frame_and_proprioception(dataset))
    exp.train_jointencoder(frame_and_proprioception(dataset))
    exp.save_image_reconstructions(frame_and_proprioception(dataset).take(10))
    exp.train_readout(frame_proprioception_and_readout_target(dataset))


if __name__ == '__main__':
    cross_modality()
