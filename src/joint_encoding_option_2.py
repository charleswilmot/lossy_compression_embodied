import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from experiments import JointEncodingOption1, JointEncodingOption2
from dataset import get_batched_dataset


@hydra.main(config_path="../conf/training/", config_name="joint_encoding_option_2.yaml")
def joint_encoding(config):
    # print(OmegaConf.to_yaml(config))
    # print("\n" * 10)
    dataset = get_batched_dataset(
        get_original_cwd() + '/' + config.dataset.path,
        config.dataset.batch_size,
        n_epochs=1,
        z_score_frames=False
    )
    exp = JointEncodingOption2(config)
    exp.train_autoencoder(dataset.map(lambda x: x['frame']))
    exp.z_score_encoding(dataset.map(lambda x: x['frame']).take(10))
    exp.train_jointencoder(dataset.map(lambda x: (x['frame'], x['arm0_positions'])))
    exp.save_image_reconstructions(dataset.map(lambda x: (x['frame'], x['arm0_positions'])).take(10))
    exp.train_readout(dataset.map(lambda x: (x['frame'], x['arm0_positions'], x['arm0_end_eff'])))


if __name__ == '__main__':
    joint_encoding()
