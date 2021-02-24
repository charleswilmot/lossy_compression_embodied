import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from experiments import JointEncodingOption1, JointEncodingOption2
from dataset import get_batched_dataset


@hydra.main(config_path="../conf/joint_encoding/", config_name="config.yaml")
def joint_encoding(config):
    # print(OmegaConf.to_yaml(config))
    # print("\n" * 10)
    dataset = get_batched_dataset(
        get_original_cwd() + '/' + config.dataset.path,
        config.dataset.batch_size,
        n_epochs=1,
        z_score_frames=False
    )
    exp = JointEncodingOption1(config)
    exp.train_jointencoder(dataset.map(lambda x: (x['frame'], x['arm0_joints'])))
    exp.save_image_reconstructions(dataset.map(lambda x: (x['frame'], x['arm0_joints'])).take(10))
    exp.train_readout(dataset.map(lambda x: (x['frame'], x['arm0_joints'], x['arm0_end_eff'])))


if __name__ == '__main__':
    joint_encoding()
