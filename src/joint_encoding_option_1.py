import tensorflow as tf
import hydra
import zlib
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from experiments import JointEncodingOption1, JointEncodingOption2
from dataset import get_batched_dataset, frame_only, frame_and_proprioception, frame_proprioception_and_readout_target
from database import ResultDatabase
from datetime import datetime


@hydra.main(config_path="../conf/training/", config_name="joint_encoding_option_1.yaml")
def joint_encoding(config):
    full_conf = zlib.compress(OmegaConf.to_yaml(config, resolve=True))
    dataset = get_batched_dataset(
        get_original_cwd() + '/' + config.dataset.path,
        config.dataset.batch_size,
        n_epochs=1,
        z_score_frames=False
    )
    exp = JointEncodingOption1(config)
    exp.train_jointencoder(frame_and_proprioception(dataset))
    frame_error_map = exp.save_image_reconstructions(frame_and_proprioception(dataset).take(10))
    exp.train_readout(frame_proprioception_and_readout_target(dataset))
    readouts = exp.get_readouts(frame_proprioception_and_readout_target(dataset).take(1000))
    db = ResultDatabase(get_original_cwd() + '/results/' + config.database_name)
    db.insert(
        date_time=datetime.now(),
        collection=config.collection,
        experiment_type='joint_encoding_option_1',
        bottleneck_size=config.layer_sizes.bottleneck_size,
        pre_encoding_size=config.layer_sizes.pre_encoding_size,
        arm0_end_eff_readout=readouts['arm0_end_eff'],
        arm0_positions_readout=readouts['arm0_positions'],
        arm0_velocities_readout=readouts['arm0_velocities'],
        arm1_end_eff_readout=readouts['arm1_end_eff'],
        arm1_positions_readout=readouts['arm1_positions'],
        arm1_velocities_readout=readouts['arm1_velocities'],
        frame_error_map=frame_error_map,
        full_conf=full_conf,
    )
    db.commit()


if __name__ == '__main__':
    joint_encoding()
