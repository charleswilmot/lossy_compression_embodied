from simulation import SimulationPool
from os.path import normpath, isdir
from os import makedirs
from shutil import rmtree
import numpy as np
from numpy_save import appendable_array_file
from PIL import Image
import hydra
from hydra.utils import get_original_cwd
import numpy as np
from welford import Welford


class DataGenerator:
    def __init__(self, config):
        self.name = config.name
        self.path_root = get_original_cwd() + '/' + config.path
        self.path = normpath(self.path_root + '/' + self.name + '/')
        if isdir(self.path):
            rmtree(self.path)
        makedirs(self.path)
        self.array_on_disk = appendable_array_file(self.path + '/table.dat')
        self.n_simulations = config.n_simulations
        self.simulations = SimulationPool(self.n_simulations)
        self.cam_resolution = config.cam_resolution
        self.n_steps_per_movment = config.n_steps_per_movment
        DISTANCE = 0.75
        self.simulations.add_arm(position=[-DISTANCE, 0.0, 0.0])
        self.simulations.add_arm(position=[ DISTANCE, 0.0, 0.0])
        self.cameras = self.simulations.add_camera(
            position=[0.0, 1.6, 1.0],
            orientation=[13 * np.pi / 20, 0.0, 0.0],
            resolution=self.cam_resolution,
            view_angle=90.0,
        )
        self.simulations.start_sim()
        self._buffer_dtype = np.dtype([
            ('arm0_end_eff', np.float32, (3,)),
            ('arm0_positions', np.float32, (7,)),
            ('arm0_velocities', np.float32, (7,)),
            ('arm0_forces', np.float32, (7,)),
            ('arm1_end_eff', np.float32, (3,)),
            ('arm1_positions', np.float32, (7,)),
            ('arm1_velocities', np.float32, (7,)),
            ('arm1_forces', np.float32, (7,)),
            ('frame_path', np.unicode_, 12),
        ])
        self._buffer = np.zeros(shape=self.n_simulations, dtype=self._buffer_dtype)
        self._welfords = {
            'arm0_end_eff': Welford(shape=(3,)),
            'arm0_positions': Welford(shape=(7,)),
            'arm0_velocities': Welford(shape=(7,)),
            'arm0_forces': Welford(shape=(7,)),
            'arm1_end_eff': Welford(shape=(3,)),
            'arm1_positions': Welford(shape=(7,)),
            'arm1_velocities': Welford(shape=(7,)),
            'arm1_forces': Welford(shape=(7,)),
        }
        self._frame_welford = Welford(shape=(self.cam_resolution[1], self.cam_resolution[0], 3))
        self._current_frame_id = 0
        with self.simulations.specific([0]):
            self._intervals = self.simulations.get_joint_intervals()[0]

    def set_random_target_position(self):
        with self.simulations.distribute_args():
            positions = np.random.uniform(
                size=(self.n_simulations, 14),
                low=self._intervals[:, 0],
                high=self._intervals[:, 1]
            )
            self.simulations.set_joint_target_positions(positions)

    def generate(self, n):
        for i in range(n // (self.n_simulations * self.n_steps_per_movment)):
            self.set_random_target_position()
            for j in range(self.n_steps_per_movment):
                self.move_arms()
                self.save_frames()
                self.flush_buffer()
                self.aggregate_mean_std()
                self._current_frame_id += self.n_simulations
                print("generated {: 8d}/{: 8d} frames".format(self._current_frame_id, n))

    def aggregate_mean_std(self):
        for key, welford in self._welfords.items():
            welford(self._buffer[key])

    def move_arms(self):
        self.simulations.step_sim()
        joint_positions = np.array(self.simulations.get_joint_positions())
        joint_velocities = np.array(self.simulations.get_joint_velocities())
        joint_forces = np.array(self.simulations.get_joint_forces())
        end_eff_positions = np.array(self.simulations.get_tips())
        self._buffer['arm0_positions'] = joint_positions[:, :7]
        self._buffer['arm1_positions'] = joint_positions[:, 7:]
        self._buffer['arm0_velocities'] = joint_velocities[:, :7]
        self._buffer['arm1_velocities'] = joint_velocities[:, 7:]
        self._buffer['arm0_forces'] = joint_forces[:, :7]
        self._buffer['arm1_forces'] = joint_forces[:, 7:]
        self._buffer['arm0_end_eff'] = end_eff_positions[:, :3]
        self._buffer['arm1_end_eff'] = end_eff_positions[:, 3:]
        self._buffer['frame_path'] = [
            '{:08d}.jpg'.format(x)
            for x in range(
                self._current_frame_id,
                self._current_frame_id + self.n_simulations
            )
        ]

    def save_frames(self):
        ids = np.arange(
            self._current_frame_id,
            self._current_frame_id + self.n_simulations
        )
        with self.simulations.distribute_args():
            frames = (np.array(self.simulations.get_frame(self.cameras)) * 255).astype(np.uint8)
        self._frame_welford(frames)
        for id, frame in zip(ids, frames):
            Image.fromarray(frame).save(self.path + '/{:08d}.jpg'.format(id))

    def flush_buffer(self):
        self.array_on_disk.append(self._buffer)

    def close(self):
        self.array_on_disk.close()
        mean_std = np.zeros(shape=2, dtype=self._buffer_dtype)
        for key, welford in self._welfords.items():
            mean_std[key][0] = welford.mean
            mean_std[key][1] = welford.std
        with appendable_array_file(self.path + '/mean_std.dat') as f:
            f.append(mean_std)
        with appendable_array_file(self.path + '/mean_std_frame.dat') as f:
            f.append(np.stack([self._frame_welford.mean, self._frame_welford.std], axis=0))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.close()

@hydra.main(config_path="../conf/generate_dataset_data/", config_name="config.yaml")
def main(config):
    with DataGenerator(config) as generator:
        generator.generate(config.n_samples)


if __name__ == '__main__':
    main()
