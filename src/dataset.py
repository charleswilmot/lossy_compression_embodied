import tensorflow as tf
from tensorflow.data import Dataset
from numpy_save import load_appendable_array_file


def get_dataset(path):
    table_path = path + '/table.dat'
    table = load_appendable_array_file(table_path)
    data_arm0_end_eff = table['arm0_end_eff']
    data_arm0_positions = table['arm0_positions']
    data_arm0_velocities = table['arm0_velocities']
    data_arm0_forces = table['arm0_forces']
    data_arm1_end_eff = table['arm1_end_eff']
    data_arm1_positions = table['arm1_positions']
    data_arm1_velocities = table['arm1_velocities']
    data_arm1_forces = table['arm1_forces']
    data_frame_path = table['frame_path']
    dataset = Dataset.from_tensor_slices({
        'arm0_end_eff': data_arm0_end_eff,
        'arm0_positions': data_arm0_positions,
        'arm0_velocities': data_arm0_velocities,
        'arm0_forces': data_arm0_forces,
        'arm1_end_eff': data_arm1_end_eff,
        'arm1_positions': data_arm1_positions,
        'arm1_velocities': data_arm1_velocities,
        'arm1_forces': data_arm1_forces,
        'frame_path': data_frame_path,
    })
    return dataset


def get_mean_std(path):
    mean_std = load_appendable_array_file(path + '/mean_std.dat')
    mean = mean_std[0]
    std = mean_std[1]
    return mean, std


def get_mean_std_frame(path):
    mean_std_frame = load_appendable_array_file(path + '/mean_std_frame.dat')
    mean_frame = mean_std_frame[0]
    std_frame = mean_std_frame[1]
    return mean_frame, std_frame


def get_batched_dataset(path, batch_size, n_epochs=None, z_score_frames=False):
    mean, std = get_mean_std(path)
    mean_frame, std_frame = get_mean_std_frame(path)

    if z_score_frames:
        def preprocess(element):
            return {
                'arm0_end_eff': (element['arm0_end_eff'] - mean['arm0_end_eff']) / std['arm0_end_eff'],
                'arm0_positions': (element['arm0_positions'] - mean['arm0_positions']) / std['arm0_positions'],
                'arm0_velocities': (element['arm0_velocities'] - mean['arm0_velocities']) / std['arm0_velocities'],
                'arm0_forces': (element['arm0_forces'] - mean['arm0_forces']) / std['arm0_forces'],
                'arm1_end_eff': (element['arm1_end_eff'] - mean['arm1_end_eff']) / std['arm1_end_eff'],
                'arm1_positions': (element['arm1_positions'] - mean['arm1_positions']) / std['arm1_positions'],
                'arm1_velocities': (element['arm1_velocities'] - mean['arm1_velocities']) / std['arm1_velocities'],
                'arm1_forces': (element['arm1_forces'] - mean['arm1_forces']) / std['arm1_forces'],
                'frame': (tf.cast(
                    tf.io.decode_jpeg(
                        tf.io.read_file(
                            path + '/' + element['frame_path']
                        ),
                        channels=3
                    ),
                    tf.float32
                ) - mean_frame) / (std_frame + 0.2)
            }
    else:
        def preprocess(element):
            return {
                'arm0_end_eff': (element['arm0_end_eff'] - mean['arm0_end_eff']) / std['arm0_end_eff'],
                'arm0_positions': (element['arm0_positions'] - mean['arm0_positions']) / std['arm0_positions'],
                'arm0_velocities': (element['arm0_velocities'] - mean['arm0_velocities']) / std['arm0_velocities'],
                'arm0_forces': (element['arm0_forces'] - mean['arm0_forces']) / std['arm0_forces'],
                'arm1_end_eff': (element['arm1_end_eff'] - mean['arm1_end_eff']) / std['arm1_end_eff'],
                'arm1_positions': (element['arm1_positions'] - mean['arm1_positions']) / std['arm1_positions'],
                'arm1_velocities': (element['arm1_velocities'] - mean['arm1_velocities']) / std['arm1_velocities'],
                'arm1_forces': (element['arm1_forces'] - mean['arm1_forces']) / std['arm1_forces'],
                'frame': (tf.cast(
                    tf.io.decode_jpeg(
                        tf.io.read_file(
                            path + '/' + element['frame_path']
                        ),
                        channels=3
                    ),
                    tf.float32
                ) - 127.5) / 127.5
            }

    dataset = get_dataset(path)
    dataset = dataset.shuffle(batch_size * 20, reshuffle_each_iteration=True)
    dataset = dataset.map(preprocess)
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(400)
    return dataset


def frame_and_proprioception(dataset):
    return dataset.map(lambda x: (x['frame'], tf.concat(
        [
            # x['arm0_end_eff'],
            x['arm0_positions'],
            x['arm0_velocities'],
            # x['arm0_forces'],
        ],
        axis=-1,
    )))


def frame_proprioception_and_readout_target(dataset):
    return dataset.map(lambda x: (
        x['frame'],
        tf.concat(
            [
                # x['arm0_end_eff'],
                x['arm0_positions'],
                x['arm0_velocities'],
                # x['arm0_forces'],
            ],
            axis=-1,
        ),
        tf.concat(
            [
                x['arm0_end_eff'],
                x['arm0_positions'],
                x['arm0_velocities'],
                # x['arm0_forces'],
                x['arm1_end_eff'],
                x['arm1_positions'],
                x['arm1_velocities'],
                # x['arm1_forces'],
            ],
            axis=-1,
        )
    ))


def frame_only(dataset):
    return dataset.map(lambda x: x['frame'])


if __name__ == '__main__':
    import sys
    # d = get_dataset(sys.argv[1])
    d = get_batched_dataset(sys.argv[1], 1, 1)
