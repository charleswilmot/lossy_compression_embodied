import tensorflow as tf
from tensorflow.data import Dataset
from numpy_save import load_appendable_array_file


def get_dataset(path):
    table_path = path + '/table.dat'
    table = load_appendable_array_file(table_path)
    data_arm0_end_eff = table['arm0_end_eff']
    data_arm0_joints = table['arm0_joints']
    data_arm1_end_eff = table['arm1_end_eff']
    data_arm1_joints = table['arm1_joints']
    data_frame_path = table['frame_path']
    dataset = Dataset.from_tensor_slices({
        'arm0_end_eff': data_arm0_end_eff,
        'arm0_joints': data_arm0_joints,
        'arm1_end_eff': data_arm1_end_eff,
        'arm1_joints': data_arm1_joints,
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


def get_batched_dataset(path, batch_size, n_epochs=None):
    mean, std = get_mean_std(path)
    mean_frame, std_frame = get_mean_std_frame(path)

    def preprocess(element):
        return {
            'arm0_end_eff': (element['arm0_end_eff'] - mean['arm0_end_eff']) / std['arm0_end_eff'],
            'arm0_joints': (element['arm0_joints'] - mean['arm0_joints']) / std['arm0_joints'],
            'arm1_end_eff': (element['arm1_end_eff'] - mean['arm1_end_eff']) / std['arm1_end_eff'],
            'arm1_joints': (element['arm1_joints'] - mean['arm1_joints']) / std['arm1_joints'],
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

    dataset = get_dataset(path)
    dataset = dataset.shuffle(batch_size * 5, reshuffle_each_iteration=True)
    dataset = dataset.map(preprocess)
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size * 5)
    return dataset


if __name__ == '__main__':
    import sys
    # d = get_dataset(sys.argv[1])
    d = get_batched_dataset(sys.argv[1], 1, 1)
