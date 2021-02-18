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


def get_batched_dataset(path, batch_size, n_epochs=None):

    def read_pictures(element):
        return {
            'arm0_end_eff': element['arm0_end_eff'],
            'arm0_joints': element['arm0_joints'],
            'arm1_end_eff': element['arm1_end_eff'],
            'arm1_joints': element['arm1_joints'],
            'frame': tf.io.decode_jpeg(tf.io.read_file(
                path + '/' + element['frame_path']
            )),
        }

    dataset = get_dataset(path)
    dataset = dataset.shuffle(batch_size * 5, reshuffle_each_iteration=True)
    dataset = dataset.map(read_pictures)
    dataset = dataset.repeat(n_epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size * 5)
    return dataset


if __name__ == '__main__':
    import sys
    # d = get_dataset(sys.argv[1])
    d = get_batched_dataset(sys.argv[1], 1, 1)
