import matplotlib.pyplot as plt
import numpy as np
from database import ResultDatabase
import pandas as pd



def plot_readouts(ax, db, collection, experiment_type):
    results = db.get_dataframe('''
    SELECT
        bottleneck_size,
        arm0_end_eff_readout,
        arm0_positions_readout,
        arm0_velocities_readout,
        arm1_end_eff_readout,
        arm1_positions_readout,
        arm1_velocities_readout
    FROM
        results
    WHERE
        collection='{}'
    AND
        experiment_type='{}'
    '''.format(
        collection,
        experiment_type,
    ))
    results = results.sort_values(by=['bottleneck_size'])
    bn_size = [0] + list(results['bottleneck_size'].values)
    ax.plot(bn_size, [1] + list(results['arm0_end_eff_readout'].values), label='arm0_end_eff', color=(66 / 255, 135 / 255, 245 / 255))
    ax.plot(bn_size, [1] + list(results['arm0_positions_readout'].values), label='arm0_positions', color=(53 / 255, 111 / 255, 204 / 255))
    ax.plot(bn_size, [1] + list(results['arm0_velocities_readout'].values), label='arm0_velocities', color=(36 / 255, 78 / 255, 145 / 255))
    ax.plot(bn_size, [1] + list(results['arm1_end_eff_readout'].values), label='arm1_end_eff', color=(240 / 255, 127 / 255, 62 / 255))
    ax.plot(bn_size, [1] + list(results['arm1_positions_readout'].values), label='arm1_positions', color=(189 / 255, 99 / 255, 47 / 255))
    ax.plot(bn_size, [1] + list(results['arm1_velocities_readout'].values), label='arm1_velocities', color=(135 / 255, 70 / 255, 32 / 255))
    ax.set_title(experiment_type.replace('_', ' '))
    ax.set_xlabel('bottleneck size')
    ax.set_ylabel('readout error')
    # ax.legend()


def plot_frame_rec_err(ax, db, collection, experiment_type):
    results = db.get_dataframe('''
    SELECT
        bottleneck_size,
        frame_error_map
    FROM
        results
    WHERE
        collection='{}'
    AND
        experiment_type='{}'
    '''.format(
        collection,
        experiment_type,
    ))
    results = results.sort_values(by=['bottleneck_size'])
    bn_size = results['bottleneck_size'].values
    frame_error_map = np.stack(results['frame_error_map'].values)
    stop = frame_error_map.shape[2] // 2
    frame_error_left = np.mean(frame_error_map[:, :, :stop], axis=(1, 2))
    frame_error_right = np.mean(frame_error_map[:, :, stop:], axis=(1, 2))
    frame_error = np.mean(frame_error_map, axis=(1, 2))
    ax.plot(bn_size, frame_error, label='both')
    ax.plot(bn_size, frame_error_left, label='left')
    ax.plot(bn_size, frame_error_right, label='right')
    ax.set_title(experiment_type.replace('_', ' '))
    ax.set_xlabel('bottleneck size')
    ax.set_ylabel('frame reconstruction error')
    # ax.legend()


def plot_frame_err_map(ax, db, collection, experiment_type):
    results = db.get_dataframe('''
    SELECT
        bottleneck_size,
        frame_error_map
    FROM
        results
    WHERE
        collection='{}'
    AND
        experiment_type='{}'
    '''.format(
        collection,
        experiment_type,
    ))
    results = results.sort_values(by=['bottleneck_size'])
    bn_size = results['bottleneck_size'].values
    frame_error_map = np.stack(results['frame_error_map'].values)
    frame_error_map = frame_error_map.reshape((-1, frame_error_map.shape[2]))
    ax.imshow(frame_error_map)
    ax.set_title(experiment_type.replace('_', ' '))


if __name__ == '__main__':
    import sys

    database_path = sys.argv[1]
    collection = sys.argv[2]
    save = False
    dpi = 50
    # save = True
    # dpi = 300

    db = ResultDatabase(database_path)

    fig = plt.figure(figsize=(16, 4), dpi=dpi)
    ax = fig.add_subplot(141)
    plot_readouts(ax, db, collection, 'joint_encoding_option_1')
    ax = fig.add_subplot(142)
    plot_readouts(ax, db, collection, 'joint_encoding_option_2')
    ax = fig.add_subplot(143)
    plot_readouts(ax, db, collection, 'cross_modality_option_1')
    ax = fig.add_subplot(144)
    plot_readouts(ax, db, collection, 'cross_modality_option_2')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    if save:
        fig.savefig('/tmp/readouts.png')
    else:
        plt.show()


    fig = plt.figure(figsize=(16, 4), dpi=dpi)
    ax = fig.add_subplot(141)
    plot_frame_rec_err(ax, db, collection, 'joint_encoding_option_1')
    ax = fig.add_subplot(142)
    plot_frame_rec_err(ax, db, collection, 'joint_encoding_option_2')
    ax = fig.add_subplot(143)
    plot_frame_rec_err(ax, db, collection, 'cross_modality_option_1')
    ax = fig.add_subplot(144)
    plot_frame_rec_err(ax, db, collection, 'cross_modality_option_2')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    if save:
        fig.savefig('/tmp/frame_rec_err.png')
    else:
        plt.show()


    fig = plt.figure(figsize=(16, 16), dpi=dpi)
    ax = fig.add_subplot(141)
    plot_frame_err_map(ax, db, collection, 'joint_encoding_option_1')
    ax = fig.add_subplot(142)
    plot_frame_err_map(ax, db, collection, 'joint_encoding_option_2')
    ax = fig.add_subplot(143)
    plot_frame_err_map(ax, db, collection, 'cross_modality_option_1')
    ax = fig.add_subplot(144)
    plot_frame_err_map(ax, db, collection, 'cross_modality_option_2')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    if save:
        fig.savefig('/tmp/frame_err_map.png')
    else:
        plt.show()
