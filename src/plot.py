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
    results.loc[-1] = [0, 1, 1, 1, 1, 1, 1]
    results.index += 1
    results.loc[-1] = [0, 1, 1, 1, 1, 1, 1]
    results.index += 1
    results = results.sort_values(by=['bottleneck_size'])
    grouped = results.groupby(['bottleneck_size'], as_index=False)
    means = grouped.mean()
    stds = grouped.std()
    bn_size = means['bottleneck_size'].values
    names = [
        'arm0_end_eff_readout',
        'arm0_positions_readout',
        'arm0_velocities_readout',
        'arm1_end_eff_readout',
        'arm1_positions_readout',
        'arm1_velocities_readout',
    ]
    colors = [
        (66 / 255, 135 / 255, 245 / 255),
        (53 / 255, 111 / 255, 204 / 255),
        (36 / 255, 78 / 255, 145 / 255),
        (240 / 255, 127 / 255, 62 / 255),
        (189 / 255, 99 / 255, 47 / 255),
        (135 / 255, 70 / 255, 32 / 255),
    ]
    for name, color in zip(names, colors):
        ax.fill_between(
            bn_size,
            means[name].values - stds[name].values,
            means[name].values + stds[name].values,
            color=color,
            alpha=0.3,
        )
        ax.plot(
            bn_size,
            means[name].values,
            color=color,
            label=name.replace('_readout', '').replace('_', ' ')
        )
    ax.set_title(experiment_type.replace('_', ' '))
    ax.set_xlabel('bottleneck size')
    ax.set_ylabel('readout error')


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

    def left_right_both(series):
        stop = series['frame_error_map'].shape[1] // 2
        return pd.Series({
            'bottleneck_size': series['bottleneck_size'],
            'frame_error': np.mean(series['frame_error_map']),
            'frame_error_left': np.mean(series['frame_error_map'][:, :stop]),
            'frame_error_right': np.mean(series['frame_error_map'][:, stop:]),
        }, dtype=object)

    results = results.apply(left_right_both, axis=1)
    grouped = results.groupby(['bottleneck_size'])
    means = grouped.mean()
    stds = grouped.std()
    bn_size = means.index.values
    names = ['frame_error', 'frame_error_left', 'frame_error_right']
    labels = ['both', 'left', 'right']
    colors = [(153 / 255, 141 / 255, 153 / 255), (240 / 255, 127 / 255, 62 / 255), (66 / 255, 135 / 255, 245 / 255)]
    for name, label, color in zip(names, labels, colors):
        ax.fill_between(
            bn_size,
            means[name].values - stds[name].values,
            means[name].values + stds[name].values,
            alpha=0.3,
            color=color
        )
        ax.plot(
            bn_size,
            means[name].values,
            label=label,
            color=color
        )
    ax.set_title(experiment_type.replace('_', ' '))
    ax.set_xlabel('bottleneck size')
    ax.set_ylabel('frame reconstruction error')


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
    grouped = results.groupby(['bottleneck_size'])

    def mean(df):
        return pd.DataFrame({
                'frame_error_map': [np.mean(np.stack(df['frame_error_map'].values), axis=0)]
            },
            index=[df['bottleneck_size'].values[0]],
        )

    means = grouped.apply(mean).reset_index(level=1, drop=True)
    frame_error_map = np.stack(means['frame_error_map'].values)
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
    experiments = ['joint_encoding_option_1', 'joint_encoding_option_2'] #, 'cross_modality_option_1', 'cross_modality_option_2']
    n_experiments = len(experiments)

    db = ResultDatabase(database_path)

    fig = plt.figure(figsize=(4 * n_experiments, 4), dpi=dpi)
    for i, experiment_type in enumerate(experiments):
        ax = fig.add_subplot(1, n_experiments, i + 1)
        plot_readouts(ax, db, collection, experiment_type)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    if save:
        fig.savefig('/tmp/readouts.png')
    else:
        plt.show()


    fig = plt.figure(figsize=(4 * n_experiments, 4), dpi=dpi)
    for i, experiment_type in enumerate(experiments):
        ax = fig.add_subplot(1, n_experiments, i + 1)
        plot_frame_rec_err(ax, db, collection, experiment_type)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    if save:
        fig.savefig('/tmp/frame_rec_err.png')
    else:
        plt.show()


    fig = plt.figure(figsize=(4 * n_experiments, 16), dpi=dpi)
    for i, experiment_type in enumerate(experiments):
        ax = fig.add_subplot(1, n_experiments, i + 1)
        plot_frame_err_map(ax, db, collection, experiment_type)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    if save:
        fig.savefig('/tmp/frame_err_map.png')
    else:
        plt.show()
