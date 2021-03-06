import matplotlib.pyplot as plt
import numpy as np
from database import ResultDatabase
import pandas as pd
from PIL import Image, ImageFont, ImageDraw
import os


BLUE_1 = (66 / 255, 135 / 255, 245 / 255)
BLUE_2 = (53 / 255, 111 / 255, 204 / 255)
BLUE_3 = (36 / 255, 78 / 255, 145 / 255)
RED_1 = (240 / 255, 127 / 255, 62 / 255)
RED_2 = (189 / 255, 99 / 255, 47 / 255)
RED_3 = (135 / 255, 70 / 255, 32 / 255)
GREY_1 = (153 / 255, 141 / 255, 153 / 255)


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
    colors = [BLUE_1, BLUE_2, BLUE_3, RED_1, RED_2, RED_3]
    linestyles = ['-', '-', '--', '--', '--', ':']
    for name, color, linestyle in zip(names, colors, linestyles):
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
            linestyle=linestyle,
            label=name.replace('_readout', '').replace('_', ' ').replace('arm0', 'R arm').replace('arm1', 'L arm').replace('positions', 'pos').replace('velocities', 'vel')
        )
    ax.set_title(experiment_type.replace('joint_encoding', 'JE').replace('1a', '1').replace('cross_modality', 'CM').replace('_', ' '))
    ax.set_xlabel('bottleneck size')
    if ax.get_subplotspec().colspan[0] == 0:
        ax.set_ylabel('Readout error (MSE)')
    ax.set_ylim([-0.05, 1.05])


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
            # 'frame_error': np.mean(series['frame_error_map']),
            'frame_error_left': np.mean(series['frame_error_map'][:, :stop]),
            'frame_error_right': np.mean(series['frame_error_map'][:, stop:]),
        }, dtype=object)

    results = results.apply(left_right_both, axis=1)
    grouped = results.groupby(['bottleneck_size'])
    means = grouped.mean()
    stds = grouped.std()
    bn_size = means.index.values
    # names = ['frame_error', 'frame_error_left', 'frame_error_right']
    # labels = ['both', 'left half', 'right half']
    # colors = [GREY_1, RED_1, BLUE_1]
    names = ['frame_error_left', 'frame_error_right']
    labels = ['left half', 'right half']
    colors = [RED_1, BLUE_1]
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
    ax.set_title(experiment_type.replace('joint_encoding', 'JE').replace('1a', '1').replace('cross_modality', 'CM').replace('_', ' '))
    ax.set_xlabel('bottleneck size')
    if ax.get_subplotspec().colspan[0] == 0:
        ax.set_ylabel('Image reconstruction error (MSE)')
    ax.set_ylim([0.006, 0.032])


def plot_frame_error_map(ax, db, collection, experiment_type):
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
    n_maps = frame_error_map.shape[0]
    n_tapes = 4
    n_maps_per_tape = n_maps // n_tapes
    map_height = frame_error_map.shape[1]
    tape_length = n_maps_per_tape * map_height
    frame_error_map = frame_error_map.reshape((-1, frame_error_map.shape[2]))
    labels = np.zeros_like(frame_error_map)
    fontsize = 14
    font = ImageFont.truetype("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf", fontsize)
    img = Image.new("F", (map_height, frame_error_map.shape[0]), 0.0)
    draw = ImageDraw.Draw(img)
    for i, bn_size in enumerate(means.index):
        draw.text((0, i * map_height + (map_height - fontsize) / 2), "{: 2d}".format(bn_size), fill=frame_error_map.max(), font=font)
    labels = np.array(img)
    frame_error_map = np.concatenate([labels, frame_error_map], axis=-1)
    frame_error_map = frame_error_map.reshape((n_tapes, tape_length, frame_error_map.shape[1]))
    frame_error_map = np.concatenate([x for x in frame_error_map], axis=-1)
    ax.imshow(frame_error_map, cmap='Greys')
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title(experiment_type.replace('joint_encoding', 'JE').replace('1a', '1').replace('cross_modality', 'CM').replace('_', ' '))


def generate_plots(db, collection, experiment_types, fig_ratio=0.7, dpi=300, save=False, prefix=''):
    os.makedirs('/tmp/plots', exist_ok=True)
    n_experiments = len(experiment_types)
    subplots_adjust_1_args = {
        1: dict(left=0.10, bottom=0.12, right=0.85, top=0.90, wspace=0.30, hspace=0.05),
        2: dict(left=0.13, bottom=0.20, right=0.70, top=0.90, wspace=0.40, hspace=0.05),
        3: dict(left=0.10, bottom=0.12, right=0.85, top=0.90, wspace=0.30, hspace=0.05),
        4: dict(left=0.10, bottom=0.12, right=0.85, top=0.90, wspace=0.30, hspace=0.05),
    }
    subplots_adjust_2_args = {
        1: dict(left=0.02, bottom=0.02, right=0.98, top=0.90, wspace=0.05, hspace=0.05),
        2: dict(left=0.02, bottom=0.02, right=0.98, top=0.90, wspace=0.05, hspace=0.05),
        3: dict(left=0.02, bottom=0.02, right=0.98, top=0.90, wspace=0.05, hspace=0.05),
        4: dict(left=0.02, bottom=0.02, right=0.98, top=0.90, wspace=0.05, hspace=0.05),
    }


    fig = plt.figure(figsize=(fig_ratio * 4 * n_experiments, fig_ratio * 4), dpi=dpi)
    for i, experiment_type in enumerate(experiment_types):
        ax = fig.add_subplot(1, n_experiments, i + 1)
        plot_readouts(ax, db, collection, experiment_type)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    fig.subplots_adjust(**subplots_adjust_1_args[n_experiments])
    if save:
        fig.savefig('/tmp/plots/{}readouts_dpi{}_ratio{}.png'.format(prefix, dpi, fig_ratio))
    else:
        plt.show()


    fig = plt.figure(figsize=(fig_ratio * 4 * n_experiments, fig_ratio * 4), dpi=dpi)
    for i, experiment_type in enumerate(experiment_types):
        ax = fig.add_subplot(1, n_experiments, i + 1)
        plot_frame_rec_err(ax, db, collection, experiment_type)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='right')
    fig.subplots_adjust(**subplots_adjust_1_args[n_experiments])
    if save:
        fig.savefig('/tmp/plots/{}frame_rec_err_dpi{}_ratio{}.png'.format(prefix, dpi, fig_ratio))
    else:
        plt.show()


    fig = plt.figure(figsize=(fig_ratio * 4 * n_experiments, fig_ratio * 3), dpi=dpi)
    for i, experiment_type in enumerate(experiment_types):
        ax = fig.add_subplot(1, n_experiments, i + 1)
        plot_frame_error_map(ax, db, collection, experiment_type)

    handles, labels = ax.get_legend_handles_labels()
    fig.subplots_adjust(**subplots_adjust_2_args[n_experiments])
    if save:
        fig.savefig('/tmp/plots/{}frame_error_map_dpi{}_ratio{}.png'.format(prefix, dpi, fig_ratio))
    else:
        plt.show()


if __name__ == '__main__':
    import sys

    database_path = sys.argv[1]
    collection = sys.argv[2]
    db = ResultDatabase(database_path)


    generate_plots(
        db,
        collection,
        ['joint_encoding_option_1a', 'cross_modality_option_1'],
        save=True,
        prefix='option_1_'
    )

    generate_plots(
        db,
        collection,
        ['joint_encoding_option_2', 'cross_modality_option_2'],
        save=True,
        prefix='option_2_'
    )
