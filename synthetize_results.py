import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def get_experiments(path):
    directories = [os.path.join(path, x) for x in os.listdir(path)
                   if os.path.exists(os.path.join(path, x, 'exp_config.txt'))]
    return directories


def parse_configuration(path):
    config_path = os.path.join(path, 'exp_config.txt')
    config_df = pd.read_csv(config_path, sep="    ", header=None).T
    config_df.columns = config_df.iloc[0]
    config_df = config_df.reindex(config_df.index.drop(0))

    return json.loads(config_df.to_json())


def parse_learning_curves(path):
    curves_path = os.path.join(path, 'learning_curves.npy')
    curves = np.load(curves_path)
    return curves.item()


def parse_log(path):
    log_path = os.path.join(path, 'log.txt')
    log_df = pd.read_csv(log_path, delimiter='\t', header=None, names=['Epoch', 'Train', 'Val', 'Best', 'Time'])
    log_df.replace(regex=True, inplace=True, to_replace=r'[a-zA-Z\:\s\(\)]', value=r'')

    wall_clock_time = []
    running_time = 0
    for time in log_df['Time']:
        running_time += float(time)
        wall_clock_time.append(running_time)

    return log_df, wall_clock_time


def plot_model_comparison(config, curves, log, walltime):
    valid_ppl = curves.get('val_ppls')
    train_ppl = curves.get('train_ppls')
    epochs = range(len(valid_ppl))
    wall_clock_time = walltime
    path = os.path.join(cwd + '/results/', 'model_comparison/')

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1", axisbelow='True')

    ax.plot(epochs, train_ppl, 'b', epochs, valid_ppl, 'r')

    ax.set_xlabel("Epochs", color="k")
    ax.set_ylabel("Perplexity", color="k")
    ax.set_ylim(0, 700)
    ax.tick_params(axis='x', colors="k")
    ax.tick_params(axis='y', colors="k")

    plt.grid(linestyle='dotted')

    legend = ax.legend(['Train', 'Validation'], loc='best', fancybox=True, facecolor="white", framealpha=1)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig_name = path + 'model_comparison_epochs_{}.jpg'.format(config['model']['1'].lower())
    plt.savefig(fig_name, dpi=300, bbox_inches='tight', pad_inches=0)

    fig = plt.figure()
    ax = fig.add_subplot(111, label="1", axisbelow='True')

    ax.plot(wall_clock_time, train_ppl, "b", wall_clock_time, valid_ppl, 'r')

    ax.set_xlabel('Wall Clock Time ($s$)', color="k")
    ax.set_ylabel("Perplexity", color="k")
    ax.set_ylim(0, 700)
    ax.tick_params(axis='x', colors="k")
    ax.tick_params(axis='y', colors="k")

    plt.grid(linestyle='dotted')

    legend = ax.legend(['Train', 'Validation'], loc='best', fancybox=True, facecolor="white", framealpha=1)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig_name = path + 'model_comparison_time_{}.jpg'.format(config['model']['1'].lower())
    plt.savefig(fig_name, dpi=300, bbox_inches='tight', pad_inches=0)


def plot_exploration_of_optimizers(fig, ax, curves, learning_curves, log, walltime, x, set):
    pass


def synthesize_model_comparison(path):
    # Get all experiments in folder
    experiments = get_experiments(path)

    for experiment in experiments:
        config = parse_configuration(experiment)
        log, walltime = parse_log(experiment)
        curves = parse_learning_curves(experiment)

        # Plot experiment
        plot_model_comparison(config, curves, log, walltime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to results')
    args = parser.parse_args()
    cwd = os.getcwd()
    results_path = cwd + '/results/'
    # model_comparison_path = os.path.join(cwd + '/results/', 'model_comparison/')
    # print(model_comparison_path)
    # synthesize_model_comparison(model_comparison_path)
    synthesize_model_comparison(results_path)

