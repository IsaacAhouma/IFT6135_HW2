import json
import argparse
from subprocess import call
import multiprocessing


def run_experiment(experiment):
    for experiment in file.values():
        command = "python ptb-lm.py --model={} --optimizer={} --initial_lr={} --batch_size={} --hidden_size={} --num_layers={} --dp_keep_prob={}"
        command = command.format(
            experiment.get('model'),
            experiment.get('optimizer'),
            experiment.get('learning_rate'),
            experiment.get('batch_size'),
            experiment.get('hidden_size'),
            experiment.get('num_layers'),
            experiment.get('dropout')
        )
        return call(command)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='hyperparameters.json', help='JSON with configurations')
    args = parser.parse_args()

    with open(args.path, 'r') as f:
        file = json.load(f)

    tasks = [x for x in file.values()]
    pool = multiprocessing.Pool(processes=16)
    results = []
    r = pool.map_async(run_experiment, tasks, callback=results.append)
    r.wait()  # Wait on the results
    print(results)
    print('Finished')


