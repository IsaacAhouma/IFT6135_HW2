import json
import argparse
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', default='hyperparameters.json', help='JSON with configurations')
    args = parser.parse_args()

    with open(args.path, 'r') as f:
        file = json.load(f)

    for experiment in file.values():
        command = "msub -v model='{}',optimizer='{}',lr='{}',batch='{}',hidden='{}',layers='{}',dropout='{}' run.pbs"
        command.format(
            experiment.get('model'),
            experiment.get('optimizer'),
            experiment.get('learning_rate'),
            experiment.get('batch_size'),
            experiment.get('hidden_size'),
            experiment.get('num_layers'),
            experiment.get('dropout')
        )
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        returncode = process.wait()
        stdout = process.communicate()[0]
        print(stdout)
