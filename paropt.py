import os
import json
import optuna
import re
import tempfile
from types import SimpleNamespace

import config_loader
import pexpect

CONFIG_ENV_VAR = "GRU4REC_PAROPT_CONFIG"
DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "config", "paropt.json"
)


def generate_command(config, optimized_param_str, measure, log_primary_metric=True):
    """Create command to invoke run.py with a temporary config file."""
    param_parts = [config.fixed_parameters] if config.fixed_parameters else []
    if optimized_param_str:
        param_parts.append(optimized_param_str)

    run_config = {
        "path": config.path,
        "test": config.test if isinstance(config.test, list) else [config.test],
        "gru4rec_model": config.gru4rec_model,
        "parameter_string": ",".join(param_parts),
        "measure": measure,
        "eval_type": config.eval_type,
        "device": config.device,
        "item_key": config.item_key,
        "session_key": config.session_key,
        "time_key": config.time_key,
        "primary_metric": config.primary_metric,
        "log_primary_metric": log_primary_metric,
    }

    tmp = tempfile.NamedTemporaryFile("w", delete=False, suffix=".json")
    json.dump(run_config, tmp)
    tmp.close()
    command = f'GRU4REC_RUN_CONFIG="{tmp.name}" python run.py'
    return command, tmp.name


def run_once(config, optimized_param_str):
    command, cfg_path = generate_command(config, optimized_param_str, config.measure, True)
    cmd = pexpect.spawnu(command, timeout=None, maxread=1)
    line = cmd.readline()
    val = None
    while line:
        line = line.strip()
        print(line)
        if re.match('PRIMARY METRIC: -*\\d\\.\\d+e*-*\\d*', line):
            t = line.split(':')[1].lstrip()
            val = float(t)
            break
        line = cmd.readline()
    cmd.wait()
    os.remove(cfg_path)
    return val


class Parameter:
    def __init__(self, name, dtype, values, step=None, log=False):
        assert dtype in ['int', 'float', 'categorical']
        assert type(values) == list
        assert len(values) == 2 or dtype == 'categorical'
        self.name = name
        self.dtype = dtype
        self.values = values
        self.step = step
        if self.step is None and self.dtype == 'int':
            self.step = 1
        self.log = log

    @classmethod
    def fromjson(cls, json_string):
        obj = json.loads(json_string)
        return Parameter(
            obj['name'],
            obj['dtype'],
            obj['values'],
            obj['step'] if 'step' in obj else None,
            obj['log'] if 'log' in obj else False,
        )

    def __call__(self, trial):
        if self.dtype == 'int':
            return trial.suggest_int(
                self.name, int(self.values[0]), int(self.values[1]), step=self.step, log=self.log
            )
        if self.dtype == 'float':
            return trial.suggest_float(
                self.name, float(self.values[0]), float(self.values[1]), step=self.step, log=self.log
            )
        if self.dtype == 'categorical':
            return trial.suggest_categorical(self.name, self.values)

    def __str__(self):
        desc = 'PARAMETER {} \t type={}'.format(self.name, self.dtype)
        if self.dtype == 'int' or self.dtype == 'float':
            desc += ' \t range=[{}..{}] (step={}) \t {} scale'.format(
                self.values[0],
                self.values[1],
                self.step if self.step is not None else 'N/A',
                'UNIFORM' if not self.log else 'LOG',
            )
        if self.dtype == 'categorical':
            desc += ' \t options: [{}]'.format(','.join([str(x) for x in self.values]))
        return desc


def objective(trial, par_space, config):
    optimized_param_str = []
    for par in par_space:
        val = par(trial)
        optimized_param_str.append('{}={}'.format(par.name, val))
    optimized_param_str = ','.join(optimized_param_str)
    val = run_once(config, optimized_param_str)
    return val


def main():
    config_path = os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)
    config = SimpleNamespace(**config_loader.load_paropt_config(config_path))

    par_space = []
    with open(config.optuna_parameter_file, 'rt') as f:
        print('-' * 80)
        print('PARAMETER SPACE')
        for line in f:
            par = Parameter.fromjson(line)
            print('\t' + str(par))
            par_space.append(par)
        print('-' * 80)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, par_space, config), n_trials=config.ntrials)

    print('Running final eval @{}:'.format(config.final_measure))
    optimized_param_str = ','.join(['{}={}'.format(k, v) for k, v in study.best_params.items()])
    command, cfg_path = generate_command(config, optimized_param_str, config.final_measure, False)
    cmd = pexpect.spawnu(command, timeout=None, maxread=1)
    line = cmd.readline()
    while line:
        line = line.strip()
        print(line)
        line = cmd.readline()
    cmd.wait()
    os.remove(cfg_path)


if __name__ == "__main__":
    main()

