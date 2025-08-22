import os

import config_loader

CONFIG_ENV_VAR = "GRU4REC_RUN_CONFIG"
DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "run.json")


def main():
    config_path = os.environ.get(CONFIG_ENV_VAR, DEFAULT_CONFIG_PATH)
    args = config_loader.load_run_config(config_path)

    orig_cwd = os.getcwd()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    import numpy as np  # noqa: F401
    import pandas as pd
    import datetime as dt  # noqa: F401
    import sys
    import time
    from collections import OrderedDict
    import importlib
    GRU4Rec = importlib.import_module(args.gru4rec_model).GRU4Rec
    import evaluation
    import importlib.util
    import joblib
    os.chdir(orig_cwd)

    def load_data(fname, args):
        if fname.endswith(".pickle"):
            print(f"Loading data from pickle file: {fname}")
            data = joblib.load(fname)
            if args.session_key not in data.columns:
                print(
                    'ERROR. The column specified for session IDs "{}" is not in the data file ({})'.format(
                        args.session_key, fname
                    )
                )
                print(
                    'The default column name is "SessionId", but you can specify otherwise by setting the `session_key` parameter of the model.'
                )
                sys.exit(1)
            if args.item_key not in data.columns:
                print(
                    'ERROR. The column specified for item IDs "{}" is not in the data file ({})'.format(
                        args.item_key, fname
                    )
                )
                print(
                    'The default column name is "ItemId", but you can specify otherwise by setting the `item_key` parameter of the model.'
                )
                sys.exit(1)
            if args.time_key not in data.columns:
                print(
                    'ERROR. The column specified for time "{}" is not in the data file ({})'.format(
                        args.time_key, fname
                    )
                )
                print(
                    'The default column name is "Time", but you can specify otherwise by setting the `time_key` parameter of the model.'
                )
                sys.exit(1)
        else:
            with open(fname, "rt") as f:
                header = f.readline().strip().split("\t")
            if args.session_key not in header:
                print(
                    'ERROR. The column specified for session IDs "{}" is not in the data file ({})'.format(
                        args.session_key, fname
                    )
                )
                print(
                    'The default column name is "SessionId", but you can specify otherwise by setting the `session_key` parameter of the model.'
                )
                sys.exit(1)
            if args.item_key not in header:
                print(
                    'ERROR. The colmn specified for item IDs "{}" is not in the data file ({})'.format(
                        args.item_key, fname
                    )
                )
                print(
                    'The default column name is "ItemId", but you can specify otherwise by setting the `item_key` parameter of the model.'
                )
                sys.exit(1)
            if args.time_key not in header:
                print(
                    'ERROR. The column specified for time "{}" is not in the data file ({})'.format(
                        args.time_key, fname
                    )
                )
                print(
                    'The default column name is "Time", but you can specify otherwise by setting the `time_key` parameter of the model.'
                )
                sys.exit(1)
            print(f"Loading data from TAB separated file: {fname}")
            data = pd.read_csv(
                fname,
                sep="\t",
                usecols=[args.session_key, args.item_key, args.time_key],
                dtype={args.session_key: "int32", args.item_key: "str"},
            )
        return data

    if (args.parameter_string is not None) + (args.parameter_file is not None) + (args.load_model) != 1:
        print(
            "ERROR. Exactly one of the following parameters must be provided: --parameter_string, --parameter_file, --load_model"
        )
        sys.exit(1)

    if args.load_model:
        print(f'Loading trained model from file: {args.path} (to device "{args.device}")')
        gru = GRU4Rec.loadmodel(args.path, device=args.device)
    else:
        if args.parameter_file:
            param_file_path = os.path.abspath(args.parameter_file)
            param_dir, param_file = os.path.split(param_file_path)
            spec = importlib.util.spec_from_file_location(param_file.split(".py")[0], os.path.abspath(args.parameter_file))
            params = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(params)
            gru4rec_params = params.gru4rec_params
            print(f"Loaded parameters from file: {param_file_path}")
        if args.parameter_string:
            gru4rec_params = OrderedDict([x.split("=") for x in args.parameter_string.split(",")])
        print(f'Creating GRU4Rec model on device "{args.device}"')
        gru = GRU4Rec(device=args.device)
        gru.set_params(**gru4rec_params)
        print("Loading training data...")
        data = load_data(args.path, args)
        print("Started training")
        t0 = time.time()
        gru.fit(
            data,
            sample_cache_max_size=args.sample_store_size,
            item_key=args.item_key,
            session_key=args.session_key,
            time_key=args.time_key,
        )
        t1 = time.time()
        print("Total training time: {:.2f}s".format(t1 - t0))
        if args.save_model is not None:
            print(f"Saving trained model to: {args.save_model}")
            gru.savemodel(args.save_model)

    if args.test is not None:
        if args.primary_metric.lower() == "recall":
            pm_index = 0
        elif args.primary_metric.lower() == "mrr":
            pm_index = 1
        else:
            raise RuntimeError(
                "Invalid value `{}` for `primary_metric` parameter".format(args.primary_metric)
            )
        for test_file in args.test:
            print("Loading test data...")
            test_data = load_data(test_file, args)
            print(
                "Starting evaluation (cut-off={}, using {} mode for tiebreaking)".format(
                    args.measure, args.eval_type
                )
            )
            t0 = time.time()
            res = evaluation.batch_eval(
                gru,
                test_data,
                batch_size=512,
                cutoff=args.measure,
                mode=args.eval_type,
                item_key=args.item_key,
                session_key=args.session_key,
                time_key=args.time_key,
            )
            t1 = time.time()
            print("Evaluation took {:.2f}s".format(t1 - t0))
            for c in args.measure:
                print("Recall@{}: {:.6f} MRR@{}: {:.6f}".format(c, res[0][c], c, res[1][c]))

            if args.log_primary_metric:
                print("PRIMARY METRIC: {}".format([x for x in res[pm_index].values()][0]))


if __name__ == "__main__":
    main()

