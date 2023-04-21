import json
import os
import sys
from contextlib import contextmanager
from importlib import util
from inspect import getmembers, isfunction
from typing import Dict

import fire
import pandas as pd
from pandas import DataFrame
from zeno import (
    DistillReturn,
    ModelReturn,
    ZenoParameters,
    distill,
    metric,
    model,
    zeno,
)
from zeno.api import MetricReturn, ZenoOptions


class ZenoEvals:
    def __init__(
        self,
        results_file: str,
        second_results_file: str = None,
        functions_file: str = None,
    ):
        self.dfs = {}
        self.results_file = results_file
        self.second_results_file = second_results_file
        self.functions_file = functions_file

    def generate_zeno_config(self) -> ZenoParameters:
        if not os.path.exists(self.results_file):
            print("ERROR: file '{}' does not exist.".format(self.results_file))
            sys.exit(1)

        data = []
        with open(self.results_file) as f:
            for line in f:
                data.append(json.loads(line))

        df, metric_names = read_results_file(data)
        self.dfs[get_model_name(data)] = df

        models = [get_model_name(data)]
        if self.second_results_file is not None:
            data2 = []
            with open(self.second_results_file) as f:
                for line in f:
                    data2.append(json.loads(line))

            models.append(get_model_name(data2))
            df2, _ = read_results_file(data2)
            self.dfs[get_model_name(data2)] = df2

        functions = [get_model_fn(self.dfs)]
        if self.functions_file is not None:
            functions = functions + parse_testing_file(self.functions_file)

        base_df_columns = ["id", "prompt"]

        for m in metric_names:
            functions = functions + [get_metric_function(m)]

        if "expected" in df.columns:
            functions = functions + [get_correct_fn(self.dfs), avg_correct]
            base_df_columns = base_df_columns + ["expected"]

        zeno_config = ZenoParameters(
            metadata=df[base_df_columns],
            models=models,
            functions=functions,
            view="openai-chat",
            data_column="prompt",
            id_column="id",
            cache_path="./.zeno_cache_" + data[0]["spec"]["eval_name"],
            port=8080,
            batch_size=100,
            samples=5,
        )

        if "expected" in df.columns:
            zeno_config.label_column = "expected"

        return zeno_config


def get_model_fn(dfs: Dict[str, DataFrame]):
    def model_fn(name):
        model_df = dfs[name]

        def mod(df, ops: ZenoOptions):
            return ModelReturn(model_output=model_df["sampled"].loc[df["id"]].tolist())

        return mod

    return model(model_fn)


def get_correct_fn(dfs: Dict[str, DataFrame]):
    def correct(df, ops: ZenoOptions):
        mod = [mod for mod in list(dfs.keys()) if mod in ops.output_column][0]
        return DistillReturn(distill_output=dfs[mod]["correct"].loc[df["id"]])

    return distill(correct)


@contextmanager
def add_to_path(p):
    old_path = sys.path
    sys.path = sys.path[:]
    sys.path.insert(0, p)
    try:
        yield
    finally:
        sys.path = old_path


def parse_testing_file(test_file):
    # To allow relative imports in test files,
    # add their directory to path temporarily.
    with add_to_path(os.path.dirname(os.path.abspath(test_file))):
        spec = util.spec_from_file_location(str(test_file), test_file)
        test_module = util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(test_module)  # type: ignore

    functions = []
    for func_name, func in getmembers(test_module):
        if isfunction(func):
            if (
                hasattr(func, "predict_function")
                or hasattr(func, "distill_function")
                or hasattr(func, "metric_function")
                or hasattr(func, "inference_function")
            ):
                functions.append(func)
    return functions


def get_metric_function(metric_name):
    def metric_function(df, ops: ZenoOptions):
        if len(df) == 0:
            return MetricReturn(metric=0.0)
        if (
            df[metric_name].dtype == "object"
            and df[metric_name].value_counts().shape[0] <= 2
        ):
            return MetricReturn(
                metric=df[metric_name].eq(df[metric_name][0]).mul(1).mean()
            )
        return MetricReturn(metric=df[metric_name].mean())

    metric_function.__name__ = metric_name

    return metric(metric_function)


@metric
def avg_correct(df, ops: ZenoOptions):
    return MetricReturn(
        metric=df[ops.distill_columns["correct"]].astype(int).mean() * 100
    )


def read_results_file(data):
    data_res = [d for d in data if "event_id" in d]
    sampling_df = pd.DataFrame(
        [
            {
                "id": d["sample_id"],
                "prompt": d["data"]["prompt"],
                "sampled": d["data"]["sampled"][0],
            }
            for d in data_res
            if "type" in d and d["type"] == "sampling"
        ]
    )

    match_df = pd.DataFrame(
        [
            {
                "id": d["sample_id"],
                "correct": d["data"]["correct"],
                "expected": d["data"]["expected"],
            }
            for d in data_res
            if "type" in d and d["type"] == "match"
        ]
    )

    metric_names = []
    for d in data_res:
        if "type" in d and d["type"] == "metrics":
            metric_names = list(d["data"].keys())
            break

    metrics = []
    for d in data_res:
        if "type" in d and d["type"] == "metrics":
            met_obj = {"id": d["sample_id"]}
            for name in metric_names:
                met_obj[name] = d["data"][name]
            metrics.append(met_obj)
    metrics_df = pd.DataFrame(metrics)

    df = sampling_df
    if len(match_df) > 0:
        df = df.join(match_df.set_index("id"), on="id")
    if len(metrics_df) > 0:
        df = df.join(metrics_df.set_index("id"), on="id")

    df.set_index("id", inplace=True, drop=False)
    return df, metric_names


def get_model_name(data):
    name = data[0]["spec"]["completion_fns"][0]
    return name.replace(".", "_")


def main(
    results_file: str, second_results_file: str = None, functions_file: str = None
):
    """Visualize a result from OpenAI evals using Zeno.

    Args:
            results_file (path): Result .jsonl file from OpenAI evals.
            Often stored in the /tmp/evallogs/ directory.

            second_results_file (path): Second result .jsonl file from OpenAI
            evals for comparison. Often stored in the /tmp/evallogs/ directory.

            functions_file (path, optional): Path to a Python file containing
            additional Zeno processing functions. Defaults to None.
    """

    eval = ZenoEvals(results_file, second_results_file, functions_file)
    config = eval.generate_zeno_config()
    zeno(config)


def cli():
    fire.Fire(main)
