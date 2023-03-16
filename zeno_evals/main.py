from contextlib import contextmanager
from inspect import getmembers, isfunction
import json
import os
import re
import string
import sys
from importlib import util

import fire
import pandas as pd
from zeno import DistillReturn, ModelReturn, distill, metric, model, zeno
from zeno.api import DistillReturn, MetricReturn, ZenoOptions


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


def normalize(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.split("\n")[0]
    s = s.lower()
    exclude = set(string.punctuation)
    s = "".join(char for char in s if char not in exclude)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


@model
def model_fn(name):
    def mod(df, ops):
        return ModelReturn(model_output=df["sample"])

    return mod


@distill
def match(df, ops):
    matches = []
    for i, row in df.iterrows():
        if type(row["ideal"]) == list:
            matches.append(
                any(
                    normalize(b).startswith(normalize(row[ops.output_column]))
                    for b in row["ideal"]
                )
            )
        else:
            matches.append(
                normalize(row["ideal"]).startswith(normalize(row[ops.output_column]))
            )
    return DistillReturn(distill_output=matches)


@distill
def includes(df, ops):
    matches = []
    for i, row in df.iterrows():
        if type(row["ideal"]) == list:
            matches.append(
                any(
                    normalize(row[ops.output_column]) in normalize(b)
                    for b in row["ideal"]
                )
            )
        else:
            matches.append(normalize(row[ops.output_column]) in normalize(row["ideal"]))
    return DistillReturn(distill_output=matches)


@distill
def fuzzy_match(df, ops):
    matches = []
    for i, row in df.iterrows():
        if type(row["ideal"]) == list:
            matches.append(
                any(
                    normalize(b) in normalize(row[ops.output_column])
                    or normalize(row[ops.output_column]) in normalize(b)
                    for b in row["ideal"]
                )
            )
        else:
            matches.append(normalize(row[ops.output_column]) in normalize(row["ideal"]))
    return DistillReturn(distill_output=matches)


def get_match_function(template):
    if template == "match":
        return match
    elif template == "includes":
        return includes
    elif template == "fuzzy_match":
        return fuzzy_match
    else:
        raise ValueError("Invalid template: {}".format(template))


def get_accuracy_function(template):
    @metric
    def accuracy(df, ops: ZenoOptions):
        m = df[ops.distill_columns[template]].astype(int).mean()
        if pd.notna(m):
            return MetricReturn(metric=m * 100)
        return MetricReturn(metric=0)

    return accuracy


def main(file: str, template: str = "match", functions_file: str = None):
    """Visualize a result from OpenAI evals using Zeno.

    Args:
            file (path): Result .jsonl file from OpenAI evals. Often stored in the /tmp/evallogs/ directory.

            template (str, optional): Evaluation template to use. Can be "match", "includes" or "fuzzy_match". Defaults to "match".

            functions_file (path, optional): Path to a Python file containing additional Zeno processing functions. Defaults to None.
    """

    if not os.path.exists(file):
        print("ERROR: file '{}' does not exist.".format(file))
        sys.exit(1)

    data = []
    with open(file) as f:
        for line in f:
            data.append(json.loads(line))

    raw_samples = [
        {"input": d["data"]["input"], "id": d["sample_id"], "ideal": d["data"]["ideal"]}
        for d in data[2:]
        if "type" in d and d["type"] == "raw_sample"
    ]
    samples = [
        {"id": d["sample_id"], "sample": d["data"]["sampled"]}
        for d in data[2:]
        if "type" in d and d["type"] == "sampling"
    ]

    df_samples = pd.DataFrame(samples)
    df = pd.DataFrame(raw_samples)
    df = df.join(df_samples.set_index("id"), on="id")

    extra_fns = []
    if functions_file is not None:
        extra_fns = parse_testing_file(functions_file)

    zeno(
        {
            "metadata": df,
            "models": [data[0]["spec"]["model_name"]],
            "functions": [
                model_fn,
                get_accuracy_function(template),
                get_match_function(template),
            ]
            + extra_fns,
            "view": "openai-chat",
            "data_column": "input",
            "id_column": "id",
            "label_column": "ideal",
            "cache_path": "./.zeno_cache_"
            + template
            + os.path.basename(file).split("_")[0],
            "port": 8080,
            "batch_size": 100,
        }
    )


def cli():
    fire.Fire(main)
