import json
import os
import sys

import fire
import pandas as pd
from zeno import DistillReturn, ModelReturn, distill, metric, model, zeno
from zeno.api import DistillReturn, MetricReturn, ZenoOptions


@model
def model_fn(name):
    def mod(df, ops):
        return ModelReturn(model_output=df["sample"])

    return mod


@distill
def match(df, ops):
    matches = []
    for i, row in df.iterrows():
        matches.append(row[ops.output_column] == row["ideal"])
    return DistillReturn(distill_output=matches)


@metric
def accuracy(df, ops: ZenoOptions):
    m = df[ops.distill_columns["match"]].astype(int).mean()
    if pd.notna(m):
        return MetricReturn(metric=m)
    return MetricReturn(metric=0)


def main(file: str, template: str = "Match"):
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
        if d["type"] == "raw_sample"
    ]
    samples = [
        {"id": d["sample_id"], "sample": d["data"]["sampled"]}
        for d in data[2:]
        if d["type"] == "sampling"
    ]

    df_samples = pd.DataFrame(samples)
    df = pd.DataFrame(raw_samples)
    df = df.join(df_samples.set_index("id"), on="id")

    zeno(
        {
            "metadata": df,
            "models": [data[0]["spec"]["model_name"]],
            "functions": [model_fn, match, accuracy],
            "view": "openai-chat",
            "data_column": "input",
            "id_column": "id",
            "label_column": "ideal",
            "port": 8080,
        }
    )


def cli():
    fire.Fire(main)
