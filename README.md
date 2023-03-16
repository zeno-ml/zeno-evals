# Zeno 🤝 OpenAI Evals

Use [Zeno](https://github.com/zeno-ml/zeno) to visualize the results of [OpenAI Evals](https://github.com/openai/evals/blob/main/docs/eval-templates.md).

### Usage

```bash
pip install zeno-evals
```

Run an evaluation following the [evals instructions](https://github.com/openai/evals/blob/main/docs/run-evals.md). This will produce a cache file in `/tmp/evallogs/`.

Pass this file to the `zeno-evals` command:

```bash
zeno-evals /tmp/evallogs/my_eval_cache.jsonl
```

### Example

We include an example looking at the [MedMCQA](https://github.com/openai/evals/pull/141) dataset (Thanks to @SinanAkkoyun):

```bash
zeno-evals ./example_medicine/example.jsonl --functions_file=./example_medicine/distill.py
```

### Todo

- [ ] Support model-graded evaluations
- [ ] Support custom evaluation templates (e.g. BLEU for translation)
