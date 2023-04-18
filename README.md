# Zeno 🤝 OpenAI Evals

Use [Zeno](https://github.com/zeno-ml/zeno) to visualize the results of [OpenAI Evals](https://github.com/openai/evals/blob/main/docs/eval-templates.md).

https://user-images.githubusercontent.com/4563691/225655166-9fd82784-cf35-47c1-8306-96178cdad7c1.mov

_Example using `zeno-evals` to explore the results of an OpenAI eval on multiple choice medicine questions (MedMCQA)_

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

Single example looking at US tort law questions:

```bash
zeno-evals ./examples/example.jsonl
```

And an example of comparison between two models:

```bash
zeno-evals ./examples/crossword-turbo.jsonl --second-results-file ./examples/crossword-turbo-0301.jsonl
```

And lastly, we can pass additional [Zeno functions](https://zenoml.com/docs/api) to provide more context to the results:

```bash
pip install wordfreq
zeno-evals ./examples/crossword-turbo.jsonl --second-results-file ./examples/crossword-turbo-0301.jsonl --functions_file ./examples/crossword_fns.py
```
