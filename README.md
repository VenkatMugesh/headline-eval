# Headline eval

Eval harness for ad-headline prompts. It generates one headline per case through Groq, grades the output with deterministic checks, and on v2 adds an LLM tone judge. A gate compares the candidate run with the saved baseline and exits non-zero when code-check quality drops.

## Prompts

Both prompts live in `src/eval/run.py`.

- **v1** asks for an ad headline for a product, audience, and tone.
- **v2** asks for one headline, a word cap, the exact product name, and plain text with no quotes, lists, or explanation.

The generator is `qwen/qwen3.8-27b` at temperature 0, `reasoning_effort=none`, and `max_tokens=60`.

## Cases

`src/eval/cases.jsonl` has 20 cases. Each case is a product, an audience, a tone (`playful`, `premium`, `urgent`, or `friendly`), and a `max_words` limit.

## Graders

Code graders in `src/eval/graders.py` run on every version:

| Criterion | Passes when |
| --- | --- |
| `word_limit` | Word count is at most `max_words` |
| `mentions_product` | Output contains the product name, case-insensitive |
| `no_banned_words` | Output avoids `guaranteed`, `risk-free`, `100%`, `#1`, and `miracle` |
| `single_line` | Output is one line and does not start with `-`, `*`, or `1.` |

Tone is graded only on v2, by `openai/gpt-oss-120b` with `reasoning_effort=medium`. The judge returns JSON `{"passed": true or false, "reason": "one sentence"}` and is retried once if that JSON fails validation. Tone definitions in the judge prompt:

- **playful** — light, witty, fun
- **premium** — elegant, exclusive, calm
- **urgent** — time pressure, act now
- **friendly** — warm, simple, conversational

## Gate

`run_gate` reads `src/eval/results/v1.json` as the baseline and `src/eval/results/v2.json` as the candidate. It scores the four code criteria. Tone stays out of the gate.

The gate fails when any of these is true:

- a candidate run recorded an error
- a code criterion is under 90%
- a code criterion falls by more than one case versus the baseline (`1 / n_cases`)

Exit code is `0` on pass and `1` on fail.

Checked-in results (20 cases, 0 errors):

| Criterion | v1 | v2 |
| --- | ---: | ---: |
| `word_limit` | 0% | 100% |
| `mentions_product` | 85% | 95% |
| `no_banned_words` | 100% | 100% |
| `single_line` | 0% | 100% |
| `tone` | — | 75% |

On those files the code-criterion gate passes. Tone at 75% is stored with the v2 run and is outside the gate.

## Human check

`src/eval/human_labels.json` stores the first 10 v2 outputs. Each row has a boolean under a key equal to the case id (`"c01": false`) — the human tone label. `judge_agreement` prints how often that label matches the stored judge grade. On the checked-in file the match is 10/10.

## Setup

Python 3.12 or newer (the client uses PEP 695 generics). From the repo root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install groq pydantic python-dotenv
```

Put the Groq key in `src/api-key.env` (loaded by `llm.py` and `run.py`):

```
GROQ_API_KEY=your-key
```

Generation and judging are paced at 24 call starts per minute.

## Run

A fresh eval overwrites `src/eval/results/v1.json` and `src/eval/results/v2.json`, then prints pass rates, error counts, and every failing grade:

```bash
PYTHONPATH=src python -c "import asyncio; from eval.run import main; asyncio.run(main())"
```

The module entry point runs the gate on the saved result files:

```bash
PYTHONPATH=src python -m eval.run
```

## Layout

```
src/eval/
  run.py               prompts, eval loop, report, gate
  graders.py           deterministic checks
  llm.py               Groq client, JSON retry, rate pacer
  schemas.py           Case, Grade, JudgeVerdict, CaseResult
  cases.jsonl          20 eval cases
  human_labels.json    human tone labels for the first 10 v2 outputs
  results/v1.json      saved baseline
  results/v2.json      saved candidate
```
