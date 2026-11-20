
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
