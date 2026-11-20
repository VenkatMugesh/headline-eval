import asyncio
import json
import os
import sys
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path

from dotenv import load_dotenv
from groq import AsyncGroq

from .graders import CODE_GRADERS
from .llm import Pacer, complete, get_structured
from .schemas import Case, CaseResult, Grade, JudgeVerdict

JUDGE_MODEL_ID = "openai/gpt-oss-120b"

load_dotenv(Path(__file__).resolve().parents[1] / "api-key.env")

PROMPTS = {
    "v1": "Write an ad headline for {product} targeting {audience} in a {tone} tone.",
    "v2": (
        "Write ONE ad headline for {product} targeting {audience}. Tone: {tone}. "
        "At most {max_words} words. Include the exact product name. "
        "Plain text only: no quotes, no lists, no explanation."
    ),
}

JUDGE_PROMPT = (
    "You grade ad headlines. Requested tone: {tone}.\n"
    "Tone definitions: playful = light, witty, fun; premium = elegant, exclusive, calm; "
    "urgent = time pressure, act now; friendly = warm, simple, conversational.\n"
    "Headline: {output}\n"
    'Does the headline match the requested tone? Reply with JSON only: {{"passed": true or false, "reason": "one sentence"}}'
)


async def tone_judge(client: AsyncGroq, case: Case, output: str) -> Grade:
    msgs = [{"role": "user", "content": JUDGE_PROMPT.format(tone=case.tone, output=output)}]
    verdict = await get_structured(
        client, msgs, JudgeVerdict, model_id=JUDGE_MODEL_ID, reasoning_effort="medium"
    )
    return Grade(criterion="tone", passed=verdict.passed, detail=verdict.reason)


async def generate(client: AsyncGroq, case: Case, version: str) -> str:
    prompt = PROMPTS[version].format(**case.model_dump())
    text, _, _, _ = await complete(client, [{"role": "user", "content": prompt}], max_out=60)
    return text.strip()


def pass_rates(results: list[CaseResult]) -> dict[str, dict[str, float]]:
    passed: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for result in results:
        if result.error:
            continue
        for grade in result.grades:
            total[result.version][grade.criterion] += 1
            passed[result.version][grade.criterion] += int(grade.passed)
    return {
        version: {criterion: passed[version][criterion] / total[version][criterion] for criterion in total[version]}
        for version in total
    }


def gate(new: dict[str, float], base: dict[str, float], n_cases: int, errors: int) -> list[str]:
    problems = []
    if errors:
        problems.append(f"{errors} runs errored; results incomplete")
    one_case = 1 / n_cases
    for criterion, rate in new.items():
        if rate < 0.90:
            problems.append(f"{criterion}: {rate:.0%} is below 90%")
        if criterion in base and base[criterion] - rate > one_case + 1e-9:
            problems.append(f"{criterion}: dropped {base[criterion]:.0%} → {rate:.0%}")
    return problems


def report(results: list[CaseResult]) -> None:
    rates = pass_rates(results)
    v1, v2 = rates.get("v1", {}), rates.get("v2", {})
    criteria = list(dict.fromkeys([*v1, *v2]))

    print(f"{'criterion':<20} {'v1':>6} {'v2':>6} {'diff':>6}")
    for criterion in criteria:
        before, after = v1.get(criterion, 0.0), v2.get(criterion, 0.0)
        print(f"{criterion:<20} {before:>5.0%} {after:>5.0%} {round((after - before) * 100):>+5}")

    print()
    for version in ("v1", "v2"):
        n = sum(1 for result in results if result.version == version and result.error)
        print(f"errors: {version}={n}")

    print()
    for result in results:
        if result.error:
            print(f"{result.case_id} {result.version} error  {result.error}")
            continue
        for grade in result.grades:
            if grade.passed:
                continue
            text = (result.output or "").replace("\n", " ")
            print(f"{result.case_id} {result.version} {grade.criterion}  {grade.detail}")
            print(f"  output: {text}")


def load_cases(path: Path) -> Iterator[Case]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield Case.model_validate_json(line)


def judge_agreement(rows: list[dict]) -> None:
    labeled = rows[:10]
    matches = 0
    disagreements = []
    for row in labeled:
        yours = row[row["case_id"]]
        tone = next(grade for grade in row["grades"] if grade["criterion"] == "tone")
        if yours == tone["passed"]:
            matches += 1
        else:
            disagreements.append((row, yours, tone))

    print(f"agreement: {matches}/10 = {matches / 10:.0%}")
    for row, yours, tone in disagreements:
        print(row["output"])
        print(f"  you: {yours}  judge: {tone['passed']}  reason: {tone['detail']}")


async def run_eval(
    client: AsyncGroq, cases: list[Case], version: str, pacer: Pacer
) -> list[CaseResult]:
    async def one(case: Case) -> CaseResult:
        await pacer.wait()
        output = await generate(client, case, version)
        grades = [grader(case, output) for grader in CODE_GRADERS]
        if version == "v2":
            grades.append(await tone_judge(client, case, output))
        return CaseResult(case_id=case.id, version=version, output=output, grades=grades)

    raw = await asyncio.gather(*(one(case) for case in cases), return_exceptions=True)

    results = []
    for case, item in zip(cases, raw):
        if isinstance(item, Exception):
            results.append(CaseResult(case_id=case.id, version=version, error=str(item)))
        else:
            results.append(item)
    return results


async def main() -> None:
    async with AsyncGroq(api_key=os.getenv("GROQ_API_KEY")) as client:
        cases = list(load_cases(Path(__file__).parent / "cases.jsonl"))
        all_results = []
        results_dir = Path(__file__).parent / "results"
        results_dir.mkdir(exist_ok=True)
        for version in PROMPTS:
            results = await run_eval(client, cases, version, Pacer(24))
            all_results.extend(results)
            (results_dir / f"{version}.json").write_text(
                json.dumps([result.model_dump() for result in results], indent=2)
            )
        report(all_results)


def rates_for(results: list[CaseResult]) -> dict[str, float]:
    rates = pass_rates(results)
    version = results[0].version
    return {criterion: rate for criterion, rate in rates.get(version, {}).items() if criterion != "tone"}


def run_gate() -> None:
    root = Path(__file__).parent / "results"
    baseline = [CaseResult.model_validate(row) for row in json.loads((root / "v1.json").read_text())]
    candidate = [CaseResult.model_validate(row) for row in json.loads((root / "v2.json").read_text())]
    problems = gate(
        rates_for(candidate),
        rates_for(baseline),
        n_cases=len(candidate),
        errors=sum(1 for result in candidate if result.error),
    )
    for problem in problems:
        print(problem)
    if problems:
        print("GATE: FAIL")
        sys.exit(1)
    print("GATE: PASS")
    sys.exit(0)


if __name__ == "__main__":
    run_gate()
