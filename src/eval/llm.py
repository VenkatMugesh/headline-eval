import asyncio
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import AsyncGroq
from pydantic import BaseModel, ValidationError

load_dotenv(Path(__file__).resolve().parents[1] / "api-key.env")

MODEL_ID = "qwen/qwen3.8-27b"
REASONING_EFFORT = "none"

class LLMOutputInvalid(Exception):
    pass


async def complete(client: AsyncGroq, messages: list[dict], max_out: int = 400, model_id: str = MODEL_ID, reasoning_effort: str = REASONING_EFFORT) -> tuple[str, int, int, str]:
    try:
        response = await client.chat.completions.create(
            model=model_id,
            max_tokens=max_out,
            messages=messages,
            temperature=0,
            reasoning_effort=reasoning_effort,
        )
        return response.choices[0].message.content, response.usage.prompt_tokens, response.usage.completion_tokens, response.choices[0].finish_reason
    except Exception as e:  # noqa: BLE001 — wrap any client failure as LLMOutputInvalid
        raise LLMOutputInvalid(f"failed to complete: {e}")

async def get_structured[T: BaseModel](
    client: AsyncGroq, messages: list[dict], model_cls: type[T], max_attempts: int = 2, model_id: str = MODEL_ID, reasoning_effort: str = REASONING_EFFORT) -> T:
    msgs = list(messages)                       
    for attempt in range(1, max_attempts + 1):
        text, _, _, finish = await complete(client, msgs, max_out=400, model_id=model_id, reasoning_effort=reasoning_effort)
        if finish == "length":
            raise LLMOutputInvalid("truncated: raise max_out")   
        try:
            return model_cls.model_validate_json(text)
        except ValidationError as e:
            msgs.append({"role": "assistant", "content": text})
            msgs.append({"role": "user", "content":
                "Your JSON failed validation:\n" + format_errors(e) + "\nReturn corrected JSON only."})
    raise LLMOutputInvalid(f"still invalid after {max_attempts} attempts")

def format_errors(e: ValidationError) -> str:
    lines = []
    for err in e.errors():
        path = ".".join(str(part) for part in err["loc"]) or "(whole response)"
        lines.append(f"- {path}: {err['msg']}")
    return "\n".join(lines)



class Pacer:
    """Lets at most `per_minute` calls start per minute, evenly spaced."""
    def __init__(self, per_minute: int):
        self.interval = 60 / per_minute    
        self.next_at = 0.0                  
        self.lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self.lock:
            now = time.monotonic()
            if now < self.next_at:
                await asyncio.sleep(self.next_at - now)
            self.next_at = max(now, self.next_at) + self.interval