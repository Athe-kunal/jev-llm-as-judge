"""Async client for a local vLLM server, with a SQLite response cache."""

import asyncio
import hashlib
import json
import sqlite3
import time
from typing import Any

import openai


def chat_prefix(user: str, think: bool = False) -> str:
    """Qwen chat format for one user message, up to the reply.

    Args:
      user: The user message.
      think: Let the model reason in a <think> block first (default: an
        empty think block, so it answers at once).
    """
    head = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return head + ("<think>\n" if think else "<think>\n\n</think>\n\n")


class Client:
    """Completion client with a cache and a ledger of token counts."""

    def __init__(self, cfg: dict[str, Any]):
        """Creates the client.

        Args:
          cfg: Parsed config with `llm` (base_url, model, concurrency,
            timeout_s) and `data_dir`.
        """
        llm_cfg = cfg["llm"]
        self.model = llm_cfg["model"]
        self.stats = {"calls": 0, "cached": 0}
        self._api = openai.AsyncOpenAI(
            base_url=llm_cfg["base_url"],
            api_key="none",
            timeout=llm_cfg.get("timeout_s", 300),
        )
        self._sem = asyncio.Semaphore(llm_cfg["concurrency"])
        self._db = f"{cfg['data_dir']}/cache.db"
        self._ledger = f"{cfg['data_dir']}/ledger.jsonl"
        with sqlite3.connect(self._db) as con:
            con.execute(
                "create table if not exists c (k text primary key, v text)"
            )

    def _get(self, key: str) -> Any:
        with sqlite3.connect(self._db) as con:
            hit = con.execute("select v from c where k=?", (key,)).fetchone()
        if hit:
            self.stats["cached"] += 1
            return json.loads(hit[0])
        return None

    def _put(self, key: str, value: Any) -> None:
        with sqlite3.connect(self._db) as con:
            con.execute(
                "insert or replace into c values (?,?)",
                (key, json.dumps(value)),
            )
        self.stats["calls"] += 1

    async def complete(
        self, role: str, prompt: str, max_tokens: int, top: int = 0
    ) -> dict[str, Any]:
        """Greedy completion of a raw (already templated) prompt.

        Args:
          role: Name of the pipeline step, for the ledger.
          prompt: The full prompt text, chat template included.
          max_tokens: Generation limit.
          top: Number of top log-probabilities to return for the first
            generated token (0 for none).

        Returns:
          {"text": generated text, "first": [[token, logprob], ...],
          "finish": finish reason}. Cached entries written before the finish
          reason was stored have no "finish" key.
        """
        key = hashlib.sha256(
            json.dumps([self.model, prompt, max_tokens, top]).encode()
        ).hexdigest()
        hit = self._get(key)
        if hit is not None:
            return hit
        start = time.time()
        for attempt in range(5):
            try:
                async with self._sem:
                    resp = await self._api.completions.create(
                        model=self.model,
                        prompt=prompt,
                        max_tokens=max_tokens,
                        temperature=0,
                        logprobs=top or None,
                    )
                break
            except openai.OpenAIError:
                if attempt == 4:
                    raise
                await asyncio.sleep(2**attempt)
        choice = resp.choices[0]
        first = []
        if top and choice.logprobs and choice.logprobs.top_logprobs:
            first = sorted(
                choice.logprobs.top_logprobs[0].items(), key=lambda t: -t[1]
            )
        out = {
            "text": choice.text,
            "first": first,
            "finish": choice.finish_reason,  # "length" means it was cut off
        }
        self._put(key, out)
        with open(self._ledger, "a") as f:
            entry = {
                "role": role,
                "in": resp.usage.prompt_tokens,
                "out": resp.usage.completion_tokens,
                "wall_s": round(time.time() - start, 2),
            }
            f.write(json.dumps(entry) + "\n")
        return out
