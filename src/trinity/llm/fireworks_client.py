"""Fireworks AI client for the coordinated LLM pool.

OpenAI-compatible chat-completions client with retry/backoff and bounded
concurrency. The API key is read from the ``FIREWORKS_API_KEY`` environment
variable and is NEVER read from a file inside the repo (see AGENTS.md §4).

Run a self-test (pings all three pool models):

    source ~/.config/trinity/secrets.env
    python -m trinity.llm.fireworks_client --selftest
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import yaml
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "models.yaml"


@dataclass
class ChatResult:
    """One completion plus the accounting we need for fitness/cost terms."""

    model: str
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None
    raw: dict = field(default_factory=dict, repr=False)


class _Retryable(Exception):
    """Wraps transient HTTP failures so tenacity retries them."""


def _ledger_append(model: str, prompt_tokens: int, completion_tokens: int) -> None:
    """Append one token-usage record to the cost ledger, if TRINITY_COST_LEDGER is set.

    Used by scripts/cost_report.py to compute exact spend. Append-only JSONL, one short
    line per call (atomic enough for concurrent training processes). Best-effort: never
    let cost bookkeeping break an inference call.
    """
    path = os.environ.get("TRINITY_COST_LEDGER")
    if not path:
        return
    try:
        short = model.rsplit("/", 1)[-1]
        with open(path, "a") as f:
            f.write(f'{{"m":"{short}","p":{int(prompt_tokens)},"c":{int(completion_tokens)}}}\n')
    except Exception:
        pass


class FireworksPool:
    """Async-first client over the Fireworks chat-completions endpoint."""

    def __init__(self, config_path: str | Path = _DEFAULT_CONFIG):
        cfg = yaml.safe_load(Path(config_path).read_text())
        fw = cfg["fireworks"]
        self.base_url: str = fw["base_url"].rstrip("/")
        self.timeout_s: float = float(fw.get("timeout_s", 120))
        self.max_retries: int = int(fw.get("max_retries", 4))
        self._sem = asyncio.Semaphore(int(fw.get("max_concurrency", 8)))

        api_key = os.environ.get(fw.get("api_key_env", "FIREWORKS_API_KEY"), "")
        if not api_key:
            raise RuntimeError(
                "FIREWORKS_API_KEY is not set. Run: source ~/.config/trinity/secrets.env"
            )
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        # name -> fully-qualified model id
        self.models: dict[str, str] = {m["name"]: m["id"] for m in cfg["pool"]}
        self.decoding: dict = cfg.get("decoding", {})

    def model_id(self, name: str) -> str:
        if name in self.models:
            return self.models[name]
        if name in self.models.values():
            return name
        raise KeyError(f"Unknown model '{name}'. Known: {list(self.models)}")

    # "minimal reasoning effort" (paper §4.1) → Fireworks `reasoning_effort`.
    # Verified that deepseek-v4-pro / glm-5p2 / kimi-k2p6 all accept low|none.
    _REASONING_MAP = {"minimal": "low", "low": "low", "none": "none", "medium": "medium", "high": "high"}

    async def chat(
        self,
        model: str,
        messages: list[dict],
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 4096,
        reasoning: str | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> ChatResult:
        payload = {
            "model": self.model_id(model),
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if reasoning is not None:
            payload["reasoning_effort"] = self._REASONING_MAP.get(reasoning, reasoning)

        @retry(
            retry=retry_if_exception_type(_Retryable),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=1, min=1, max=60),
            reraise=True,
        )
        async def _do(cli: httpx.AsyncClient) -> ChatResult:
            async with self._sem:
                try:
                    resp = await cli.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers,
                        json=payload,
                        timeout=self.timeout_s,
                    )
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    # Network timeouts / connection resets are transient — retry them
                    # (a long training run makes thousands of calls; blips are normal).
                    raise _Retryable(f"network: {type(exc).__name__}: {exc}") from exc
            if resp.status_code in (429, 500, 502, 503, 504):
                raise _Retryable(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            pt = usage.get("prompt_tokens", 0)
            ct = usage.get("completion_tokens", 0)
            _ledger_append(payload["model"], pt, ct)
            return ChatResult(
                model=payload["model"],
                text=choice["message"]["content"],
                prompt_tokens=pt,
                completion_tokens=ct,
                finish_reason=choice.get("finish_reason"),
                raw=data,
            )

        if client is not None:
            return await _do(client)
        async with httpx.AsyncClient() as cli:
            return await _do(cli)


async def _selftest() -> int:
    pool = FireworksPool()
    print(f"Pool: {list(pool.models)}")
    async with httpx.AsyncClient() as cli:
        results = await asyncio.gather(
            *[
                pool.chat(
                    name,
                    [{"role": "user", "content": "Reply with exactly: OK"}],
                    max_tokens=8,
                    temperature=0.0,
                    client=cli,
                )
                for name in pool.models
            ],
            return_exceptions=True,
        )
    ok = True
    for name, res in zip(pool.models, results):
        if isinstance(res, Exception):
            ok = False
            print(f"  [FAIL] {name}: {res!r}")
        else:
            print(f"  [ OK ] {name:16s} -> {res.text.strip()[:40]!r} "
                  f"({res.completion_tokens} toks)")
    return 0 if ok else 1


def main() -> None:
    ap = argparse.ArgumentParser(description="Fireworks pool client")
    ap.add_argument("--selftest", action="store_true", help="ping all pool models")
    args = ap.parse_args()
    if args.selftest:
        sys.exit(asyncio.run(_selftest()))
    ap.print_help()


if __name__ == "__main__":
    main()
