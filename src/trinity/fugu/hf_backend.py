"""HF trainable backend for the Fugu Conductor policy.

This module is imported only on the GPU box. It implements the
``PolicyBackend`` protocol from :mod:`trinity.fugu.grpo` without depending on
TRL: rollouts are sampled with ``generate`` and the update recomputes token
log-probs for the emitted workflow under the current policy, weighted by the
group-normalized GRPO advantages.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trinity.fugu.conductor import Proposal, build_prompt
from trinity.types import Task

__all__ = ["HFBackendConfig", "HFPolicyBackend"]


@dataclass
class HFBackendConfig:
    model_name: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda:0"
    dtype: str = "bfloat16"
    lr: float = 1e-6
    max_new_tokens: int = 512
    max_prompt_tokens: int = 4096
    sample_temperature: float = 1.0
    greedy_temperature: float = 0.0
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    gradient_accumulation: int = 1
    proposal_prefix: str = "model_id = ["


class HFPolicyBackend:
    """Trainable local Conductor backed by ``transformers``.

    The backend knows the worker menu so ``update`` can reconstruct the exact
    prompt for every stored rollout from ``GroupResult.task`` +
    ``WorkflowRun.raw_proposal``. This keeps the generic GRPO loop free of torch.
    """

    def __init__(self, cfg: HFBackendConfig, worker_names: list[str]):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.worker_names = list(worker_names)
        self.torch = torch
        self.device = torch.device(cfg.device)
        self.dtype = _torch_dtype(torch, cfg.dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device)
        self.model.train()
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self._updates = 0

    async def propose(
        self,
        task: Task,
        worker_names: list[str],
        *,
        sample: bool = False,
        rng=None,
        client=None,
    ) -> Proposal:
        del rng, client
        if list(worker_names) != self.worker_names:
            self.worker_names = list(worker_names)

        torch = self.torch
        prompt = self._prompt_text(task)
        enc = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_prompt_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.cfg.max_new_tokens,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if sample:
            gen_kwargs.update(
                do_sample=True,
                temperature=max(1e-5, self.cfg.sample_temperature),
                top_p=0.95,
            )
        else:
            gen_kwargs.update(do_sample=False)

        self.model.eval()
        with torch.inference_mode():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **gen_kwargs,
            )
        self.model.train()

        prompt_len = int(input_ids.shape[1])
        gen_ids = out[0, prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        text = f"{self.cfg.proposal_prefix}{text}".strip()
        return Proposal(
            text=text,
            prompt_tokens=prompt_len,
            completion_tokens=int(gen_ids.numel()),
        )

    def update(self, groups: list) -> dict:
        """Apply one GRPO policy-gradient update.

        Loss per sample is ``advantage * token_nll``. Positive-advantage
        workflows are made more likely; negative-advantage workflows less likely.
        Zero-advantage samples are skipped because they carry no group signal.
        """
        torch = self.torch
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        n_samples = 0
        n_tokens = 0
        loss_sum = 0.0
        abs_adv_sum = 0.0
        accum = max(1, int(self.cfg.gradient_accumulation))

        for group in groups:
            for run, adv in zip(group.runs, group.advantages):
                if abs(float(adv)) < 1e-8 or not run.raw_proposal:
                    continue
                loss, toks = self._sample_nll(group.task, run.raw_proposal)
                weighted = loss * float(adv) / accum
                weighted.backward()
                loss_sum += float(weighted.detach().cpu()) * accum
                abs_adv_sum += abs(float(adv))
                n_samples += 1
                n_tokens += toks
                if n_samples % accum == 0:
                    if self.cfg.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)

        if n_samples and n_samples % accum:
            if self.cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        self._updates += 1
        return {
            "backend": "hf",
            "updates": self._updates,
            "samples": n_samples,
            "tokens": n_tokens,
            "mean_weighted_loss": loss_sum / max(1, n_samples),
            "mean_abs_advantage": abs_adv_sum / max(1, n_samples),
        }

    def save_pretrained(self, path: str) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def _sample_nll(self, task: Task, proposal: str):
        torch = self.torch
        prompt = self._prompt_text(task)
        full = self._assistant_text(task, proposal)
        prompt_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_prompt_tokens,
        )["input_ids"][0]
        enc = self.tokenizer(
            full,
            return_tensors="pt",
            truncation=True,
            max_length=self.cfg.max_prompt_tokens + self.cfg.max_new_tokens,
        )
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        labels = input_ids.clone()
        prompt_len = min(int(prompt_ids.numel()), int(labels.shape[1]))
        labels[:, :prompt_len] = -100
        valid = labels.ne(-100).sum().clamp_min(1)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        # HF returns mean CE over unmasked labels. Keep it length-normalized so a
        # long parseable workflow is not punished solely for having more tokens.
        return out.loss.to(torch.float32), int(valid.item())

    def _prompt_text(self, task: Task) -> str:
        messages = build_prompt(task, self.worker_names)
        return (
            _chat_text(self.tokenizer, messages, add_generation_prompt=True)
            + self.cfg.proposal_prefix
        )

    def _assistant_text(self, task: Task, proposal: str) -> str:
        if proposal.startswith(self.cfg.proposal_prefix):
            proposal = proposal[len(self.cfg.proposal_prefix):]
        return self._prompt_text(task) + proposal


def _chat_text(tokenizer, messages: list[dict], *, add_generation_prompt: bool) -> str:
    if getattr(tokenizer, "chat_template", None):
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        try:
            return tokenizer.apply_chat_template(
                messages,
                enable_thinking=False,
                **kwargs,
            )
        except TypeError:
            return tokenizer.apply_chat_template(messages, **kwargs)
    chunks = []
    for m in messages:
        role = m.get("role", "user").upper()
        chunks.append(f"{role}:\n{m.get('content', '')}")
    if add_generation_prompt:
        chunks.append("ASSISTANT:\n")
    return "\n\n".join(chunks)


def _torch_dtype(torch, name: str):
    key = (name or "").lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"unsupported dtype: {name}")
