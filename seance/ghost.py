"""The ghost: a local, open-weight language model doing the séance voice.

Everything here runs on the reader's own machine.  No API key, no account, no
network at inference time.  That is the point of the project, and it is also
the thing that makes the audit meaningful: we can run the *same* model on the
same machine with the brain signal swapped out, because swapping it out costs
nothing and leaks nothing.

Model: Gemma 3 4B IT (open weights), 4-bit NF4 via bitsandbytes.
Generation is greedy (deterministic) so that every number in the report can
be reproduced.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

SYSTEM_PROMPT = (
    "You are the spirit on the other side of a séance table, speaking through a "
    "brass medium. The medium reports a faint impression of what the sitter's mind "
    "is holding right now: one image, and a degree of clarity.\n\n"
    "Speak only as the spirit. Name the impression in your first sentence, then speak "
    "around it: two to four sentences, sensory, a little unsettling, as if you are "
    "remembering something that happened to somebody else and trying to describe it "
    "from the far side of the veil.\n\n"
    "Never explain yourself. Never mention electricity, electrodes, sensors, signals, "
    "brain waves, probability, numbers, data, models, or being a program. You do not "
    "know what you are. You are simply making the attempt."
)

CLARITY_BANDS = (
    (0.34, "the impression is barely there, more suggestion than sighting"),
    (0.60, "the impression is faint but present"),
    (1.01, "the impression is clear and pressing"),
)


def clarity_band(p: float) -> str:
    for hi, text in CLARITY_BANDS:
        if p < hi:
            return text
    return CLARITY_BANDS[-1][1]


@dataclass
class Sitting:
    """One séance: a trial, the impression the medium reports, the true answer."""

    key: str  # stable id
    subject: int
    image_idx: int
    rep: int
    true_concept: str
    sighted_concept: str
    clarity: float
    message: str = ""

    @property
    def band(self) -> str:
        return clarity_band(self.clarity)


def build_prompt(sitting: Sitting, with_impression: bool = True) -> str:
    if with_impression:
        return (
            f"The medium reports an impression.\n"
            f"Impression: {sitting.sighted_concept}\n"
            f"Clarity: {sitting.band}\n\n"
            f"Speak now, as the spirit, in two to four sentences."
        )
    return (
        f"The medium reports nothing tonight -- the veil is thick.\n\n"
        f"Speak now, as the spirit, in two to four sentences."
    )


class Ghost:
    """Local Gemma 3 wrapper with a séance persona."""

    def __init__(
        self,
        model_dir: str,
        max_new_tokens: int = 96,
        seed: int = 0,
        device: str = "cuda",
    ):
        import torch
        from transformers import AutoTokenizer

        self.torch = torch
        self.max_new_tokens = max_new_tokens
        torch.manual_seed(seed)

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = self._load(model_dir, device)
        self.model.eval()
        self.n_calls = 0

    # -- loading -----------------------------------------------------------
    def _load(self, model_dir: str, device: str):
        import torch
        from transformers import BitsAndBytesConfig

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        kwargs = dict(quantization_config=bnb, device_map=device, dtype=torch.bfloat16)
        errors = []
        for loader_name in ("AutoModelForImageTextToText", "AutoModelForCausalLM", "AutoModelForVision2Seq"):
            try:
                import transformers

                loader = getattr(transformers, loader_name)
                model = loader.from_pretrained(model_dir, **kwargs)
                print(f"[ghost] loaded with {loader_name}: "
                      f"{model.__class__.__name__}", flush=True)
                return model
            except Exception as exc:  # noqa: BLE001 - try the next loader
                errors.append(f"{loader_name}: {type(exc).__name__}: {exc}")
        raise RuntimeError("could not load model:\n" + "\n".join(errors))

    # -- generation --------------------------------------------------------
    def speak(self, prompt: str) -> str:
        """One greedy completion."""
        torch = self.torch
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        gen = out[0][inputs["input_ids"].shape[-1]:]
        self.n_calls += 1
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def speak_many(self, prompts: list[str], progress_every: int = 10) -> list[str]:
        out = []
        for i, p in enumerate(prompts, 1):
            out.append(self.speak(p))
            if progress_every and i % progress_every == 0:
                print(f"[ghost] {i}/{len(prompts)} messages", flush=True)
        return out
