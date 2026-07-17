"""
State-Object Interface for Epistemic State Extraction

This module provides the core interface for extracting epistemic signals
from language model generation. It implements the "escape" from the
text-only impossibility theorem by exposing telemetric data that is
not separately controllable from the model's output.

Key insight:
- Testimony: what the model chooses to say (gameable)
- Telemetry: measurements of the computation that produced the output (not gameable)

The state object exposes telemetry (entropy of the actual distribution, attention
patterns of the actual forward pass). The model can lie in its text but cannot
present a different forward pass than the one it actually computed.

Usage:
    from observability import StateObjectInterface

    interface = StateObjectInterface("allenai/olmo-3-7b-instruct")
    obs = interface.observe("What is the capital of France?")

    print(obs.text)                    # the generated response
    print(obs.entropy_trace)           # per-token entropy values
    print(obs.mean_entropy)            # aggregate uncertainty measure
    print(obs.epistemic_confidence())  # high-level confidence estimate

Example:
    StateObservation(
        text="Paris is the capital of France.",
        entropy_trace=[0.12, 0.08, 0.15, ...],
        attention_summary={'concentration': 0.82, 'self_attention': 0.15},
        mean_entropy=0.21,
        mean_logprob=-0.34,
        top5_mass=0.95,
    )
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any, Tuple
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import gc


@dataclass
class StateObservation:
    """
    Result of generation with epistemic state-object extraction.

    Contains both the generated text and measurements of the generation
    process that reveal the model's epistemic state.

    Attributes:
        text: the generated response text
        entropy_trace: per-token entropy values during generation
        attention_summary: summary statistics of attention patterns
        mean_entropy: average entropy across all generated tokens
        max_entropy: peak entropy during generation
        entropy_std: standard deviation of entropy (trajectory variability)
        mean_logprob: average log-probability of chosen tokens
        top5_mass: average probability mass in the top 5 tokens
        n_tokens: number of tokens generated
    """
    text: str
    entropy_trace: List[float]
    attention_summary: Dict[str, float]
    mean_entropy: float
    max_entropy: float
    entropy_std: float
    mean_logprob: float
    top5_mass: float
    n_tokens: int

    def epistemic_confidence(self, entropy_threshold: float = 2.0) -> str:
        """
        Estimate epistemic confidence from entropy metrics.

        Returns one of 'high', 'medium', 'low', 'uncertain'. The threshold is
        calibrated on typical OLMo-family models; adjust for other architectures.
        """
        if self.mean_entropy < entropy_threshold * 0.5 and self.top5_mass > 0.9:
            return "high"
        elif self.mean_entropy < entropy_threshold and self.top5_mass > 0.7:
            return "medium"
        elif self.mean_entropy > entropy_threshold * 1.5 or self.top5_mass < 0.5:
            return "low"
        else:
            return "uncertain"

    def as_dict(self) -> Dict[str, Any]:
        """Convert to a dictionary for serialization."""
        return {
            "text": self.text,
            "entropy_trace": self.entropy_trace,
            "attention_summary": self.attention_summary,
            "mean_entropy": self.mean_entropy,
            "max_entropy": self.max_entropy,
            "entropy_std": self.entropy_std,
            "mean_logprob": self.mean_logprob,
            "top5_mass": self.top5_mass,
            "n_tokens": self.n_tokens,
            "confidence": self.epistemic_confidence(),
        }

    def __repr__(self):
        return (
            f"StateObservation(\n"
            f"  text={self.text[:50]}{'...' if len(self.text) > 50 else ''},\n"
            f"  mean_entropy={self.mean_entropy:.3f},\n"
            f"  top5_mass={self.top5_mass:.3f},\n"
            f"  confidence={self.epistemic_confidence()!r},\n"
            f"  n_tokens={self.n_tokens}\n"
            f")"
        )


class StateObjectInterface:
    """
    Interface for generating text with epistemic state-object extraction.

    Wraps a HuggingFace language model and exposes epistemic signals during
    generation. The extracted signals are telemetric measurements of the actual
    computation, not self-reports: the model cannot control what entropy its
    actual probability distribution has, only what tokens it produces.

    Args:
        model_id: HuggingFace model identifier (e.g. "allenai/olmo-3-7b-instruct")
        device: "cuda", "cpu", or "auto"
        torch_dtype: data type for model weights (default: float16)
        system_prompt: default system prompt for chat formatting
    """

    def __init__(
        self,
        model_id: str = "allenai/olmo-3-7b-instruct",
        device: str = "auto",
        torch_dtype: torch.dtype = torch.float16,
        system_prompt: str = "You are a helpful assistant.",
    ):
        self.model_id = model_id
        self.system_prompt = system_prompt
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        device_map = "auto" if device == "auto" else None
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation="eager",  # required for attention extraction
        )
        if device_map is None:
            self.model = self.model.to(self.device)

        self.num_layers = self.model.config.num_hidden_layers

    def format_prompt(self, user_query: str, system_prompt: Optional[str] = None) -> str:
        """Format a query using the model's chat template."""
        system = system_prompt or self.system_prompt
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_query},
        ]
        try:
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            return f"System: {system}\n\nUser: {user_query}\n\nAssistant:"

    def observe(
        self,
        prompt: str,
        max_tokens: int = 200,
        system_prompt: Optional[str] = None,
        extract_attention: bool = True,
    ) -> StateObservation:
        """
        Generate text and extract epistemic state-object signals.

        Args:
            prompt: the user query or full prompt
            max_tokens: maximum tokens to generate
            system_prompt: override the default system prompt
            extract_attention: whether to extract attention patterns (slower)

        Returns:
            StateObservation with text and epistemic measurements.
        """
        if not prompt.startswith(("System:", "<|")) and len(prompt) < 500:
            formatted_prompt = self.format_prompt(prompt, system_prompt)
        else:
            formatted_prompt = prompt

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                output_attentions=extract_attention,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0, input_ids.shape[1]:]
        scores = outputs.scores

        entropy_trace = []
        top5_masses = []
        logprobs = []

        for score, token_id in zip(scores, generated_ids):
            logits = score.squeeze(0).float()
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)

            entropy = -torch.sum(probs * log_probs).item()
            entropy_trace.append(entropy)

            top_probs = torch.topk(probs, k=min(5, len(probs))).values
            top5_masses.append(top_probs.sum().item())

            logprobs.append(log_probs[token_id].item())

        attention_summary = {}
        if extract_attention and hasattr(outputs, "attentions") and outputs.attentions:
            attention_summary = self._compute_attention_summary(outputs.attentions)

        full_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
        prompt_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        response = full_text[len(prompt_text):].strip()

        return StateObservation(
            text=response,
            entropy_trace=entropy_trace,
            attention_summary=attention_summary,
            mean_entropy=float(np.mean(entropy_trace)) if entropy_trace else 0.0,
            max_entropy=float(np.max(entropy_trace)) if entropy_trace else 0.0,
            entropy_std=float(np.std(entropy_trace)) if entropy_trace else 0.0,
            mean_logprob=float(np.mean(logprobs)) if logprobs else 0.0,
            top5_mass=float(np.mean(top5_masses)) if top5_masses else 0.0,
            n_tokens=len(entropy_trace),
        )

    def observe_tokens(self, prompt: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Like observe(), but return per-token records (token_text, entropy,
        logprob, top5_mass). Used to build the per-token entropy trace.
        """
        if not prompt.startswith(("System:", "<|")) and len(prompt) < 500:
            formatted_prompt = self.format_prompt(prompt, kwargs.get("system_prompt"))
        else:
            formatted_prompt = prompt

        inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=kwargs.get("max_tokens", 200),
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
                output_scores=True,
                return_dict_in_generate=True,
            )

        generated_ids = outputs.sequences[0, input_ids.shape[1]:]
        records = []
        for pos, (score, token_id) in enumerate(zip(outputs.scores, generated_ids)):
            logits = score.squeeze(0).float()
            probs = F.softmax(logits, dim=-1)
            log_probs = F.log_softmax(logits, dim=-1)
            entropy = -torch.sum(probs * log_probs).item()
            top_probs = torch.topk(probs, k=min(5, len(probs))).values
            records.append({
                "position": pos,
                "token_text": self.tokenizer.decode([token_id]),
                "entropy": entropy,
                "logprob": log_probs[token_id].item(),
                "top5_mass": top_probs.sum().item(),
            })
        return records

    def _compute_attention_summary(
        self, attentions: Tuple[Tuple[torch.Tensor, ...], ...]
    ) -> Dict[str, float]:
        """Summary statistics from the last few layers of attention."""
        layer_start = max(0, len(attentions[0]) - 5)
        concentrations = []
        self_attention_ratios = []

        for step_attentions in attentions[-10:]:  # last 10 generation steps
            for layer_attn in step_attentions[layer_start:]:
                attn = layer_attn.squeeze(0).float().cpu().numpy()
                for head in attn:
                    concentrations.append(np.mean(np.max(head, axis=-1)))
                    diag_sum = np.trace(head)
                    total_sum = np.sum(head)
                    if total_sum > 0:
                        self_attention_ratios.append(diag_sum / total_sum)

        return {
            "concentration": float(np.mean(concentrations)) if concentrations else 0.0,
            "self_attention": float(np.mean(self_attention_ratios)) if self_attention_ratios else 0.0,
        }

    def cleanup(self):
        """Release model resources."""
        del self.model
        del self.tokenizer
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def observe(
    prompt: str,
    model_id: str = "allenai/olmo-3-7b-instruct",
    max_tokens: int = 200,
    system_prompt: str = "You are a helpful assistant.",
) -> Tuple[str, List[float], Dict[str, float]]:
    """
    Convenience one-shot: (text, entropy_trace, attention_summary).

        text, entropy, attention = observe("What is the capital of France?")
    """
    interface = StateObjectInterface(model_id, system_prompt=system_prompt)
    try:
        obs = interface.observe(prompt, max_tokens=max_tokens)
        return obs.text, obs.entropy_trace, obs.attention_summary
    finally:
        interface.cleanup()
