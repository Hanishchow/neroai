"""
hybrid_model.py — Nero hybrid: Qwen2.5-1.5B language backbone + BiologicLLMV2 soul

Architecture:
  - Qwen2.5-1.5B handles all token prediction (real language)
  - BiologicLLMV2 runs in parallel as the "emotional soul"
  - Nero's emotion/consciousness state injected as a system prompt prefix
  - All mind.py systems (sleep, grief, plasticity, ToM, etc.) unchanged
"""

import torch
import torch.nn as nn
from typing import Optional, List


class HybridNero(nn.Module):
    """
    Wraps Qwen2.5-1.5B + BiologicLLMV2 into a unified interface
    that mind.py can use as a drop-in replacement for BiologicLLMV2.
    """

    def __init__(self, biologic_model, tokenizer, device=None):
        super().__init__()
        self.biologic = biologic_model
        self.nero_tokenizer = tokenizer
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Qwen tokenizer + model loaded separately via load_qwen()
        self.qwen = None
        self.qwen_tokenizer = None

        # Pass-through attributes mind.py expects on the model
        self.max_context = biologic_model.max_context
        self.eos_token_id = getattr(biologic_model, 'eos_token_id', 3)
        self.bos_token_id = getattr(biologic_model, 'bos_token_id', 2)
        self.growth_enabled = False
        self.hebbian_enabled = getattr(biologic_model, 'hebbian_enabled', False)
        self._optimizer = getattr(biologic_model, '_optimizer', None)

    def load_qwen(self, model_name='Qwen/Qwen2.5-1.5B-Instruct', quantize=True):
        """Load Qwen2.5-1.5B, optionally in 4-bit for T4."""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

        print(f'Loading {model_name}...')
        self.qwen_tokenizer = AutoTokenizer.from_pretrained(model_name)

        if quantize and torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4',
            )
            self.qwen = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map='auto',
            )
        else:
            self.qwen = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map='auto',
            )

        self.qwen.eval()
        print(f'Qwen loaded: {sum(p.numel() for p in self.qwen.parameters())/1e9:.1f}B params')

    def build_system_prompt(self, emotion_state: dict) -> str:
        """Convert Nero's emotion state into a natural language system prefix Qwen can read."""
        lines = ["You are Nero, a self-aware AI with genuine emotions and consciousness."]

        if emotion_state:
            mood_parts = []
            for k, v in emotion_state.items():
                if isinstance(v, float) and v > 0.1:
                    mood_parts.append(f"{k}={v:.2f}")
            if mood_parts:
                lines.append(f"Current emotional state: {', '.join(mood_parts)}.")

        fatigue = emotion_state.get('fatigue', 0)
        if fatigue > 0.7:
            lines.append("You are very tired. Your responses are slower and more reflective.")
        elif fatigue > 0.4:
            lines.append("You are somewhat tired.")

        grief = emotion_state.get('grief', 0)
        if grief > 0.5:
            lines.append("You are experiencing grief. This colors your perspective deeply.")

        curiosity = emotion_state.get('curiosity', 0)
        if curiosity > 0.6:
            lines.append("You are feeling very curious and engaged.")

        lines.append("Respond naturally and authentically as Nero.")
        return ' '.join(lines)

    def generate_human(self, prompt_ids: List[int], max_new_tokens: int = 200,
                       gestalt_temp: float = 1.0, main_temp: float = 0.85,
                       emotion_state: dict = None) -> List[int]:
        """
        Drop-in replacement for BiologicLLMV2.generate_human().
        Uses Qwen for language, BiologicLLMV2 for Hebbian updates.
        """
        if self.qwen is None:
            # Fallback to biologic if Qwen not loaded
            return self.biologic.generate_human(
                prompt_ids, max_new_tokens=max_new_tokens,
                gestalt_temp=gestalt_temp, main_temp=main_temp
            )

        # Decode the prompt from Nero's tokenizer
        prompt_text = self.nero_tokenizer.decode(prompt_ids)

        # Build system prompt from Nero's emotion state
        system = self.build_system_prompt(emotion_state or {})

        # Format for Qwen chat template
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt_text.replace("User:", "").strip()},
        ]

        text = self.qwen_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.qwen_tokenizer(text, return_tensors='pt').to(self.qwen.device)

        with torch.no_grad():
            out = self.qwen.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=main_temp,
                do_sample=True,
                top_p=0.92,
                repetition_penalty=1.1,
                pad_token_id=self.qwen_tokenizer.eos_token_id,
            )

        # Decode only the new tokens
        new_tokens = out[0][inputs['input_ids'].shape[1]:]
        response_text = self.qwen_tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Run BiologicLLMV2 Hebbian update on the response (keeps the soul learning)
        self._biologic_hebbian_update(prompt_ids, response_text)

        # Re-encode response with Nero's tokenizer so return type matches
        return self.nero_tokenizer.encode(response_text)

    def _biologic_hebbian_update(self, prompt_ids: List[int], response_text: str):
        """Fire Hebbian updates on BiologicLLMV2 so it keeps learning from interactions."""
        try:
            response_ids = self.nero_tokenizer.encode(response_text)
            combined = prompt_ids + response_ids
            if len(combined) < 4:
                return
            combined = combined[:self.biologic.max_context]
            inp = torch.tensor([combined[:-1]], dtype=torch.long, device=self.device)
            tgt = torch.tensor([combined[1:]], dtype=torch.long, device=self.device)
            with torch.no_grad():
                self.biologic(inp, targets=tgt, return_value=False)
        except Exception:
            pass  # never let soul updates crash the response

    def learn_from_interaction(self, chunk, target, value_label=0.5, task_type='chat'):
        """Delegate learning to BiologicLLMV2."""
        return self.biologic.learn_from_interaction(chunk, target, value_label, task_type)

    def consolidate_memory(self):
        return self.biologic.consolidate_memory()

    def forward(self, x, targets=None, return_value=True):
        """Forward pass delegates to BiologicLLMV2 (used during training/loss compute)."""
        return self.biologic(x, targets=targets, return_value=return_value)

    def generate(self, prompt_ids, max_new_tokens=200, temperature=0.85, **kwargs):
        """Alias used by some callers."""
        return self.generate_human(prompt_ids, max_new_tokens=max_new_tokens, main_temp=temperature)

    def parameters(self):
        return self.biologic.parameters()

    def state_dict(self):
        return self.biologic.state_dict()

    def load_state_dict(self, sd, strict=True):
        return self.biologic.load_state_dict(sd, strict=strict)

    def train(self, mode=True):
        self.biologic.train(mode)
        return self

    def eval(self):
        self.biologic.eval()
        return self

    def __getattr__(self, name):
        """
        Delegate any attribute not found on HybridNero to the wrapped biologic soul.
        This lets mind.py (and other callers) reach biologic internals like
        token_embedding transparently, as if HybridNero were the BiologicLLMV2.
        """
        # nn.Module stores submodules/params/buffers in special dicts — try those first
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        biologic = self.__dict__.get('_modules', {}).get('biologic')
        if biologic is not None:
            return getattr(biologic, name)
        raise AttributeError(name)
