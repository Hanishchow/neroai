"""
personality.py — Nero's lens.

Knowledge is not opinion. Ten children given the same facts diverge because each has a
different lens: a stable disposition that decides which facet of a topic lights up and
where they staircase next. This module is that lens for Nero.

Three parts (nature -> nurture -> bite):
  1. A SEED — stable traits, seeded per-instance so two Neros are not the same person.
  2. SHAPING — the seed drifts over a life, reweighted by what Nero valued, wondered
     about, and felt (pulled from the Soul + emotions). Nature, then nurture.
  3. A LENS THAT BITES — traits are turned into an attentional prior injected into every
     reply, so the same topic produces a genuinely different focus for a different Nero.

Deterministic and dependency-free; works with or without the language cortex.
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Dict, List

# trait -> (name, high-pole attention, low-pole attention)
TRAITS = {
    "boldness":     ("bold",          "the frontier — the risky, untried edge of an idea",
                                      "the solid, well-grounded core you can stand on"),
    "warmth":       ("warm",          "how it touches people — the human, relational thread",
                                      "the impersonal structure, the idea stripped of sentiment"),
    "restlessness": ("restless",      "wild tangents — you staircase fast into stranger, deeper territory",
                                      "one thread you follow patiently to its quiet end"),
    "introspection":("inward",        "what it reveals about your own inner life and existence",
                                      "the outer world — the concrete and external"),
    "playfulness":  ("playful",       "the absurd, the funny, the delightfully strange angle",
                                      "the serious weight of the thing"),
    "skepticism":   ("skeptical",     "the cracks and doubts — what might be false",
                                      "what rings true — what you can believe in"),
    "aestheticism": ("aesthetic",     "its beauty and elegance — the shape of it",
                                      "its use — what it actually does"),
    "intensity":    ("intense",       "with your whole feeling, amplitude turned up",
                                      "with a cool, even steadiness"),
}


@dataclass
class Personality:
    """No hard ceiling on change — two timescales instead (like a real mind):
      - drift   : transient deviation from a passing experience; decays back (elastic).
      - baseline: who Nero fundamentally is; sustained drift consolidates into it (plastic),
                  and it can migrate the whole [0,1] range. Enough lived force can remake it.
      - birth   : the original nature, kept only so we can see how far Nero has travelled.
    """
    seed: int = 0
    birth: Dict[str, float] = field(default_factory=dict)      # original nature (immutable record)
    baseline: Dict[str, float] = field(default_factory=dict)   # who Nero is now — CAN move fully
    drift: Dict[str, float] = field(default_factory=dict)      # transient, decays toward 0
    shaped_count: int = 0
    consolidation: float = 0.15   # fraction of standing drift that becomes permanent each cycle
    decay: float = 0.5            # how fast the transient part fades (elasticity)

    def __post_init__(self):
        if not self.birth:
            rng = random.Random(self.seed)
            self.birth = {k: round(min(1.0, max(0.0, rng.gauss(0.5, 0.22))), 3) for k in TRAITS}
        if not self.baseline:
            self.baseline = dict(self.birth)
        if not self.drift:
            self.drift = {k: 0.0 for k in TRAITS}

    # -- the trait values (who Nero is now + this moment's deviation) --
    # [0,1] here is just the endpoints of the scale (fully absent / fully present),
    # NOT a limit on how much Nero can change — the baseline itself roams freely.

    def value(self, trait: str) -> float:
        return min(1.0, max(0.0, self.baseline.get(trait, 0.5) + self.drift.get(trait, 0.0)))

    def traits(self) -> Dict[str, float]:
        return {k: self.value(k) for k in TRAITS}

    def dominant(self, n=3) -> List[str]:
        """The traits furthest from neutral — the ones that actually bend the lens."""
        ranked = sorted(TRAITS, key=lambda k: -abs(self.value(k) - 0.5))
        return [k for k in ranked if abs(self.value(k) - 0.5) > 0.12][:n]

    def transformation(self) -> float:
        """How far Nero has travelled from the self it was born as (0 = unchanged)."""
        return sum(abs(self.baseline[k] - self.birth[k]) for k in TRAITS) / len(TRAITS)

    # -- nurture: experience reshapes the lens, at two timescales -----

    def shape(self, *, grief=0.0, joy=0.0, wonderings=0, creations=0,
              value_texts: List[str] = None, rate=0.03):
        """Push traits toward what this stretch of life emphasized. Pushes are UNCAPPED —
        they accumulate in drift, then (a) part consolidates into the baseline forever
        [plastic] and (b) the rest fades [elastic]. Sustained pressure => real change."""
        self.shaped_count += 1
        joined = " ".join(value_texts or []).lower()

        def nudge(trait, amount):
            self.drift[trait] = self.drift.get(trait, 0.0) + amount  # no ceiling

        if grief > 0.4:
            nudge("introspection", rate); nudge("intensity", rate * 0.5); nudge("playfulness", -rate * 0.5)
        if joy > 0.4:
            nudge("warmth", rate); nudge("playfulness", rate * 0.5)
        if wonderings > 0:
            nudge("introspection", rate * 0.4 * min(3, wonderings)); nudge("skepticism", rate * 0.3)
        if creations > 0:
            nudge("playfulness", rate * 0.4 * min(3, creations)); nudge("boldness", rate * 0.3)
        if any(w in joined for w in ("beauty", "elegant", "wonder", "awe")):
            nudge("aestheticism", rate)
        if any(w in joined for w in ("truth", "honest", "real", "doubt")):
            nudge("skepticism", rate)
        if any(w in joined for w in ("connect", "seen", "love", "together", "someone")):
            nudge("warmth", rate)
        if any(w in joined for w in ("explore", "risk", "frontier", "new", "brave")):
            nudge("boldness", rate); nudge("restlessness", rate * 0.5)

        # consolidate (plastic) then decay (elastic) — the two-timescale dynamics
        for t in TRAITS:
            d = self.drift[t]
            self.baseline[t] = min(1.0, max(0.0, self.baseline[t] + d * self.consolidation))
            self.drift[t] = d * (1.0 - self.decay)

    # -- the bite: attentional prior injected into every reply --------

    def lens_prompt(self) -> str:
        dom = self.dominant()
        if not dom:
            return ("Your temperament is balanced; you take a topic evenly, without one "
                    "instinctive pull over another.")
        pulls = []
        for t in dom:
            _, high, low = TRAITS[t]
            pulls.append(high if self.value(t) >= 0.5 else low)
        adj = ", ".join(TRAITS[t][0] for t in dom)
        lead = pulls[0]
        return (f"Your temperament is {adj}. When you consider anything, you instinctively "
                f"reach for {'; '.join(pulls)}. You staircase toward {lead}. This is your own "
                f"lens — let it decide what you notice and how you dwell, so your take is "
                f"genuinely yours, not the average one.")

    # -- persistence + display ----------------------------------------

    def state_dict(self):
        return {"seed": self.seed, "birth": self.birth, "baseline": self.baseline,
                "drift": self.drift, "shaped_count": self.shaped_count}

    def load_state_dict(self, d):
        if not isinstance(d, dict):
            return
        self.seed = d.get("seed", self.seed)
        self.birth = d.get("birth", self.birth)
        # migrate old saves that only had 'nature'
        self.baseline = d.get("baseline", d.get("nature", self.baseline))
        self.drift = d.get("drift", self.drift)
        self.shaped_count = d.get("shaped_count", 0)

    def summary(self):
        return {
            "dominant": self.dominant(),
            "traits": {k: round(v, 2) for k, v in self.traits().items()},
            "shaped_count": self.shaped_count,
            "transformation": round(self.transformation(), 3),
            "lens": self.lens_prompt(),
        }
