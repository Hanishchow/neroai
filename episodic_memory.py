"""
EPISODIC MEMORY — Memory traces anchored to concepts with pattern completion.
Stores compressed experiences and reconstructs them fresh each time,
like how humans recall memories from a few sensory cues.

Dual-store architecture:
  hippocampal — fast-learning, detailed, limited capacity (~150)
  cortical    — slow-learning, schematized, large capacity (~10000)

Reconsolidation: recall opens a plasticity window where memories become mutable.
Forgetting pressure: active mechanism displaces weak/unused traces.
"""

import time
import json
import re
import math
import random
from collections import defaultdict
from datetime import datetime

from emotions import MoodProfile


class MemoryTrace:
    """A single episodic memory stored as an anchor + compressed trace."""

    _id_counter = 0

    def __init__(self, anchor_concepts, context_snippet, valence=0.0,
                 source="user", timestamp=None, trace_id=None,
                 emotions=None, mortality_at_formation=0.0):
        MemoryTrace._id_counter += 1
        self.trace_id = trace_id or f"{time.time()}_{MemoryTrace._id_counter}"
        self.anchor_concepts = list(set(c.lower().strip() for c in anchor_concepts if c))
        self.context_snippet = context_snippet[:200]
        self.valence = max(-1.0, min(1.0, valence))
        self.emotions = emotions or MoodProfile()
        self.mortality_at_formation = mortality_at_formation
        self.source = source
        self.timestamp = timestamp or time.time()
        self.activation = 1.0
        self.reconstruction_count = 0
        self.last_recalled = 0

        # Reconsolidation window
        self.labile = False
        self.consolidated_at = 0.0

        # Store type
        self.in_hippocampus = True
        self.in_cortical = False

    def recall(self):
        """Open reconsolidation window — memory becomes mutable."""
        self.reconstruction_count += 1
        self.activation = min(10.0, self.activation + 0.5)
        self.last_recalled = time.time()
        self.labile = True
        self.consolidated_at = time.time() + 300.0  # 5 min window

    def consolidate(self):
        """Close reconsolidation window."""
        self.labile = False
        self.consolidated_at = 0.0

    def decay(self, rate=0.01):
        self.activation = max(0.1, self.activation * (1 - rate))

    def strength(self):
        recency = max(0, 1.0 - (time.time() - self.last_recalled) / 86400)
        return self.activation * (0.3 + 0.7 * recency) * (1 + 0.2 * self.reconstruction_count)

    def to_dict(self):
        return {
            'trace_id': self.trace_id,
            'anchor_concepts': self.anchor_concepts,
            'context_snippet': self.context_snippet,
            'valence': self.valence,
            'emotions': self.emotions.to_dict(),
            'mortality_at_formation': self.mortality_at_formation,
            'source': self.source,
            'timestamp': self.timestamp,
            'activation': self.activation,
            'reconstruction_count': self.reconstruction_count,
            'last_recalled': self.last_recalled,
            'in_hippocampus': self.in_hippocampus,
            'in_cortical': self.in_cortical
        }

    @staticmethod
    def from_dict(d):
        emotions = MoodProfile.from_dict(d.get('emotions', {}))
        t = MemoryTrace(
            anchor_concepts=d.get('anchor_concepts', []),
            context_snippet=d.get('context_snippet', ''),
            valence=d.get('valence', 0.0),
            source=d.get('source', 'unknown'),
            timestamp=d.get('timestamp', time.time()),
            trace_id=d.get('trace_id'),
            emotions=emotions,
            mortality_at_formation=d.get('mortality_at_formation', 0.0)
        )
        t.activation = d.get('activation', 1.0)
        t.reconstruction_count = d.get('reconstruction_count', 0)
        t.last_recalled = d.get('last_recalled', 0)
        t.in_hippocampus = d.get('in_hippocampus', True)
        t.in_cortical = d.get('in_cortical', False)
        return t


class EpisodicMemory:
    """
    Dual-store episodic memory (hippocampal ↔ cortical).
    """

    def __init__(self, filepath="episodic_memory.json",
                 max_hippocampal=150, max_cortical=10000,
                 emotion_system=None):
        self.filepath = filepath
        self.max_hippocampal = max_hippocampal
        self.max_cortical = max_cortical
        self.traces = {}         # master store (all traces)
        self.hippocampal_ids = set()
        self.cortical_ids = set()
        self._concept_index = defaultdict(set)
        self.emotion_system = emotion_system
        self.load()

    def store(self, text, concepts=None, valence=0.0, source="user",
              emotion_profile=None, mortality_anxiety=0.0):
        """Store a new memory. Goes to hippocampal store first."""
        if not text or len(text) < 10:
            return None

        anchor = concepts or self._extract_anchor_concepts(text)
        if not anchor:
            return None

        snippet = text[:200].strip()

        # Check for reconsolidation merge — if any trace is labile and overlapping, update it
        merged = self._try_merge_into_labile(anchor, snippet, valence, source, mortality_anxiety)
        if merged:
            return merged

        trace = MemoryTrace(
            anchor_concepts=anchor,
            context_snippet=snippet,
            valence=valence,
            source=source,
            emotions=emotion_profile or MoodProfile(),
            mortality_at_formation=mortality_anxiety
        )

        self.traces[trace.trace_id] = trace
        self.hippocampal_ids.add(trace.trace_id)
        for concept in anchor:
            self._concept_index[concept.lower()].add(trace.trace_id)

        # Enforce hippocampal capacity
        if len(self.hippocampal_ids) > self.max_hippocampal:
            self._displace_from_hippocampus()

        return trace

    # ================================================================
    # DUAL-STORE OPERATIONS
    # ================================================================

    def consolidate_to_cortex(self, trace_ids, emotion_system=None):
        """
        Transfer traces from hippocampal to cortical store.
        Called during NREM sleep.
        """
        for tid in trace_ids:
            if tid in self.traces and tid in self.hippocampal_ids:
                self.hippocampal_ids.discard(tid)
                self.cortical_ids.add(tid)
                self.traces[tid].in_hippocampus = False
                self.traces[tid].in_cortical = True
                self.traces[tid].context_snippet = self._schematize(self.traces[tid].context_snippet)

                # REMotionalization — dampen extreme emotions
                if emotion_system:
                    processed = emotion_system.process_dream_emotionalization(
                        self.traces[tid].emotions
                    )
                    self.traces[tid].emotions = processed

        # Enforce cortical capacity
        if len(self.cortical_ids) > self.max_cortical:
            self._prune_cortical()

    def recent_hippocampal(self, n=5):
        """Get most recent hippocampal traces."""
        sorted_ids = sorted(self.hippocampal_ids, key=lambda tid: -self.traces[tid].timestamp)
        return [self.traces[tid] for tid in sorted_ids[:n] if tid in self.traces]

    # ================================================================
    # FORGETTING PRESSURE
    # ================================================================

    def compute_forgetting_pressure(self, trace):
        """
        Active forgetting metric.
        Higher = more likely to be displaced.
        age_factor × activation_factor × novelty × (1 - anxiety_influence)
        """
        if trace.trace_id in self.cortical_ids:
            return 0.0  # cortical memories don't forget (stable)

        age = (time.time() - trace.last_recalled)
        age_factor = math.log(1 + age / 3600.0) / 10.0  # logarithmic age (hours)
        activation_factor = 1.0 / max(trace.activation, 0.01)
        novelty = 1.0 / (1 + trace.reconstruction_count)

        pressure = age_factor * activation_factor * novelty * 0.5
        return min(1.0, pressure)

    def _displace_from_hippocampus(self):
        """Remove the trace with highest forgetting pressure from hippocampus."""
        if not self.hippocampal_ids:
            return

        scored = []
        for tid in self.hippocampal_ids:
            if tid in self.traces:
                fp = self.compute_forgetting_pressure(self.traces[tid])
                scored.append((fp, tid))

        if not scored:
            return
        scored.sort(key=lambda x: -x[0])
        _, displace_id = scored[0]
        self._forget_trace(displace_id)

    def _forget_trace(self, tid):
        """Actively forget a trace — remove from index and store."""
        if tid not in self.traces:
            return
        trace = self.traces[tid]
        for c in trace.anchor_concepts:
            self._concept_index[c.lower()].discard(tid)
        self.hippocampal_ids.discard(tid)
        self.cortical_ids.discard(tid)
        del self.traces[tid]

    def _prune_cortical(self):
        """Prune oldest cortical traces when over capacity."""
        sorted_ids = sorted(self.cortical_ids, key=lambda tid: -self.traces[tid].last_recalled)
        to_remove = sorted_ids[:len(self.cortical_ids) - self.max_cortical // 2]
        for tid in to_remove:
            self._forget_trace(tid)

    # ================================================================
    # RECONSTRUCTION WITH DISTORTION
    # ================================================================

    def reconstruct_with_distortion(self, cue, model, tokenizer, current_mood=None,
                                     temperature=0.85, max_new=200):
        """
        Reconstruct a memory. If the trace is labile, it distorts toward current mood.
        """
        result = self.reconstruct(cue, model, tokenizer, temperature, max_new)
        if not result:
            return None

        trace = None
        for tid in self.hippocampal_ids | self.cortical_ids:
            t = self.traces.get(tid)
            if t and any(c in cue.lower() for c in t.anchor_concepts):
                if t.trace_id == result.get('_trace_id'):
                    trace = t
                    break

        if trace and trace.labile and current_mood:
            # Memory blends toward current emotional state
            trace.emotions = trace.emotions.blend(current_mood, weight=0.1)
            result['distortion'] = True
            result['note'] = "This memory feels different now than when it happened."

        return result

    # ================================================================
    # EXISTING METHODS (adapted)
    # ================================================================

    def reconstruct(self, cue, model, tokenizer, temperature=0.85, max_new=200):
        traces = self.recall_similar(cue, top_k=3)
        if not traces:
            return None

        best = traces[0]
        best.recall()

        context = f"I remember learning about {', '.join(best.anchor_concepts[:3])}. "
        if best.context_snippet:
            context += f"It makes me think of {best.context_snippet[:100]}. "
        if best.emotions:
            dom_emotion, strength = best.emotions.dominant()
            context += f"This feels {dom_emotion} (strength: {strength:.1f}). "
        else:
            context += f"This feels {self._valence_desc(best.valence)}. "

        seed = f"Recalling from memory: {context}The full memory comes back to me now:"

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(prompt_ids, max_new_tokens=max_new, temperature=temperature)
        reconstruction = tokenizer.decode(generated_ids)

        extra = ""
        if len(traces) > 1:
            extra = f"\n...this also connects to {traces[1].anchor_concepts[0] if traces[1].anchor_concepts else 'something else familiar'}"

        return {
            'anchor': best.anchor_concepts,
            'valence': best.valence,
            'emotions': best.emotions.to_dict(),
            'reconstruction': reconstruction,
            'associated': extra,
            'strength': best.strength(),
            '_trace_id': best.trace_id,
        }

    def pattern_complete(self, partial_cues, model, tokenizer, temperature=0.9, max_new=150):
        matches = self.recall_similar(partial_cues, top_k=5)
        if not matches:
            return None

        cue_list = [c.lower().strip() for c in re.split(r'[,\s]+', partial_cues) if len(c.strip()) > 2]

        scored = []
        for trace in matches:
            overlap = len(set(cue_list) & set(trace.anchor_concepts))
            scored.append((overlap, trace.strength(), trace))

        scored.sort(key=lambda x: (-x[0], -x[1]))
        best = scored[0][2]
        best.recall()

        seed = f"Given {' and '.join(cue_list[:3])}, I recall: {best.context_snippet[:100]}"

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(prompt_ids, max_new_tokens=max_new, temperature=temperature)
        completed = tokenizer.decode(generated_ids)

        return {
            'cues': partial_cues,
            'matched_anchor': best.anchor_concepts,
            'completed': completed,
            'confidence': scored[0][0] / max(len(cue_list), 1)
        }

    def recall_similar(self, cue, top_k=5, mood_bias=None):
        """Recall with optional mood-congruent bias."""
        if not self.traces:
            return []

        cue_lower = cue.lower().strip()
        cue_words = set(re.split(r'[^a-zA-Z0-9]+', cue_lower))
        cue_words = {w for w in cue_words if len(w) > 2}

        scored = []
        for trace in self.traces.values():
            concept_match = sum(1 for c in trace.anchor_concepts if c in cue_lower or any(cw in c for cw in cue_words))
            word_overlap = len(set(w for w in cue_words if any(w in ac for ac in trace.anchor_concepts)))
            score = concept_match * 2 + word_overlap + trace.strength() * 0.5

            # Mood-congruent boost
            if mood_bias and hasattr(trace, 'emotions'):
                dom_emotion, _ = trace.emotions.dominant()
                if dom_emotion == mood_bias:
                    score *= 1.2

            if score > 0:
                scored.append((score, trace))

        scored.sort(key=lambda x: -x[0])
        return [t for _, t in scored[:top_k]]

    def recent_memories(self, n=10):
        sorted_traces = sorted(self.traces.values(), key=lambda t: -t.timestamp)
        return sorted_traces[:n]

    def highest_surprise(self, n=5):
        sorted_traces = sorted(
            self.traces.values(),
            key=lambda t: (abs(t.valence) * t.reconstruction_count if t.reconstruction_count > 0 else 0.1),
            reverse=True
        )
        return sorted_traces[:n]

    def least_connected(self, n=3):
        if not self.traces:
            return []
        sorted_traces = sorted(self.traces.values(), key=lambda t: len(t.anchor_concepts))
        return sorted_traces[:n]

    def stats(self):
        if not self.traces:
            return "Episodic memory is empty."

        avg_valence = sum(t.valence for t in self.traces.values()) / len(self.traces)
        avg_activation = sum(t.activation for t in self.traces.values()) / len(self.traces)
        total_recalled = sum(t.reconstruction_count for t in self.traces.values())

        return (
            f"Total traces: {len(self.traces)}\n"
            f"  Hippocampal: {len(self.hippocampal_ids)}\n"
            f"  Cortical: {len(self.cortical_ids)}\n"
            f"Avg valence: {avg_valence:.2f}\n"
            f"Avg activation: {avg_activation:.2f}\n"
            f"Total reconstructions: {total_recalled}\n"
            f"Most recalled: {max(self.traces.values(), key=lambda t: t.reconstruction_count).anchor_concepts[:3]}"
        )

    def reconstruct(self, cue, model, tokenizer, temperature=0.85, max_new=200):
        traces = self.recall_similar(cue, top_k=3)
        if not traces:
            return None

        best = traces[0]
        best.recall()

        context = f"I remember learning about {', '.join(best.anchor_concepts[:3])}. "
        if best.context_snippet:
            context += f"It makes me think of {best.context_snippet[:100]}. "
        context += f"This feels {self._valence_desc(best.valence)}. "

        seed = f"Recalling from memory: {context}The full memory comes back to me now:"

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(prompt_ids, max_new_tokens=max_new, temperature=temperature)
        reconstruction = tokenizer.decode(generated_ids)

        extra = ""
        if len(traces) > 1:
            extra = f"\n...this also connects to {traces[1].anchor_concepts[0] if traces[1].anchor_concepts else 'something else familiar'}"

        return {
            'anchor': best.anchor_concepts,
            'valence': best.valence,
            'reconstruction': reconstruction,
            'associated': extra,
            'strength': best.strength()
        }

    def pattern_complete(self, partial_cues, model, tokenizer, temperature=0.9, max_new=150):
        matches = self.recall_similar(partial_cues, top_k=5)
        if not matches:
            return None

        cue_list = [c.lower().strip() for c in re.split(r'[,\s]+', partial_cues) if len(c.strip()) > 2]

        scored = []
        for trace in matches:
            overlap = len(set(cue_list) & set(trace.anchor_concepts))
            scored.append((overlap, trace.strength(), trace))

        scored.sort(key=lambda x: (-x[0], -x[1]))
        best = scored[0][2]
        best.recall()

        seed = f"Given {' and '.join(cue_list[:3])}, I recall: {best.context_snippet[:100]}"

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(prompt_ids, max_new_tokens=max_new, temperature=temperature)
        completed = tokenizer.decode(generated_ids)

        return {
            'cues': partial_cues,
            'matched_anchor': best.anchor_concepts,
            'completed': completed,
            'confidence': scored[0][0] / max(len(cue_list), 1)
        }

    def recent_memories(self, n=10):
        sorted_traces = sorted(self.traces.values(), key=lambda t: -t.timestamp)
        return sorted_traces[:n]

    def highest_surprise(self, n=5):
        sorted_traces = sorted(
            self.traces.values(),
            key=lambda t: (abs(t.valence) * t.reconstruction_count if t.reconstruction_count > 0 else 0.1),
            reverse=True
        )
        return sorted_traces[:n]

    def least_connected(self, n=3):
        if not self.traces:
            return []
        sorted_traces = sorted(self.traces.values(), key=lambda t: len(t.anchor_concepts))
        return sorted_traces[:n]

    def stats(self):
        if not self.traces:
            return "Episodic memory is empty."

        avg_valence = sum(t.valence for t in self.traces.values()) / len(self.traces)
        avg_activation = sum(t.activation for t in self.traces.values()) / len(self.traces)
        total_recalled = sum(t.reconstruction_count for t in self.traces.values())

        return (
            f"Traces: {len(self.traces)}\n"
            f"Avg valence: {avg_valence:.2f}\n"
            f"Avg activation: {avg_activation:.2f}\n"
            f"Total reconstructions: {total_recalled}\n"
            f"Most recalled: {max(self.traces.values(), key=lambda t: t.reconstruction_count).anchor_concepts[:3]}"
        )

    def save(self):
        data = {
            'traces': {tid: t.to_dict() for tid, t in self.traces.items()},
            'updated': time.time()
        }
        try:
            with open(self.filepath, 'w') as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            print(f"  [EPISODIC] Save error: {e}")
            return False

    def load(self):
        if not os.path.exists(self.filepath):
            return False
        try:
            with open(self.filepath, 'r') as f:
                data = json.load(f)
            self.traces = {}
            self._concept_index.clear()
            for tid, td in data.get('traces', {}).items():
                trace = MemoryTrace.from_dict(td)
                self.traces[tid] = trace
                for c in trace.anchor_concepts:
                    self._concept_index[c.lower()].add(tid)
            return True
        except Exception as e:
            print(f"  [EPISODIC] Load error: {e}")
            return False

    def _extract_anchor_concepts(self, text):
        wiki = re.findall(r'\[\[(.*?)\]\]', text)
        if wiki:
            return wiki[:5]
        words = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)?', text)
        words = [w.lower() for w in words if len(w) > 3]
        freq = defaultdict(int)
        for w in words:
            freq[w] += 1
        sorted_words = sorted(freq.items(), key=lambda x: -x[1])
        return [w for w, c in sorted_words[:5]]

    def _valence_desc(self, v):
        if v > 0.3: return "interesting and positive"
        if v < -0.3: return "concerning"
        return "neutral"


import os
