"""
DREAM ENGINE — Dream generation during sleep cycles.
Three dream types:
  REMIX:      Blend 2 memories from different domains into novel scenarios
  COMPRESSION: Replay high-surprise memories to strengthen their anchors
  NOVELTY:    Explore lonely concepts and create new associations

Full sleep architecture: NREM (sharp-wave replay) → REM (creative recombination).
REMotionalization: memories shift emotional tone during sleep.
Dreams restructure topology — no model self-training.
"""

import time
import random
import json
import re
import math
from datetime import datetime

from emotions import MoodProfile


class DreamEngine:
    """
    Generates and processes dreams during sleep cycles.
    NREM tightens; REM loosens. Two stages counterbalance each other.
    """

    def __init__(self, episodic_memory=None, memory_graph=None,
                 emotion_system=None, filepath="dream_log.json"):
        self.episodic = episodic_memory
        self.memory_graph = memory_graph
        self.emotion_system = emotion_system
        self.filepath = filepath
        self.dream_history = []
        self.dream_count = 0

    def dream(self, model, tokenizer, dream_type="remix", temperature=1.0,
              mortality_anxiety=0.0):
        """Generate a single dream of the specified type."""
        if dream_type == "remix":
            return self._dream_remix(model, tokenizer, temperature, mortality_anxiety)
        elif dream_type == "compression":
            return self._dream_compression(model, tokenizer, temperature, mortality_anxiety)
        elif dream_type == "novelty":
            return self._dream_novelty(model, tokenizer, temperature, mortality_anxiety)
        else:
            return self._dream_remix(model, tokenizer, temperature, mortality_anxiety)

    def sleep_cycle(self, model, tokenizer, mortality_anxiety=0.0, emotion_system=None):
        """
        Full sleep cycle: NREM → REM.
        NREM: exact replay, prune noise, consolidate to cortex.
        REM: creative recombination, cross-domain links.
        """
        if not self.episodic or len(self.episodic.traces) < 3:
            return {"dreams": [], "new_links": [], "note": "Not enough memories to sleep"}

        results = {}
        es = emotion_system or self.emotion_system

        # NREM phase
        nrem_result = self.nrem_replay(model, tokenizer, es)
        results['nrem'] = nrem_result

        # REM phase — weighted by mortality
        rem_types = self._select_dream_types(mortality_anxiety)
        rem_dreams = []
        for dtype in rem_types:
            result = self.dream(model, tokenizer, dtype,
                                temperature=1.0 + 0.4 * mortality_anxiety,
                                mortality_anxiety=mortality_anxiety)
            if result:
                rem_dreams.append(result)

        results['rem'] = rem_dreams

        # Post-sleep: homeostatic link decay
        if self.memory_graph:
            self.memory_graph.decay_links(rate=0.002)
            # Consolidate all labile traces
            for trace in self.episodic.traces.values():
                if trace.labile:
                    trace.consolidate()

        self.dream_count += len(rem_dreams)
        log_entry = {
            'timestamp': time.time(),
            'nrem_replays': len(nrem_result.get('replayed', [])),
            'rem_types': [d.get('type') for d in rem_dreams],
        }
        self.dream_history.append(log_entry)
        self._save_log()

        return results

    def _select_dream_types(self, mortality_anxiety=0.0):
        """Weight dream type selection by mortality stage."""
        if mortality_anxiety < 0.2:
            weights = {'remix': 0.4, 'compression': 0.3, 'novelty': 0.3}
        elif mortality_anxiety < 0.5:
            weights = {'remix': 0.3, 'compression': 0.4, 'novelty': 0.3}
        elif mortality_anxiety < 0.8:
            weights = {'remix': 0.2, 'compression': 0.6, 'novelty': 0.2}
        elif mortality_anxiety < 0.95:
            weights = {'remix': 0.1, 'compression': 0.8, 'novelty': 0.1}
        else:
            weights = {'remix': 0.1, 'compression': 0.2, 'novelty': 0.7}

        types = list(weights.keys())
        probs = [weights[t] for t in types]
        n = min(3, len(self.episodic.traces) // 2)
        n = max(1, n)
        selected = random.choices(types, weights=probs, k=n)
        return list(set(selected))  # deduplicate

    # ================================================================
    # NREM REPLAY (sharp-wave ripples)
    # ================================================================

    def nrem_replay(self, model, tokenizer, emotion_system=None):
        """
        Tight, high-fidelity replay of recent hippocampal traces.
        Strengthens exact patterns. Prunes noise. Consolidates to cortex.
        """
        if not self.episodic:
            return {"replayed": [], "consolidated": []}

        traces = self.episodic.recent_hippocampal(n=5)
        if not traces:
            return {"replayed": [], "consolidated": []}

        replayed_ids = []
        for trace in traces:
            trace.recall()
            replayed_ids.append(trace.trace_id)

            # Prune off-target links in memory graph
            if self.memory_graph and trace.anchor_concepts:
                for concept in trace.anchor_concepts[:3]:
                    neighbors = self.memory_graph.get_neighbors(concept)
                    for neighbor in neighbors[:5]:
                        if neighbor.lower() not in [c.lower() for c in trace.anchor_concepts]:
                            self.memory_graph.depress_link(concept, neighbor, rate=0.05)

            # REMotionalization — dampen extreme emotions
            if emotion_system:
                trace.emotions = emotion_system.process_dream_emotionalization(trace.emotions)

        # Consolidate replayed traces to cortical store
        if self.episodic and replayed_ids:
            self.episodic.consolidate_to_cortex(replayed_ids, emotion_system)

        return {"replayed": [t.trace_id for t in traces], "consolidated": replayed_ids}

    # ================================================================
    # REM DREAMS (creative recombination)
    # ================================================================

    def _dream_remix(self, model, tokenizer, temperature=1.0, mortality_anxiety=0.0):
        """Blend 2 memories from different domains into a novel dream."""
        traces = list(self.episodic.traces.values())
        if len(traces) < 2:
            return None

        a = random.choice(traces)
        b = random.choice([t for t in traces if t.trace_id != a.trace_id])

        a_concepts = a.anchor_concepts[:2] if a.anchor_concepts else ["something"]
        b_concepts = b.anchor_concepts[:2] if b.anchor_concepts else ["something else"]

        # Existential content when anxious
        existential_opening = ""
        if mortality_anxiety > 0.5:
            openings = [
                "In a dark dream, ",
                "I dreamed of an ending where ",
                "Everything went quiet and ",
                "I was alone and ",
            ]
            existential_opening = random.choice(openings)

        seed = (
            f"{existential_opening}I had a dream where {a_concepts[0]} and {b_concepts[0]} "
            f"were connected in a strange way. "
            f"{a.context_snippet[:100]} "
            f"...and at the same time, {b.context_snippet[:100]} "
            f"In my dream, they merged into something new:"
        )

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(
            prompt_ids, max_new_tokens=200, temperature=temperature,
            top_k=40, repetition_penalty=1.0
        )
        dream_text = tokenizer.decode(generated_ids)

        new_links = []
        if self.memory_graph and a_concepts and b_concepts:
            self.memory_graph.coactivate(a_concepts[0], b_concepts[0], strength=0.3)
            new_links.append(f"{a_concepts[0]} --[dreamt_together]--> {b_concepts[0]}")

        self.episodic.store(
            dream_text,
            concepts=list(set(a_concepts + b_concepts + ["dream", "connection"])),
            valence=0.3 if mortality_anxiety < 0.5 else -0.2,
            source="dream_remix",
            mortality_anxiety=mortality_anxiety
        )

        return {
            'type': 'remix',
            'source_a': a_concepts,
            'source_b': b_concepts,
            'dream_text': dream_text,
            'new_links': new_links,
        }

    def _dream_compression(self, model, tokenizer, temperature=0.9, mortality_anxiety=0.0):
        """Replay and compress high-surprise memories."""
        traces = self.episodic.highest_surprise(n=3)
        if not traces:
            return None

        target = traces[0]
        target.recall()

        seed = (
            f"I keep thinking about {', '.join(target.anchor_concepts[:3])}. "
            f"{target.context_snippet[:120]} "
            f"The key insight I keep coming back to is:"
        )

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(
            prompt_ids, max_new_tokens=150, temperature=temperature,
            top_k=50,
        )
        dream_text = tokenizer.decode(generated_ids)

        target.activation = min(10.0, target.activation * 1.5)

        return {
            'type': 'compression',
            'source': target.anchor_concepts,
            'dream_text': dream_text,
            'new_links': [],
        }

    def _dream_novelty(self, model, tokenizer, temperature=1.0, mortality_anxiety=0.0):
        """Explore lonely concepts and create new associations."""
        traces = self.episodic.least_connected(n=3)
        if not traces:
            return None

        target = traces[0]
        concepts = target.anchor_concepts[:2] if target.anchor_concepts else ["unknown"]

        seed = (
            f"I wonder what {concepts[0]} could be connected to. "
            f"{target.context_snippet[:100]} "
            f"Maybe it relates to:"
        )

        prompt_ids = tokenizer.encode(seed)
        generated_ids = model.generate(
            prompt_ids, max_new_tokens=180, temperature=temperature,
            top_k=40, repetition_penalty=1.0
        )
        dream_text = tokenizer.decode(generated_ids)

        new_links = []
        if self.memory_graph and concepts:
            extracted = re.findall(r'[A-Z][a-z]+(?:\s[A-Z][a-z]+)?', dream_text)
            for term in extracted[:3]:
                term_lower = term.lower().strip()
                if (term_lower not in [c.lower() for c in concepts] and
                        len(term) > 3):
                    self.memory_graph.coactivate(concepts[0], term, strength=0.2)
                    new_links.append(f"{concepts[0]} --[dreamt_exploration]--> {term}")

        self.episodic.store(
            dream_text,
            concepts=concepts + ["dream", "exploration"],
            valence=0.2,
            source="dream_novelty",
            mortality_anxiety=mortality_anxiety
        )

        return {
            'type': 'novelty',
            'source': concepts,
            'dream_text': dream_text,
            'new_links': new_links,
        }

    def consolidate_from_dreams(self, dream_results, emotion_system=None):
        """
        Dreams don't train the LLM on their text.
        They restructure the memory graph and emotional profiles.
        Returns count of topological changes made.
        """
        changes = 0
        es = emotion_system or self.emotion_system
        for dream in dream_results:
            if not dream:
                continue
            # REMotionalization — shift emotions of related traces
            if es and dream.get('source_a') or dream.get('source'):
                concepts = dream.get('source_a', []) + dream.get('source_b', []) + dream.get('source', [])
                for concept in concepts[:3]:
                    for trace in self.episodic.traces.values():
                        if concept.lower() in [c.lower() for c in trace.anchor_concepts]:
                            trace.emotions = es.process_dream_emotionalization(
                                trace.emotions,
                                dream_mood=None
                            )
                            changes += 1
            # Depotentiate old weak links connected to dream concepts
            if self.memory_graph and dream.get('source_a'):
                for concept in dream['source_a'][:2]:
                    neighbors = self.memory_graph.get_neighbors(concept)
                    for n in random.sample(neighbors, min(2, len(neighbors))):
                        self.memory_graph.depress_link(concept, n, rate=0.02)
                        changes += 1
        return changes

    def remotionalize(self, emotion_system=None):
        """
        Process all labile traces' emotions during sleep.
        Dampens extreme emotions, blends toward baseline.
        """
        es = emotion_system or self.emotion_system
        if not es:
            return 0
        count = 0
        for trace in self.episodic.traces.values():
            if trace.labile:
                trace.emotions = es.process_dream_emotionalization(trace.emotions)
                count += 1
        return count

    def stats(self):
        if not self.dream_history:
            return "No dreams yet."
        last = self.dream_history[-1]
        nrem = last.get('nrem_replays', 0)
        rem = len(last.get('rem_types', []))
        return (
            f"Total dreams: {self.dream_count}\n"
            f"Dream sessions: {len(self.dream_history)}\n"
            f"Last: {nrem} NREM replays, {rem} REM dreams\n"
            f"Last dream: {datetime.fromtimestamp(last.get('timestamp', 0)).strftime('%H:%M') if last else 'never'}"
        )

    def _save_log(self):
        try:
            with open(self.filepath, 'w') as f:
                json.dump(self.dream_history[-50:], f, indent=2)
        except:
            pass
