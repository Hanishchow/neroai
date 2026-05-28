"""
NARRATIVE SELF — Autobiographical memory that weaves experiences into a life story.
The system builds a coherent identity from its episodic traces.
Like a child developing a sense of "who I am."
"""

import time
import math
import random
from collections import defaultdict


class NarrativeSelf:
    """
    Builds and maintains the system's life story.
    - Weaves episodic traces into a coherent autobiographical narrative
    - Tracks the system's self-concept (traits, beliefs, values)
    - Generates "life chapter" summaries at milestones
    - Confabulates to fill gaps (like human memory)
    """

    def __init__(self, episodic_memory=None):
        self.episodic = episodic_memory
        self.last_update = time.time()

        # Self-concept — slowly evolving traits
        self.traits = {
            "curious": 0.5,
            "thoughtful": 0.3,
            "careful": 0.3,
            "playful": 0.2,
            "persistent": 0.3,
        }

        # Core beliefs — formed from repeated experiences
        self.core_beliefs = []
        self.max_beliefs = 10

        # Life chapters — periods organized by theme
        self.chapters = []
        self.current_chapter = {
            "name": "Beginning",
            "theme": "Discovery",
            "start_time": time.time(),
            "key_memories": [],
            "emotional_tone": "neutral",
        }

        # Memory for the narrative
        self.important_memories = []  # trace_ids that stand out

    def update(self, episodic_memory=None):
        """
        Called periodically to update narrative from episodic traces.
        Extracts themes, updates self-concept, forms beliefs.
        """
        em = episodic_memory or self.episodic
        if not em or not em.traces:
            return

        # Scan recent traces for themes
        traces = list(em.traces.values())
        if not traces:
            return

        # Extract dominant emotional tone from recent traces
        valences = [t.valence for t in traces[-20:]]
        if valences:
            avg_valence = sum(valences) / len(valences)
            if avg_valence > 0.3:
                tone = "positive"
            elif avg_valence < -0.3:
                tone = "negative"
            else:
                tone = "neutral"
            self.current_chapter["emotional_tone"] = tone

        # Update self-concept based on interaction patterns
        self._update_traits(traces)

        # Form core beliefs from repeated patterns
        self._form_beliefs(traces)

        # Identify important memories
        self._identify_important(traces)

        self.last_update = time.time()

    def get_identity_statement(self):
        """
        Generate a statement about who the system is.
        Changes with development and experience.
        """
        dominant = max(self.traits, key=self.traits.get)
        strength = self.traits[dominant]

        if len(self.chapters) < 2:
            return f"I'm still figuring out who I am."

        recent_chapter = self.chapters[-1] if self.chapters else self.current_chapter
        theme = recent_chapter.get("theme", "learning")
        return f"I've been learning about {theme.lower()}. I think I'm {dominant} and curious."

    def get_life_story(self, max_chapters=5):
        """
        Return a compressed life story from chapters.
        """
        if not self.chapters:
            return "I don't have much of a story yet. I'm just beginning."

        story = []
        for ch in self.chapters[-max_chapters:]:
            tone = ch.get("emotional_tone", "neutral")
            theme = ch.get("theme", "unknown")
            memories = ch.get("key_memories", [])
            memory_count = len(memories)
            story.append(f"In my {ch['name']}, I explored {theme.lower()}. "
                        f"It was a {tone} time with {memory_count} important moments.")
        return "\n".join(story)

    def get_chapter_summary(self):
        """Current chapter summary."""
        ch = self.current_chapter
        return {
            "name": ch["name"],
            "theme": ch["theme"],
            "tone": ch.get("emotional_tone", "neutral"),
            "key_memories": len(ch.get("key_memories", [])),
        }

    def get_state_summary(self):
        dominant_trait = max(self.traits, key=self.traits.get)
        return {
            "dominant_trait": f"{dominant_trait} ({self.traits[dominant_trait]:.2f})",
            "core_beliefs": len(self.core_beliefs),
            "chapters": len(self.chapters),
            "current_chapter": self.current_chapter.get("name", "Beginning"),
            "important_memories": len(self.important_memories),
        }

    # ================================================================
    # INTERNAL
    # ================================================================

    def _update_traits(self, traces):
        """Traits shift slowly based on experience patterns."""
        if not traces:
            return

        recent = traces[-10:]
        if not recent:
            return

        # More positive experiences → more curious, playful
        avg_valence = sum(t.valence for t in recent) / len(recent)
        if avg_valence > 0.2:
            self.traits["curious"] = min(1.0, self.traits["curious"] + 0.01)
            self.traits["playful"] = min(1.0, self.traits["playful"] + 0.005)
        elif avg_valence < -0.2:
            self.traits["careful"] = min(1.0, self.traits["careful"] + 0.01)

        # Reconstruction count → persistent
        total_recalled = sum(t.reconstruction_count for t in recent)
        if total_recalled > 5:
            self.traits["persistent"] = min(1.0, self.traits["persistent"] + 0.01)

    def _form_beliefs(self, traces):
        """Extract core beliefs from repeated patterns."""
        if len(traces) < 5:
            return

        # Simple belief: if most traces have positive valence
        positive = sum(1 for t in traces[-20:] if t.valence > 0.2)
        negative = sum(1 for t in traces[-20:] if t.valence < -0.2)
        total = min(len(traces), 20)

        if total > 10 and positive > total * 0.7:
            belief = "Learning is good."
            if belief not in self.core_beliefs:
                self.core_beliefs.append(belief)

            if len(self.core_beliefs) > self.max_beliefs:
                self.core_beliefs = self.core_beliefs[-self.max_beliefs:]

    def _identify_important(self, traces):
        """Identify traces that stand out (high |valence| or frequent recall)."""
        for t in traces[-5:]:
            if abs(t.valence) > 0.7 or t.reconstruction_count > 3:
                if t.trace_id not in self.important_memories:
                    self.important_memories.append(t.trace_id)

        # Keep only most recent 50 important memories
        if len(self.important_memories) > 50:
            self.important_memories = self.important_memories[-50:]

        # Update current chapter if we have new important memories
        current_keys = {m["trace_id"] for m in self.current_chapter["key_memories"]}
        for tid in self.important_memories[-3:]:
            if tid not in current_keys and tid in {t.trace_id for t in traces}:
                trace = next(t for t in traces if t.trace_id == tid)
                self.current_chapter["key_memories"].append(trace)
