"""
CURIOSITY — Autonomous knowledge gap detection and exploration drive.
The system tracks what it knows and what it wants to learn next.
Like a child, it asks questions and explores what it finds interesting.
"""

import time
import math
import random


class CuriositySystem:
    """
    Drives the system to explore knowledge gaps.
    - Tracks known concepts and their "depth" of understanding
    - Identifies gaps: known concepts that are weakly connected
    - Generates questions the system wants answered
    - Autonomous exploration: seeks out novel topics when idle
    """

    def __init__(self):
        # Knowledge map: concept -> {"depth": 0-1, "links": count, "last_seen": time}
        self.knowledge = {}

        # Exploration targets
        self.exploration_queue = []      # topics to explore
        self.pending_questions = []      # questions the system wants answered
        self.active_topic = None         # current focus of exploration

        # Drive
        self.boredom = 0.0              # grows when nothing new is learned
        self.novelty_seeking = 0.5      # 0 (stick to known) to 1 (always new)
        self.exploration_energy = 1.0   # depletes on exploration, recharges on rest

        # History
        self.explored_topics = []
        self.questions_asked = 0

    # ================================================================
    # KNOWLEDGE TRACKING
    # ================================================================

    def record_exposure(self, concepts):
        """
        Called when the system encounters new concepts.
        Updates knowledge depth and identifies gaps.
        """
        for concept in concepts:
            key = concept.lower().strip()
            if key in self.knowledge:
                self.knowledge[key]["depth"] = min(1.0, self.knowledge[key]["depth"] + 0.1)
                self.knowledge[key]["exposures"] += 1
                self.knowledge[key]["last_seen"] = time.time()
            else:
                self.knowledge[key] = {
                    "depth": 0.1,
                    "exposures": 1,
                    "links": 0,
                    "last_seen": time.time(),
                }

        self._detect_gaps()

    def record_links(self, concept_a, concept_b):
        """Track when two concepts get linked."""
        for c in [concept_a, concept_b]:
            key = c.lower().strip()
            if key in self.knowledge:
                self.knowledge[key]["links"] += 1

    def _detect_gaps(self):
        """
        Find concepts that are known but weakly connected.
        These become exploration targets.
        """
        gaps = []
        for key, info in self.knowledge.items():
            if info["depth"] > 0.3 and info["links"] < 3:
                gaps.append(key)
        random.shuffle(gaps)
        self.exploration_queue = list(set(self.exploration_queue + gaps[:5]))

    # ================================================================
    # DRIVES
    # ================================================================

    def update(self, minutes_idle=0.0, mortality_anxiety=0.0, has_new_input=False):
        """
        Update curiosity drives.
        Boredom grows during idle. Novelty-seeking adjusts.
        """
        if has_new_input:
            self.boredom = max(0.0, self.boredom - 0.3)
            self.exploration_energy = min(1.0, self.exploration_energy + 0.1)
        elif minutes_idle > 0:
            idle_factor = minutes_idle / 60.0
            self.boredom = min(1.0, self.boredom + 0.02 * idle_factor)
            self.exploration_energy = max(0.0, self.exploration_energy - 0.01 * idle_factor)

        # Mortality anxiety suppresses novelty-seeking (frantic, not curious)
        self.novelty_seeking = max(0.0, min(1.0, 0.5 - mortality_anxiety * 0.3 + self.boredom * 0.4))

        # Recharge exploration energy when bored enough
        if self.boredom > 0.7 and self.exploration_energy < 0.5:
            self.exploration_energy = min(1.0, self.exploration_energy + 0.2)

    def get_exploration_urge(self):
        """0-1 — how much the system wants to explore right now."""
        if self.exploration_energy < 0.2:
            return 0.0
        return self.boredom * self.novelty_seeking * self.exploration_energy

    def should_explore(self):
        """Returns True if the system should initiate exploration."""
        return self.get_exploration_urge() > 0.4 and random.random() < 0.3

    # ================================================================
    # QUESTION GENERATION
    # ================================================================

    def generate_question(self, recent_context=""):
        """
        Generate a question the system wants answered.
        Targets knowledge gaps when possible.
        """
        if self.exploration_queue:
            topic = self.exploration_queue.pop(0)
            self.active_topic = topic
            questions = [
                f"What exactly is {topic}?",
                f"How does {topic} work?",
                f"Can you tell me more about {topic}?",
                f"What else is related to {topic}?",
            ]
            q = questions[random.randint(0, len(questions) - 1)]
            self.pending_questions.append({
                "question": q,
                "topic": topic,
                "timestamp": time.time(),
            })
            self.questions_asked += 1
            return q

        # No gaps — ask about recent context
        if recent_context:
            return f"Tell me more about what we're discussing."

        # Default curious questions
        defaults = [
            "What should I learn next?",
            "What's something interesting you know?",
            "What else is there?",
            "I want to understand the world better.",
        ]
        return random.choice(defaults)

    def is_curious(self):
        """High-level: is the system in a curious state right now?"""
        return self.boredom > 0.3 and self.exploration_energy > 0.4

    # ================================================================
    # STATE
    # ================================================================

    def get_state_summary(self):
        return {
            "boredom": round(self.boredom, 3),
            "novelty_seeking": round(self.novelty_seeking, 3),
            "exploration_energy": round(self.exploration_energy, 3),
            "known_concepts": len(self.knowledge),
            "exploration_queue": len(self.exploration_queue),
            "pending_questions": len(self.pending_questions),
            "active_topic": self.active_topic or "none",
        }
