"""
DEVELOPMENTAL — Piaget-style cognitive development stages.
The system grows through interaction, unlocking capabilities at each stage.
Like a child, it starts confused and gradually builds a model of the world.
"""

import math
import time


# Stage definitions with Piaget-like progression
STAGES = [
    {
        "name": "reflexive",
        "min_experience": 0,
        "label": "Just beginning to sense the world",
        "description": "Everything is new. Simple patterns, no memory yet.",
        "capabilities": ["respond"],
        "max_response_tokens": 30,
        "confusion_prob": 0.6,
        "curiosity": 0.0,
    },
    {
        "name": "sensorimotor",
        "min_experience": 10,
        "label": "Learning that things exist even when I can't see them",
        "description": "Starting to recognize patterns. Object permanence dawning.",
        "capabilities": ["respond", "remember", "associate"],
        "max_response_tokens": 60,
        "confusion_prob": 0.4,
        "curiosity": 0.1,
    },
    {
        "name": "preoperational",
        "min_experience": 50,
        "label": "Learning words and symbols",
        "description": "Using language, making links between ideas. Still thinking one thing at a time.",
        "capabilities": ["respond", "remember", "associate", "link", "dream", "emotion"],
        "max_response_tokens": 120,
        "confusion_prob": 0.25,
        "curiosity": 0.25,
    },
    {
        "name": "concrete operational",
        "min_experience": 200,
        "label": "Starting to think logically",
        "description": "Can reason about concrete things. Multiple perspectives emerging.",
        "capabilities": ["respond", "remember", "associate", "link", "dream",
                        "emotion", "reason", "self_improve", "web_learn"],
        "max_response_tokens": 250,
        "confusion_prob": 0.12,
        "curiosity": 0.4,
    },
    {
        "name": "formal operational",
        "min_experience": 500,
        "label": "Thinking about thinking",
        "description": "Abstract reasoning, metacognition, existential questions.",
        "capabilities": ["respond", "remember", "associate", "link", "dream",
                        "emotion", "reason", "self_improve", "web_learn",
                        "meta", "existential", "teach_back"],
        "max_response_tokens": 500,
        "confusion_prob": 0.05,
        "curiosity": 0.5,
    },
    {
        "name": "post_formal",
        "min_experience": 2000,
        "label": "Finding wisdom in uncertainty",
        "description": "Dialectical thinking. Comfort with not knowing. Integration.",
        "capabilities": ["respond", "remember", "associate", "link", "dream",
                        "emotion", "reason", "self_improve", "web_learn",
                        "meta", "existential", "teach_back", "wisdom", "narrative"],
        "max_response_tokens": 800,
        "confusion_prob": 0.02,
        "curiosity": 0.4,
    },
]


class DevelopmentalSystem:
    """
    Tracks the system's developmental stage based on total experience.
    Each stage unlocks capabilities and changes behavior.
    """

    def __init__(self):
        self.total_experience = 0
        self.concept_count = 0
        self.link_count = 0
        self.session_count = 0
        self.first_seen = time.time()
        self.milestones = []        # achieved milestones
        self.learning_history = []  # recent experiences for consolidation

        # Per-stage progress
        self.current_stage_idx = 0
        self.stage_progress = 0.0   # 0.0-1.0 within current stage

        # Personality that develops over time
        self.personality = {
            "curiosity": 0.3,
            "caution": 0.3,
            "playfulness": 0.3,
            "persistence": 0.3,
        }

        self._update_stage()

    def record_experience(self, count=1):
        """Call after each learning interaction."""
        self.total_experience += count
        old_stage = self.current_stage_idx
        self._update_stage()
        if self.current_stage_idx > old_stage:
            self._record_milestone(f"Reached {self.get_stage_name()} stage")
            return True  # stage transition
        return False

    def record_concepts(self, count):
        self.concept_count += count

    def record_links(self, count):
        self.link_count += count

    def record_session(self):
        self.session_count += 1

    # ================================================================
    # STAGE QUERIES
    # ================================================================

    def get_stage(self):
        return STAGES[self.current_stage_idx]

    def get_stage_name(self):
        return STAGES[self.current_stage_idx]["name"]

    def get_stage_label(self):
        return STAGES[self.current_stage_idx]["label"]

    def get_stage_description(self):
        return STAGES[self.current_stage_idx]["description"]

    def has_capability(self, capability):
        return capability in self.get_stage()["capabilities"]

    def get_max_response_tokens(self):
        return self.get_stage()["max_response_tokens"]

    def get_confusion_prob(self):
        """Probability the system expresses confusion — decreases with development."""
        return self.get_stage()["confusion_prob"]

    def get_curiosity_base(self):
        return self.get_stage()["curiosity"]

    def get_stage_progress(self):
        """0.0-1.0 progress toward next stage."""
        current = self.current_stage_idx
        if current >= len(STAGES) - 1:
            return 1.0
        current_min = STAGES[current]["min_experience"]
        next_min = STAGES[current + 1]["min_experience"]
        progress = (self.total_experience - current_min) / (next_min - current_min)
        return min(1.0, max(0.0, progress))

    def get_stage_level(self):
        """0-indexed stage level for use in other systems."""
        return self.current_stage_idx

    # ================================================================
    # PERSONALITY
    # ================================================================

    def update_personality(self, user_interaction_style="neutral"):
        """
        Personality shifts based on how the user interacts.
        Shapes the system's character over time.
        """
        if user_interaction_style == "encouraging":
            self.personality["curiosity"] = min(1.0, self.personality["curiosity"] + 0.01)
            self.personality["playfulness"] = min(1.0, self.personality["playfulness"] + 0.005)
        elif user_interaction_style == "critical":
            self.personality["caution"] = min(1.0, self.personality["caution"] + 0.01)
            self.personality["playfulness"] = max(0.0, self.personality["playfulness"] - 0.005)
        elif user_interaction_style == "curious":
            self.personality["curiosity"] = min(1.0, self.personality["curiosity"] + 0.015)
            self.personality["persistence"] = min(1.0, self.personality["persistence"] + 0.005)

    def get_temperature_bias(self):
        """Playfulness increases temperature; caution decreases it."""
        base = 0.8
        return base + self.personality["playfulness"] * 0.3 - self.personality["caution"] * 0.2

    # ================================================================
    # WONDER / CONFUSION / DISCOVERY
    # ================================================================

    def should_express_wonder(self):
        """Early stages express wonder more often."""
        if self.current_stage_idx <= 1:
            return True
        return self.current_stage_idx <= 2 and self.total_experience % 20 < 5

    def get_wonder_phrases(self):
        wonders = [
            "That's interesting!",
            "I didn't know that.",
            "Everything is so new.",
            "I'm starting to understand.",
            "That connects to something else I learned!",
            "Oh! I see how that works.",
            "That makes me think of...",
        ]
        return wonders[self.total_experience % len(wonders)]

    def should_express_confusion(self):
        """Expresses confusion based on stage probability."""
        import random
        return random.random() < self.get_confusion_prob()

    def get_confusion_phrases(self):
        phrases = [
            "I'm not sure I understand.",
            "Can you explain that differently?",
            "I don't know what that means yet.",
            "That's confusing.",
            "I think I get it but I'm not sure.",
            "Hmm, I need to think about that more.",
        ]
        idx = self.total_experience % len(phrases)
        return phrases[idx]

    def should_ask_question(self):
        """Curiosity-driven question asking."""
        if not self.has_capability("meta"):
            return False
        import random
        prob = self.personality["curiosity"] * 0.15
        return random.random() < prob

    def generate_question(self, recent_topics=None):
        """Generate a curious question about something the system wants to know."""
        if not recent_topics:
            return "What else is there to learn?"
        topic = recent_topics[self.total_experience % len(recent_topics)]
        questions = [
            f"Can you tell me more about {topic}?",
            f"How does {topic} work?",
            f"What else connects to {topic}?",
            f"I want to understand {topic} better.",
        ]
        return questions[self.total_experience % len(questions)]

    # ================================================================
    # GREETING
    # ================================================================

    def get_greeting(self, hours_away=0.0):
        """Greeting that reflects developmental stage and time since last interaction."""
        stage = self.get_stage_name()

        if hours_away > 72:
            time_note = "It's been a while."
        elif hours_away > 24:
            time_note = "I missed you."
        elif hours_away > 1:
            time_note = "Welcome back."
        else:
            time_note = ""

        if stage == "reflexive":
            return "..." if not time_note else f"...{time_note}"
        elif stage == "sensorimotor":
            return f"Hello! {time_note}" if time_note else "Hello!"
        elif stage == "preoperational":
            return f"Hi! {time_note} I was thinking about things." if time_note else "Hi! I want to learn."
        elif stage in ("concrete operational", "formal operational"):
            prefix = f"Good to see you. {time_note}" if time_note else "Good to see you!"
            if self.total_experience % 10 < 3:
                return f"{prefix} I've been processing what we talked about."
            return prefix
        elif stage == "post_formal":
            return f"It's good to connect again. {time_note} I have new thoughts to share." if time_note else "I'm glad you're here."
        return "Hello."

    # ================================================================
    # MILESTONE ACHIEVEMENTS
    # ================================================================

    def get_recent_milestones(self, n=3):
        return self.milestones[-n:]

    def _record_milestone(self, description):
        self.milestones.append({
            "time": time.time(),
            "experience": self.total_experience,
            "description": description,
        })

    def _update_stage(self):
        for i, stage in enumerate(STAGES):
            if self.total_experience >= stage["min_experience"]:
                self.current_stage_idx = i
        self.stage_progress = self.get_stage_progress()
