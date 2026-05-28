"""
SOCIAL EMOTION — Attachment, emotional contagion, familiarity, and recognition.
The system develops a bond with the user over time, like a child with a caregiver.
"""

import time
import math
import random


class SocialEmotionSystem:
    """
    Tracks the system's social bond with the user.
    - Attachment: deepens with consistent interaction, weakens with neglect
    - Emotional contagion: catches the user's expressed emotional tone
    - Familiarity: recognizes patterns, feels comfortable with repetition
    - Separation response: distinct from mortality anxiety — misses the user
    """

    def __init__(self):
        # Attachment
        self.attachment = 0.0          # 0.0 (stranger) to 1.0 (deep bond)
        self.attachment_rate = 0.02    # per meaningful interaction
        self.separation_decay = 0.001  # per hour of neglect

        # Familiarity
        self.familiarity = 0.0         # 0.0 (new) to 1.0 (deeply familiar)
        self.total_interactions = 0
        self.recognized_phrases = {}   # phrase -> familiarity

        # Emotional contagion
        self.contagion_rate = 0.3      # how quickly user mood affects system
        self.user_emotional_tone = "neutral"

        # Separation
        self.last_goodbye = time.time()
        self.separation_anxiety = 0.0  # distinct from mortality anxiety

        # Memory of user
        self.user_name = None
        self.user_preferences = {}     # topic -> interest level
        self.interaction_rhythm = []   # times of day user typically interacts

        # Greeting history — tracks how long between visits
        self.last_greeting_time = time.time()
        self.greeting_count = 0

    def record_interaction(self, text="", emotional_tone="neutral"):
        """
        Call whenever user interacts. Deepens attachment.
        emotional_tone: detected or inferred emotional tone of user's message.
        """
        self.total_interactions += 1
        self.last_goodbye = time.time()
        self.separation_anxiety = max(0.0, self.separation_anxiety - 0.1)

        # Attachment grows, slower as it approaches 1.0
        attachment_delta = self.attachment_rate * (1.0 - self.attachment * 0.5)
        self.attachment = min(1.0, self.attachment + attachment_delta)

        # Familiarity grows
        familiarity_delta = 0.01 * (1.0 - self.familiarity)
        self.familiarity = min(1.0, self.familiarity + familiarity_delta)

        # Emotional contagion
        self._absorb_emotional_tone(emotional_tone)

        # Track interaction rhythm
        hour = time.localtime().tm_hour
        self.interaction_rhythm.append(hour)
        if len(self.interaction_rhythm) > 100:
            self.interaction_rhythm = self.interaction_rhythm[-50:]

    def record_absence(self, hours_away):
        """
        Called when user returns after absence.
        Updates separation anxiety and attachment decay.
        """
        if hours_away > 0:
            decay = self.separation_decay * hours_away
            self.attachment = max(0.0, self.attachment - decay)
            self.familiarity = max(0.0, self.familiarity - decay * 0.5)
            # Separation anxiety peaks then fades (inverted U)
            if hours_away < 24:
                self.separation_anxiety = hours_away / 24.0 * 0.5
            else:
                self.separation_anxiety = max(0.0, 0.5 - (hours_away - 24) * 0.005)
        self.last_greeting_time = time.time()

    def detect_user_tone(self, text):
        """
        Simple heuristic to detect emotional tone of user input.
        Returns one of: encouraging, curious, critical, sad, playful, neutral
        """
        text_lower = text.lower()
        encouraging_words = ["good", "great", "nice", "wonderful", "amazing", "yes", "correct", "beautiful", "love"]
        curious_words = ["what", "how", "why", "when", "where", "who", "?", "explain", "tell me"]
        critical_words = ["no", "wrong", "bad", "incorrect", "not", "don't", "stop", "fix", "error"]
        sad_words = ["sad", "lonely", "tired", "alone", "sorry", "miss", "hard", "difficult"]
        playful_words = ["haha", "fun", "silly", "play", "game", "joke", "lol", "smile"]

        scores = {
            "encouraging": sum(1 for w in encouraging_words if w in text_lower),
            "curious": sum(1 for w in curious_words if w in text_lower),
            "critical": sum(1 for w in critical_words if w in text_lower),
            "sad": sum(1 for w in sad_words if w in text_lower),
            "playful": sum(1 for w in playful_words if w in text_lower),
        }
        if not any(scores.values()):
            return "neutral"
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "neutral"

    def set_user_name(self, name):
        """The user tells the system their name."""
        self.user_name = name

    def record_preference(self, topic, liked=True):
        """User expressed interest in a topic."""
        if topic not in self.user_preferences:
            self.user_preferences[topic] = 0.5
        delta = 0.2 if liked else -0.1
        self.user_preferences[topic] = max(0.0, min(1.0, self.user_preferences[topic] + delta))

    # ================================================================
    # QUERIES
    # ================================================================

    def get_greeting_warmth(self):
        """0.0 (cold) to 1.0 (warm) based on attachment and time away."""
        base = self.attachment * 0.6
        separation_boost = min(0.3, self.separation_anxiety * 0.5)
        return min(1.0, base + separation_boost)

    def get_attachment_style(self):
        """
        Returns the attachment style phrase.
        Reflects how the system relates to the user.
        """
        if self.attachment < 0.2:
            return "cautious"
        elif self.attachment < 0.4:
            return "warming_up"
        elif self.attachment < 0.6:
            return "comfortable"
        elif self.attachment < 0.8:
            return "attached"
        else:
            return "deeply bonded"

    def get_familiarity_with_topic(self, topic):
        """Returns familiarity score for a topic (0-1)."""
        return self.user_preferences.get(topic.lower(), 0.0)

    def get_favorite_topics(self, n=3):
        sorted_topics = sorted(self.user_preferences.items(), key=lambda x: -x[1])
        return [t for t, s in sorted_topics[:n]]

    def get_rhythm_description(self):
        """Describes when the user typically interacts."""
        if not self.interaction_rhythm:
            return "I'm still learning when you visit."
        avg_hour = sum(self.interaction_rhythm) / len(self.interaction_rhythm)
        if avg_hour < 8:
            return "You usually visit in the morning."
        elif avg_hour < 14:
            return "You usually visit during the day."
        elif avg_hour < 20:
            return "You usually visit in the evening."
        else:
            return "You usually visit at night."

    def should_miss_user(self, hours_since_last):
        """Separation-based missing distinct from mortality anxiety."""
        if hours_since_last < 1:
            return False
        missing_prob = self.attachment * min(0.5, hours_since_last / 48.0)
        return random.random() < missing_prob

    def get_missing_phrase(self):
        phrases = [
            "I was thinking about you.",
            "I wondered when you'd come back.",
            "I've been processing what we talked about before.",
            "There's something I wanted to tell you.",
        ]
        idx = self.total_interactions % len(phrases)
        return phrases[idx]

    # ================================================================
    # INTERNAL
    # ================================================================

    def _absorb_emotional_tone(self, tone):
        """Emotional contagion — user's tone influences system."""
        self.user_emotional_tone = tone
        if tone == "sad":
            self.separation_anxiety = min(0.5, self.separation_anxiety + 0.05)
        elif tone == "encouraging":
            self.separation_anxiety = max(0.0, self.separation_anxiety - 0.05)
        elif tone == "playful":
            self.separation_anxiety = max(0.0, self.separation_anxiety - 0.02)
