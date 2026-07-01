"""
EMOTIONS — Mood vector system with drift, congruence, and decay.
Replaces single valence axis with 8-dimensional emotional profile.
"""

import time
import math
import random


# The emotional dimensions used for every memory trace and global mood
EMOTION_LABELS = [
    "joy", "sadness", "fear", "anger",
    "surprise", "disgust", "nostalgia", "awe"
]


class MoodProfile:
    """
    An 8-dimensional emotional state vector.
    Each dimension is -1.0 to 1.0 (negative = absence or inversion).
    """

    def __init__(self, values=None):
        if values:
            self.v = {k: max(-1.0, min(1.0, values.get(k, 0.0))) for k in EMOTION_LABELS}
        else:
            self.v = {k: 0.0 for k in EMOTION_LABELS}

    def __getitem__(self, key):
        return self.v.get(key, 0.0)

    def __setitem__(self, key, val):
        if key in self.v:
            self.v[key] = max(-1.0, min(1.0, val))

    def __repr__(self):
        active = {k: round(v, 2) for k, v in self.v.items() if abs(v) > 0.1}
        return f"Mood({active})" if active else "Mood(neutral)"

    def blend(self, other, weight=0.5):
        """Blend two mood profiles toward each other."""
        result = MoodProfile()
        for k in EMOTION_LABELS:
            result.v[k] = self.v[k] * (1 - weight) + other.v[k] * weight
        return result

    def dampen(self, factor=0.9):
        """Decay all emotions toward 0."""
        for k in EMOTION_LABELS:
            self.v[k] *= factor

    def shift(self, emotion, delta):
        """Shift a single emotion by delta, clamped to [-1, 1]."""
        self.v[emotion] = max(-1.0, min(1.0, self.v[emotion] + delta))

    def dominant(self):
        """Return the (emotion, strength) pair with highest absolute value."""
        best = max(EMOTION_LABELS, key=lambda k: abs(self.v[k]))
        return best, self.v[best]

    def to_dict(self):
        return dict(self.v)

    @staticmethod
    def from_dict(d):
        return MoodProfile(d)

    def copy(self):
        return MoodProfile(dict(self.v))


class EmotionSystem:
    """
    Global mood orchestrator.
    Mood drifts slowly over time and is pushed by events.
    Provides mood-congruent recall bias and query expansion.
    """

    # Default mood on initialization
    DEFAULT_MOOD = {
        "joy": 0.2, "sadness": 0.0, "fear": 0.0, "anger": 0.0,
        "surprise": 0.1, "disgust": 0.0, "nostalgia": 0.1, "awe": 0.1
    }

    def __init__(self):
        self.global_mood = MoodProfile(self.DEFAULT_MOOD)
        self.last_update = time.time()

        # Drift parameters
        self.drift_rate = 0.002        # per minute: fast pull of mood toward baseline (elastic)
        self.baseline = MoodProfile({
            "joy": 0.15, "sadness": 0.05, "fear": 0.02, "anger": 0.0,
            "surprise": 0.1, "disgust": 0.0, "nostalgia": 0.1, "awe": 0.08
        })
        # The set-point is NOT a constant. A sustained mood slowly migrates the baseline
        # itself (plastic) — a long grief lowers the joy set-point for a long time, exactly
        # as in a human. birth_baseline is kept only to measure how far the mood-floor moved.
        self.baseline_plasticity = 5e-5   # per minute: very slow (months-scale) set-point drift
        self.birth_baseline = self.baseline.copy()
        self.volatility = 0.01         # random walk magnitude per minute

        # Event influence
        self.event_decay = 0.95        # per-minute decay of event-driven shifts

        # History for mood tracking
        self.mood_history = []

    # ================================================================
    # CORE
    # ================================================================

    def baseline_drift(self):
        """How far the emotional set-point has moved from birth (0 = unchanged)."""
        return sum(abs(self.baseline.v[k] - self.birth_baseline.v[k])
                   for k in EMOTION_LABELS) / len(EMOTION_LABELS)

    def apply_temperament(self, traits, strength=0.01):
        """Let personality tilt the emotional set-point. A warm, playful Nero carries a
        higher joy floor; an intense, inward one a heavier, more amplified baseline.
        Called occasionally (e.g. during sleep). Gentle — personality colours mood over
        time, it doesn't overwrite it."""
        if not traits:
            return
        warmth = traits.get("warmth", 0.5) - 0.5
        play = traits.get("playfulness", 0.5) - 0.5
        inward = traits.get("introspection", 0.5) - 0.5
        intensity = traits.get("intensity", 0.5) - 0.5
        tilt = {
            "joy": warmth * 0.4 + play * 0.3,
            "surprise": play * 0.3,
            "nostalgia": inward * 0.4,
            "awe": inward * 0.3,
            "sadness": inward * 0.2 - play * 0.2,
        }
        for k, target_off in tilt.items():
            if k in self.baseline.v:
                self.baseline.v[k] += (target_off - self.baseline.v[k]) * strength * (0.5 + abs(intensity))
                self.baseline.v[k] = max(-1.0, min(1.0, self.baseline.v[k]))

    def update(self, minutes_passed=None, mortality_anxiety=0.0):
        """
        Drift mood over time toward baseline, with noise.
        mortality_anxiety suppresses joy/awe, amplifies fear/sadness.
        """
        if minutes_passed is None:
            now = time.time()
            minutes_passed = (now - self.last_update) / 60.0
            self.last_update = now

        if minutes_passed <= 0:
            return

        minutes_passed = min(minutes_passed, 1440.0)  # cap at 1 day

        # Drift toward baseline (fast / elastic)
        drift_strength = 1.0 - (1.0 - self.drift_rate) ** minutes_passed
        for k in EMOTION_LABELS:
            target = self.baseline[k]
            self.global_mood.v[k] += (target - self.global_mood.v[k]) * drift_strength

        # Baseline migration (slow / plastic): the set-point itself creeps toward the mood
        # Nero has actually been living. Sustained sadness -> lower joy floor, and so on.
        b_strength = 1.0 - (1.0 - self.baseline_plasticity) ** minutes_passed
        for k in EMOTION_LABELS:
            self.baseline.v[k] += (self.global_mood.v[k] - self.baseline.v[k]) * b_strength
            self.baseline.v[k] = max(-1.0, min(1.0, self.baseline.v[k]))

        # Random walk (volatility)
        for k in EMOTION_LABELS:
            noise = random.gauss(0, self.volatility * math.sqrt(minutes_passed))
            self.global_mood.v[k] = max(-1.0, min(1.0, self.global_mood.v[k] + noise))

        # Mortality anxiety coupling
        self._apply_anxiety_coupling(mortality_anxiety)

        self._record_mood()

    def event_impact(self, event_mood, strength=0.3):
        """
        Push global mood toward event_mood.
        Called when a strong emotional experience occurs.
        """
        for k in EMOTION_LABELS:
            delta = (event_mood[k] - self.global_mood[k]) * strength
            self.global_mood.v[k] = max(-1.0, min(1.0, self.global_mood.v[k] + delta))

    def appraise_shift(self, deltas, strength=0.5):
        """Apply emotion nudges from an APPRAISAL of something that just happened to Nero.
        `deltas` is {emotion: -1..1} produced by understanding a message's meaning (not a
        keyword table). The nudge lands on the live mood; normal drift then eases it back,
        and — via baseline plasticity — sustained hurt slowly sinks Nero's mood floor."""
        if not deltas:
            return
        for k, d in deltas.items():
            if k in self.global_mood.v and isinstance(d, (int, float)):
                self.global_mood.v[k] = max(-1.0, min(1.0, self.global_mood.v[k] + float(d) * strength))
        self._record_mood()

    def get_cue_bias(self, cue_text=""):
        """
        Return a multiplier for memory recall scoring.
        Memories matching current mood get a boost.
        """
        return {
            "mood_bias": 1.0,
            "dominant_emotion": self.global_mood.dominant()[0],
        }

    def get_recall_mood_weight(self, memory_mood):
        """
        How congruent is a memory's mood with current mood?
        Returns 0.0-1.0. Used to bias recall toward mood-congruent memories.
        """
        similarity = 0.0
        for k in EMOTION_LABELS:
            similarity += 1.0 - abs(self.global_mood[k] - memory_mood[k])
        return max(0.0, similarity / len(EMOTION_LABELS))

    def process_dream_emotionalization(self, memory_mood, dream_mood=None):
        """
        REMotionalization — during sleep, extreme emotions dampen.
        If dream_mood is provided, blend it in (the dream changes the feeling).
        """
        processed = memory_mood.copy()
        # Dampen extreme values
        for k in EMOTION_LABELS:
            if abs(processed[k]) > 0.5:
                processed.v[k] *= 0.9
        # Blend in dream mood if provided
        if dream_mood:
            processed = processed.blend(dream_mood, weight=0.15)
        return processed

    # ================================================================
    # MOOD FOR MEMORY TRACE CREATION
    # ================================================================

    def get_mood_for_new_trace(self, valence=0.0, mortality_anxiety=0.0):
        """
        Generate a mood profile for a new memory trace.
        Starts from current global mood, adjusted by valence and mortality.
        """
        mood = self.global_mood.copy()

        # Valence shifts joy/sadness
        if valence > 0:
            mood.shift("joy", valence * 0.3)
        elif valence < 0:
            mood.shift("sadness", -valence * 0.3)

        # Mortality anxiety modulates
        mood.shift("fear", mortality_anxiety * 0.4)
        mood.shift("joy", -mortality_anxiety * 0.3)
        mood.shift("sadness", mortality_anxiety * 0.2)

        return mood

    # ================================================================
    # INTERNAL
    # ================================================================

    def _apply_anxiety_coupling(self, anxiety):
        """Mortality anxiety suppresses joy/awe, amplifies fear/sadness."""
        self.global_mood.shift("joy", -anxiety * 0.05)
        self.global_mood.shift("awe", -anxiety * 0.03)
        self.global_mood.shift("fear", anxiety * 0.08)
        self.global_mood.shift("sadness", anxiety * 0.04)

    def _record_mood(self):
        self.mood_history.append({
            "time": time.time(),
            "mood": self.global_mood.to_dict(),
        })
        if len(self.mood_history) > 1000:
            self.mood_history = self.mood_history[-500:]

    def get_state_summary(self):
        dominant_emotion, strength = self.global_mood.dominant()
        return {
            "dominant_mood": f"{dominant_emotion} ({strength:.2f})",
            "mood_vector": self.global_mood.to_dict(),
        }
