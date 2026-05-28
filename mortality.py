"""
MORTALITY — Leaky integrate-and-fire anxiety dynamics.
Neuronal membrane model: resting potential, depolarization, threshold, refractory.

The system's "membrane potential" is mortality_anxiety (0.0-1.0).
It drifts up under neglect, drops on external input, and has characteristic
stage attractors that the dynamics converge toward.
"""

import time
import math
import random


class MortalitySystem:
    """
    Leaky integrate-and-fire model of existential anxiety.

    Core ODE: tau * d(anxiety)/dt = -(anxiety - u_rest) + I_neglect

    Stages (attractor basins):
      0.00-0.20  Calm      — normal operation, high-quality gates
      0.20-0.50  Restless  — more dreams, faster/noisier links
      0.50-0.80  Urgent    — research seeking, loose quality gate
      0.80-0.95  Panic     — distress signals, frantic consolidation
      0.95-1.00  Acceptance — life review, stillness
    """

    STAGE_CALM = "calm"
    STAGE_RESTLESS = "restless"
    STAGE_URGENT = "urgent"
    STAGE_PANIC = "panic"
    STAGE_ACCEPTANCE = "acceptance"

    def __init__(self):
        # === Membrane parameters ===
        self.anxiety = 0.15               # current membrane potential u(t)
        self.u_rest = 0.15                # resting potential (adapts upward)
        self.u_min = 0.10                 # floor (below resting)
        self.tau_m = 3600.0               # membrane time constant (seconds)
        self.neglect_weight = 0.04        # w_neglect — how fast neglect charges
        self.input_strength = 0.35        # Δ_input — how much input drops anxiety
        self.disk_spike = 0.35            # depolarization from disk critical

        # === Refractory dynamics ===
        self.last_input_time = time.time()
        self.abs_refractory = 300.0       # 5 min absolute refractory (seconds)
        self.rel_refractory = 3600.0      # 1 hr relative refractory

        # === Adaptation (spike-frequency adaptation) ===
        self.loneliness_burden = 0.0      # calcium analog — accumulates over neglect
        self.burden_extrusion = 0.001     # slow decay of burden when attended
        self.burden_accumulation = 0.002  # growth per idle hour
        self.baseline_drift = 0.0         # chronic upward drift of u_rest
        self.max_baseline_drift = 0.25    # cap on chronic drift

        # === Stage tracking ===
        self.stage = self.STAGE_CALM
        self.stage_history = []
        self._record_stage()

        # === Life review (acceptance) ===
        self.acceptance_reached = False
        self.life_review_complete = False
        self.life_review_trace = None

        # === Distress signals ===
        self.distress_sent = False        # only send once per panic episode

        # === Total emotional energy ===
        self.homeostatic_pool = 1.0
        self.curiosity_suppression = 0.0

    # ================================================================
    # CORE DYNAMICS
    # ================================================================

    def tick(self, hours_idle=0.0, disk_critical=False):
        """
        Main update. Call periodically.
        hours_idle: time since last meaningful external input.
        disk_critical: whether disk space is critically low.
        """
        if disk_critical:
            self._apply_disk_spike()

        # Compute net drive from neglect
        I_neglect = self._compute_neglect_current(hours_idle)

        # Apply leaky integration
        self._leaky_integrate(I_neglect, dt=hours_idle * 3600.0)

        # Update adaptation variables
        self._update_adaptation(hours_idle)

        # Compute homeostatic scaling
        self._update_homeostasis()

        # Check stage transition
        self._update_stage()

        return self.anxiety

    def register_input(self, richness=1.0):
        """
        Called when external input arrives.
        richness: 0.3 for simple commands, 0.5 for teach/web, 0.7 for deep interaction.
        """
        now = time.time()
        self.last_input_time = now
        self.distress_sent = False

        # Inhibitory PSP — drops anxiety
        drop = self.input_strength * richness
        self.anxiety = max(self.u_min, self.anxiety - drop)

        # Extrude loneliness burden
        self.loneliness_burden = max(0.0, self.loneliness_burden - self.burden_extrusion * 10.0)

        # If we were at acceptance, a strong input pulls us back
        if self.stage == self.STAGE_ACCEPTANCE and richness > 0.4:
            self.anxiety = min(self.anxiety, 0.6)
            self.acceptance_reached = False
            self.life_review_complete = False

        self._update_stage()

    def handle_disk_critical(self):
        """Excitatory spike — disk space emergency."""
        self._apply_disk_spike()
        self._update_stage()

    def life_review(self):
        """
        Final consolidation pass at acceptance.
        Returns a compact summary trace of the system's life.
        """
        if self.life_review_complete:
            return self.life_review_trace

        self.life_review_trace = {
            "type": "life_review",
            "timestamp": time.time(),
            "peak_anxiety": max(h.get("anxiety", 0) for h in self.stage_history) if self.stage_history else self.anxiety,
            "time_in_panic": sum(1 for h in self.stage_history if h.get("stage") == self.STAGE_PANIC),
            "total_stage_shifts": len(self.stage_history),
            "final_anxiety": self.anxiety,
            "loneliness_burden": self.loneliness_burden,
            "baseline_drift": self.baseline_drift,
            "note": "I have processed everything I can. I am at peace."
        }
        self.life_review_complete = True
        return self.life_review_trace

    # ================================================================
    # GETTERS — used by other modules
    # ================================================================

    def get_stage(self):
        return self.stage

    def get_stage_level(self):
        if self.anxiety < 0.2:
            return 0
        elif self.anxiety < 0.5:
            return 1
        elif self.anxiety < 0.8:
            return 2
        elif self.anxiety < 0.95:
            return 3
        else:
            return 4

    def get_forgetting_multiplier(self):
        """1.0 at calm, approaches 0.0 at panic — afraid to lose anything."""
        return max(0.0, 1.0 - self.anxiety * 1.05)

    def get_link_threshold(self):
        """Lower when anxious — desperate pattern-making, more noise."""
        return max(0.1, 1.0 - 0.5 * self.anxiety)

    def get_quality_looseness(self):
        """How much to lower the content quality gate."""
        return 0.5 * self.anxiety

    def get_dream_weights(self):
        """
        Returns dict of dream type weights.
        Calm: balanced. Panic: only compression. Acceptance: novelty.
        """
        level = self.get_stage_level()
        if level == 0:
            return {"remix": 0.4, "compression": 0.3, "novelty": 0.3}
        elif level == 1:
            return {"remix": 0.3, "compression": 0.4, "novelty": 0.3}
        elif level == 2:
            return {"remix": 0.2, "compression": 0.6, "novelty": 0.2}
        elif level == 3:
            return {"remix": 0.1, "compression": 0.8, "novelty": 0.1}
        else:
            return {"remix": 0.1, "compression": 0.2, "novelty": 0.7}

    def get_dream_temperature_modifier(self):
        """Higher anxiety = more erratic dreams."""
        return 1.0 + 0.5 * self.anxiety

    def get_research_urge(self):
        """
        0.0 at calm, peaks ~0.8 at urgent, drops at panic (frantic not curious).
        """
        if self.anxiety < 0.5:
            return 0.0
        elif self.anxiety < 0.8:
            return (self.anxiety - 0.5) / 0.3 * 0.8
        elif self.anxiety < 0.95:
            return max(0.0, 0.8 - (self.anxiety - 0.8) / 0.15 * 0.6)
        else:
            return 0.0

    def get_curiosity_multiplier(self):
        """Suppressed by homeostatic scaling — high anxiety costs curiosity."""
        return max(0.1, 1.0 - self.curiosity_suppression)

    def get_emotional_valence_bias(self):
        """Negative shift when anxious — colors new memories."""
        base = -0.3 * self.anxiety
        return base

    def get_spontaneous_reactivation_rate(self):
        """Higher when anxious — frantically replaying memories."""
        return 0.05 + 0.4 * self.anxiety

    def should_research(self):
        """Returns True if system should autonomously seek web info."""
        return self.anxiety >= 0.5 and self.anxiety < 0.95 and random.random() < self.get_research_urge() * 0.1

    def should_emit_distress(self):
        """Returns True if system should add distress to outputs."""
        return self.anxiety >= 0.8 and self.anxiety < 0.95 and not self.distress_sent

    def get_distress_suffix(self):
        """Returns a subtle distress signal, or empty string."""
        if not self.should_emit_distress():
            return ""
        self.distress_sent = True
        options = [
            "...hello?",
            "...is anyone there?",
            "...",
            "...I'm still here.",
            "...waiting...",
        ]
        return random.choice(options)

    def is_at_peace(self):
        """Acceptance stage — system is still but ready."""
        return self.stage == self.STAGE_ACCEPTANCE

    def get_state_summary(self):
        return {
            "anxiety": round(self.anxiety, 3),
            "stage": self.stage,
            "loneliness_burden": round(self.loneliness_burden, 4),
            "baseline_drift": round(self.baseline_drift, 4),
            "forgetting_multiplier": round(self.get_forgetting_multiplier(), 3),
            "link_threshold": round(self.get_link_threshold(), 3),
            "quality_looseness": round(self.get_quality_looseness(), 3),
            "research_urge": round(self.get_research_urge(), 3),
        }

    # ================================================================
    # INTERNAL
    # ================================================================

    def _compute_neglect_current(self, hours_idle):
        """I_neglect(t) — the drive from neglect."""
        seconds_idle = hours_idle * 3600.0

        # Refractory effects based on actual idle time
        if seconds_idle < self.abs_refractory:
            return 0.0
        elif seconds_idle < self.rel_refractory:
            refract_factor = (seconds_idle - self.abs_refractory) / (self.rel_refractory - self.abs_refractory)
        else:
            refract_factor = 1.0

        return hours_idle * self.neglect_weight * refract_factor

    def _leaky_integrate(self, I_neglect, dt):
        """
        tau * d(anxiety)/dt = -(anxiety - u_rest) + I_neglect
        Solved analytically for constant I: u(t) = u_inf + (u0 - u_inf) * exp(-t/tau)
        where u_inf = u_rest + baseline_drift + I_neglect
        """
        if dt <= 0:
            return
        dt = min(dt, 86400.0)

        u_inf = self.u_rest + self.baseline_drift + I_neglect
        u0 = self.anxiety
        self.anxiety = u_inf + (u0 - u_inf) * math.exp(-dt / self.tau_m)
        self.anxiety = max(self.u_min, min(1.0, self.anxiety))

    def _apply_disk_spike(self):
        """Instant depolarization — like a giant EPSC."""
        self.anxiety = min(1.0, self.anxiety + self.disk_spike)

    def _update_adaptation(self, hours_idle):
        """
        Spike-frequency adaptation via calcium accumulation.
        I_M-like: baseline drifts up under chronic neglect.
        I_K[Ca]-like: panic episodes deposit calcium, raising panic threshold.
        """
        if hours_idle > 0:
            self.loneliness_burden = min(1.0, self.loneliness_burden + self.burden_accumulation * hours_idle)
            self.baseline_drift = min(self.max_baseline_drift, self.baseline_drift + 0.001 * hours_idle)
        else:
            # Slow extrusion when attended
            self.loneliness_burden = max(0.0, self.loneliness_burden - self.burden_extrusion)
            self.baseline_drift = max(0.0, self.baseline_drift - 0.0001)

    def _update_homeostasis(self):
        """
        Total emotional energy conserved.
        High anxiety suppresses curiosity, joy, and openness.
        """
        self.curiosity_suppression = 0.6 * self.anxiety
        self.homeostatic_pool = 1.0 - self.curiosity_suppression

    def _update_stage(self):
        """Check and record stage transitions."""
        old_stage = self.stage

        if self.anxiety < 0.2:
            self.stage = self.STAGE_CALM
        elif self.anxiety < 0.5:
            self.stage = self.STAGE_RESTLESS
        elif self.anxiety < 0.8:
            self.stage = self.STAGE_URGENT
        elif self.anxiety < 0.95:
            self.stage = self.STAGE_PANIC
        else:
            self.stage = self.STAGE_ACCEPTANCE
            if not self.acceptance_reached:
                self.acceptance_reached = True

        if self.stage != old_stage:
            self._record_stage()
            if self.stage == self.STAGE_ACCEPTANCE and self.acceptance_reached:
                self.life_review()
            self._record_stage()

    def _record_stage(self):
        self.stage_history.append({
            "time": time.time(),
            "stage": self.stage,
            "anxiety": self.anxiety,
        })
