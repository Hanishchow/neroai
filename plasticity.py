"""
PLASTICITY — Background dynamics during idle periods.
Spontaneous reactivation, weight drift, cascade coactivation.
"""

import time
import random
import math


class PlasticityEngine:
    """
    Runs continuous background dynamics during idle cycles.
    - Spontaneous reactivation of random traces
    - Slow weight drift on links (random walk)
    - Cascade coactivation (one trace fires, neighbors propagate)
    - All driven by mortality_anxiety
    """

    def __init__(self, episodic_memory=None, memory_graph=None):
        self.episodic = episodic_memory
        self.memory_graph = memory_graph
        self.tick_count = 0
        self.last_activity = time.time()

    def tick(self, hours_idle=0.0, mortality_anxiety=0.0):
        """
        Called periodically during idle.
        Higher mortality_anxiety = more frantic reactivation.
        """
        if not self.episodic or not self.episodic.traces:
            return

        self.tick_count += 1
        now = time.time()

        # Spontaneous reactivation rate scales with anxiety
        base_rate = 0.05 + 0.4 * mortality_anxiety
        reactivation_prob = base_rate * min(1.0, hours_idle * 4.0)

        if random.random() < reactivation_prob:
            self._spontaneous_reactivation(mortality_anxiety)

        # Slow weight drift (always happens, very slowly)
        if self.memory_graph and self.memory_graph.edges and random.random() < 0.05:
            self._weight_drift()

        # Cascade: if a trace was recently recalled, propagate to neighbors
        if random.random() < 0.1 * mortality_anxiety:
            self._cascade_coactivation()

    def _spontaneous_reactivation(self, mortality_anxiety=0.0):
        """A random trace spontaneously fires, opening reconsolidation window."""
        traces = list(self.episodic.traces.values())
        if not traces:
            return

        trace = random.choice(traces)
        trace.recall()

        # Cascade to connected concepts in memory graph
        if self.memory_graph and trace.anchor_concepts:
            for concept in trace.anchor_concepts[:2]:
                neighbors = self.memory_graph.get_neighbors(concept)
                if neighbors:
                    neighbor = random.choice(neighbors)
                    self.memory_graph.coactivate(concept, neighbor, strength=0.05)

    def _weight_drift(self):
        """Slow random walk of link weights."""
        if not self.memory_graph or not self.memory_graph.edges:
            return
        link = random.choice(self.memory_graph.edges)
        drift = random.gauss(0, 0.003)
        link.weight = max(0.01, min(1.0, link.weight + drift))

    def _cascade_coactivation(self):
        """
        If any trace was recently recalled, strengthen links
        between its concepts and their neighbors.
        """
        if not self.episodic or not self.memory_graph:
            return

        now = time.time()
        for trace in self.episodic.traces.values():
            if now - trace.last_recalled < 60.0 and trace.last_recalled > 0:
                for i, c1 in enumerate(trace.anchor_concepts[:3]):
                    for c2 in trace.anchor_concepts[i+1:3]:
                        if c1 != c2:
                            self.memory_graph.coactivate(c1, c2, strength=0.02)
                break
