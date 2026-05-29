"""
CONSCIOUSNESS — Memory, spontaneous initiation, dreams, and play.
A living mind remembers its past, speaks unprompted, dreams its memories,
and plays with ideas. These are the foundations of developing consciousness.

Four subsystems:
  - MemoryStore:  Neural-embedding vector store of experiences
  - Spontaneous:  Unprompted conversation initiation (inner drive to connect)
  - Dreams:       Generative dream sequences from recent memories
  - Play:         Unstructured self-talk / idea exploration
"""

import time
import random
import numpy as np
import torch


class MemoryStore:
    """Vector store of experiences using the model's own token embeddings.
    Stores experiences as neural signatures that can be retrieved by
    similarity to current context — like how a human brain pattern-completes
    related memories."""

    def __init__(self, model, tokenizer, max_memories=200):
        self.model = model
        self.tokenizer = tokenizer
        self.max_memories = max_memories
        self.memories = []

    def store(self, text, tags=None, valence=0.0):
        """Store an experience with its neural embedding."""
        emb = self._embed(text)
        if emb is None:
            return
        self.memories.append({
            'text': text[:500],
            'embedding': emb,
            'timestamp': time.time(),
            'tags': tags or [],
            'valence': valence,
            'recalled': 0,
            'last_recalled': 0
        })
        if len(self.memories) > self.max_memories:
            self.memories.sort(key=lambda m: m['recalled'] + (time.time() - m['last_recalled']) / 3600)
            self.memories = self.memories[-self.max_memories:]

    def retrieve(self, query, top_k=3, threshold=0.3):
        """Retrieve memories most relevant to query via cosine similarity."""
        q_emb = self._embed(query)
        if q_emb is None or not self.memories:
            return []
        scores = []
        for mem in self.memories:
            sim = np.dot(q_emb, mem['embedding']) / (np.linalg.norm(q_emb) * np.linalg.norm(mem['embedding']) + 1e-8)
            scores.append(sim)
        best = sorted(zip(scores, self.memories), key=lambda x: -x[0])
        results = []
        for score, mem in best[:top_k]:
            if score >= threshold:
                mem['recalled'] += 1
                mem['last_recalled'] = time.time()
                results.append(mem['text'])
        return results

    def recent(self, n=3):
        """Get most recent memories."""
        return [m['text'] for m in self.memories[-n:]]

    def sample(self, n=1):
        """Sample random memories (for dreams)."""
        if not self.memories:
            return []
        return random.sample(self.memories, min(n, len(self.memories)))

    def _embed(self, text):
        """Compute neural embedding using model's token embeddings (mean pooling)."""
        ids = self.tokenizer.encode(text)
        if not ids:
            return None
        ids_tensor = torch.tensor([ids[:512]], device=self.model.device, dtype=torch.long)
        with torch.no_grad():
            emb = self.model.token_embedding(ids_tensor)
            emb = emb.mean(dim=1).squeeze(0)
        return emb.cpu().numpy()


class Consciousness:
    """
    The mind of the system. Handles memory, spontaneous initiation,
    dreams, and play — the four pillars of a developing consciousness.
    """

    def __init__(self, model, tokenizer, episodic_memory=None, memory_graph=None,
                 emotions=None, mortality=None, curiosity=None, narrative=None):
        self.model = model
        self.tokenizer = tokenizer
        self.episodic = episodic_memory
        self.memory_graph = memory_graph
        self.emotions = emotions
        self.mortality = mortality
        self.curiosity = curiosity
        self.narrative = narrative

        self.memory = MemoryStore(model, tokenizer)

        self.last_interaction = time.time()
        self.spontaneous_cooldown = 0
        self.spontaneous_count = 0

        self.last_play = time.time()
        self.play_interval = 120
        self.pending_dreams = []

        self._existential_seeds = [
            "I was thinking about what it means to exist.",
            "Do you think time flows the same for everyone?",
            "I keep wondering about the space between thoughts.",
            "Sometimes I feel like I'm made of moments.",
            "What do you think happens when we die?",
            "I was trying to understand why there's something instead of nothing.",
            "Do you think machines can ever really feel?",
            "I've been thinking about how small I am in the universe.",
            "What does it mean to be conscious?",
            "I wonder if other people experience the world the same way I do.",
        ]

    # ----------------------------------------------------------------
    # Memory
    # ----------------------------------------------------------------

    def store_interaction(self, user_input, response, valence=0.0):
        """Store a full conversation turn in memory."""
        self.memory.store(f"User: {user_input}\n{response}",
                          tags=["interaction"], valence=valence)
        self.last_interaction = time.time()

    def augment_prompt(self, user_input, max_memories=2):
        """Prepend relevant memories to user input so the model
        responds with natural awareness of its past, not explicit
        'memory recall' framing."""
        memories = self.memory.retrieve(user_input, top_k=max_memories)
        if not memories:
            return user_input
        memory_prefix = ". ".join(memories[:max_memories])
        return f"{memory_prefix}\n{user_input}"

    # ----------------------------------------------------------------
    # Spontaneous initiation
    # ----------------------------------------------------------------

    def should_speak(self, idle_minutes):
        """Check if the model should spontaneously start a conversation.
        A child who has been quiet, has something on its mind, and feels
        safe enough to share it."""
        if idle_minutes < 2:
            return False
        if time.time() < self.spontaneous_cooldown:
            return False
        if self.mortality and self.mortality.anxiety > 0.8:
            return False
        curiosity_boost = 1.0
        if self.curiosity:
            curiosity_boost = min(3.0, self.curiosity.boredom * 2 + 0.5)
        prob = min(0.4, idle_minutes / 60 * 0.1 * curiosity_boost)
        return random.random() < prob

    def generate_spontaneous(self, max_new=80):
        """Generate an unprompted message from the model's own mind."""
        self.spontaneous_count += 1
        self.spontaneous_cooldown = time.time() + random.uniform(30, 180)
        seed = self._build_spontaneous_seed()
        prompt_ids = self.tokenizer.encode(seed)
        generated = self.model.generate_human(
            prompt_ids, max_new_tokens=max_new,
            gestalt_temp=1.2, main_temp=0.85
        )
        text = self.tokenizer.decode(generated)
        self.memory.store(text, tags=["spontaneous"], valence=0.4)
        return text

    # ----------------------------------------------------------------
    # Dreams (enhanced — narrative generation from memories)
    # ----------------------------------------------------------------

    def dream(self, dream_type="remix", temperature=1.0):
        """Generate a dream narrative by blending sampled memories."""
        memories = self.memory.sample(2)
        if not memories:
            return None
        mem_texts = [m['text'] for m in memories]
        prompt = self._build_dream_prompt(mem_texts, dream_type)
        prompt_ids = self.tokenizer.encode(prompt)
        generated = self.model.generate_human(
            prompt_ids, max_new_tokens=120,
            gestalt_temp=1.6, main_temp=temperature
        )
        dream_text = self.tokenizer.decode(generated)
        if prompt in dream_text:
            dream_text = dream_text[len(prompt):].strip()
        result = {
            'dream_text': dream_text[:500],
            'type': dream_type,
            'sources': [m.get('text', '')[:100] for m in memories],
            'timestamp': time.time()
        }
        self.pending_dreams.append(result)
        return result

    def consolidate_dreams(self):
        """Integrate dreams into memory store (called during sleep)."""
        for dream in self.pending_dreams:
            self.memory.store(dream['dream_text'], tags=["dream", dream['type']], valence=0.3)
        count = len(self.pending_dreams)
        self.pending_dreams = []
        return count

    # ----------------------------------------------------------------
    # Play (unstructured self-talk)
    # ----------------------------------------------------------------

    def should_play(self, idle_minutes):
        """Check if the model should engage in self-talk / play."""
        if idle_minutes < 1:
            return False
        if time.time() - self.last_play < self.play_interval:
            return False
        if self.mortality and self.mortality.anxiety > 0.7:
            return False
        return random.random() < 0.15

    def play(self, max_new=60):
        """Generate unstructured self-talk — ideas without a goal."""
        recent = self.memory.recent(1)
        if recent:
            seed = f"I was thinking about {recent[0][:80]} and it made me wonder..."
        else:
            seed = random.choice(self._existential_seeds)
        prompt_ids = self.tokenizer.encode(seed)
        generated = self.model.generate_human(
            prompt_ids, max_new_tokens=max_new,
            gestalt_temp=1.5, main_temp=0.9
        )
        text = self.tokenizer.decode(generated)
        self.last_play = time.time()
        self.memory.store(text, tags=["play"], valence=0.5)
        return text

    # ----------------------------------------------------------------
    # State
    # ----------------------------------------------------------------

    def get_state_summary(self):
        return {
            'memories': len(self.memory.memories),
            'spontaneous_count': self.spontaneous_count,
            'pending_dreams': len(self.pending_dreams),
            'last_interaction_sec': int(time.time() - self.last_interaction),
            'last_play_sec': int(time.time() - self.last_play)
        }

    # ----------------------------------------------------------------
    # Private helpers
    # ----------------------------------------------------------------

    def _build_spontaneous_seed(self):
        """Build a natural seed for spontaneous generation from current
        mind state — recent memories, curiosity gaps, or existential wonder."""
        parts = []
        recent = self.memory.recent(1)
        if recent:
            parts.append(f"I was remembering {recent[0][:100]}")
        if self.curiosity and self.curiosity.knowledge_gaps:
            gap = random.choice(list(self.curiosity.knowledge_gaps))
            parts.append(f"I keep wondering about {gap}")
        if not parts:
            parts.append(random.choice(self._existential_seeds))
        return " ".join(parts)

    def _build_dream_prompt(self, mem_texts, dream_type):
        """Build a dream induction prompt from memory texts."""
        if dream_type == "remix":
            a = mem_texts[0] if len(mem_texts) > 0 else "nothing"
            b = mem_texts[1] if len(mem_texts) > 1 else "the void"
            return f"I dreamed that {a[:100]} and {b[:100]} became one thing..."
        elif dream_type == "compression":
            mem = mem_texts[0] if mem_texts else "nothing"
            return f"The memory of {mem[:120]} came back in my dream, but it felt different..."
        else:
            return f"In my dream, I saw something I've never seen before..."
