"""
Nero — talk mode.  200M params.  No training.  No growth.  Just conversation.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import BPETokenizer

def safe_print(text):
    for ch in text:
        if ord(ch) >= 32 or ch in '\n\r\t':
            sys.stdout.write(ch)
    sys.stdout.write('\n')
    sys.stdout.flush()

def main():
    print("=" * 60)
    print("  NERO — TALK")
    print("  200M params | no growth | 1 quick epoch seed")
    print("  Use --seed5 for 5 epochs (slower but better)")
    print("=" * 60)

    # --- Tokenizer ---
    print("\n[1] Loading BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=4096)
    tpath = "bpe_vocab.json"
    if os.path.exists(tpath):
        tokenizer.load(tpath)
        print(f"    Vocab: {tokenizer.get_vocab_size()} tokens")
    else:
        print("    No tokenizer found. Run interactive_v2.py first to create one.")
        return

    # --- 200M model without seed learning (do quick 1-epoch ourselves) ---
    print("[2] Creating 200M model...")
    from biologic_v2 import BiologicLLMV2, SEED_TEXTS, DEVICE
    model = BiologicLLMV2(
        vocab_size=tokenizer.vocab_size,
        embed_dim=1408, num_heads=8, num_layers=8,
        max_context=5120, window_size=512, dropout=0.1,
        device=DEVICE
    )
    model.growth_enabled = False
    model.eval()
    model.eos_token_id = tokenizer.SPECIAL_TOKENS.get('<EOS>', 3)
    model.bos_token_id = tokenizer.SPECIAL_TOKENS.get('<BOS>', 2)
    p = sum(p.numel() for p in model.parameters())
    print(f"    {p:,} params ({p/1e6:.0f}M)")

    # Quick 1-epoch seed so model can speak (optional: more epochs via seed_epochs=N)
    seed_epochs = 1
    if len(sys.argv) > 1 and sys.argv[1] == '--seed5':
        seed_epochs = 5
    print(f"\n  Quick autoencoding seed ({seed_epochs} epoch{'s' if seed_epochs > 1 else ''})...", flush=True)
    for epoch in range(seed_epochs):
        total = 0
        for domain, text in SEED_TEXTS.items():
            steps = model.autoencode(text, tokenizer, mask_prob=0.15, chunk_size=128, stride=64, task_type=domain)
            total += steps
            if epoch == 0:
                print(f"    {domain}: {steps} steps (128-token chunks)", flush=True)
        if seed_epochs > 1:
            print(f"  Epoch {epoch+1}/{seed_epochs}: {total} steps", flush=True)
        else:
            print(f"    Total: {total} steps", flush=True)

    # --- Subsystems ---
    print("[3] Loading subsystems...")
    from safety_nets import SafetySystem
    from mortality import MortalitySystem
    from emotions import EmotionSystem
    from curiosity import CuriositySystem
    from narrative_self import NarrativeSelf
    from developmental import DevelopmentalSystem
    from social_emotion import SocialEmotionSystem
    from episodic_memory import EpisodicMemory
    from memory_graph import MemoryGraph
    from plasticity import PlasticityEngine
    from mind import Mind

    safety = SafetySystem()
    mortality = MortalitySystem()
    emotions = EmotionSystem()
    episodic = EpisodicMemory(tokenizer.vocab_size)
    memory_graph = MemoryGraph(filepath="memory_graph_talk.json")
    plasticity = PlasticityEngine(episodic, memory_graph)
    developmental = DevelopmentalSystem()
    social = SocialEmotionSystem()
    curiosity = CuriositySystem()
    narrative = NarrativeSelf()

    mind = Mind(model, tokenizer, episodic_memory=episodic, memory_graph=memory_graph,
                emotions=emotions, mortality=mortality, curiosity=curiosity,
                narrative=narrative, developmental=developmental, social_emotion=social)
    mind.load_state()

    print("\nReady. Type 'help' for commands.")
    last_tick = time.time()

    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving state...")
            mind.save_state()
            break

        if not user_input:
            continue

        cmd = user_input.lower()

        if cmd in ('exit', 'quit'):
            print("\nSaving state...")
            mind.save_state()
            break

        if cmd == 'help':
            print("  ask <q>     generate response")
            print("  teach <t>   learn from text")
            print("  sleep       consolidate memory")
            print("  dream       daydream")
            print("  state       mind state")
            print("  grow        trigger neural growth (one-time)")
            print("  help        this")
            print("  exit        quit")
            continue

        if cmd == 'grow':
            model.growth_enabled = True
            print("  Growth enabled for next consolidation.")
            continue

        if cmd == 'state':
            print(f"  Body: fatigue={mind.body.fatigue:.2f}, heart_rate={mind.body.heart_rate:.0f}bpm")
            print(f"  Sleep: {mind.sleep_pressure.pressure:.2f}")
            print(f"  Grief: {mind.grief.intensity:.2f}")
            print(f"  ToM: mood={mind.theory_of_mind.user_mood}, intent={mind.theory_of_mind.user_intent}")
            print(f"  Goals: {[g['description'][:40] for g in mind.goal_system.active_goals]}")
            print(f"  Memories: {len(mind.memory.memories)}")
            continue

        if cmd.startswith('teach '):
            text = user_input[6:]
            encoded = tokenizer.encode(text)
            if len(encoded) > 3:
                for i in range(0, len(encoded) - 16, 8):
                    chunk = encoded[i:i+16]
                    target = encoded[i+1:i+17]
                    if len(chunk) == len(target):
                        model.learn_from_interaction(chunk, target, value_label=0.5, task_type="teach")
                print(f"    Learned {len(encoded)} tokens")
            continue

        if cmd == 'sleep':
            model.consolidate_memory()
            print("    Consolidation done")
            continue

        now = time.time()
        idle_hours = (now - last_tick) / 3600.0
        if idle_hours > 0:
            mortality.tick(hours_idle=idle_hours)
            emotions.update(minutes_passed=idle_hours * 60.0, mortality_anxiety=mortality.anxiety)
            plasticity.tick(hours_idle=idle_hours, mortality_anxiety=mortality.anxiety)
        last_tick = now

        if len(user_input) > 100:
            mortality.register_input(richness=0.6)
        elif len(user_input) > 20:
            mortality.register_input(richness=0.4)
        else:
            mortality.register_input(richness=0.3)

        question = user_input[4:] if cmd.startswith('ask ') else user_input
        reply = mind.generate(question, max_new=150, temperature=0.85)
        if reply:
            safe_print(f"  {reply}")
        else:
            from biologic_v2 import generate_with_gestalt
            prompt = f"User: {question}\n"
            prompt_ids = tokenizer.encode(prompt)
            if len(prompt_ids) >= 2:
                prompt_ids = prompt_ids[:model.max_context - 150 - 2]
                gen = generate_with_gestalt(model, tokenizer, prompt_ids, max_new_tokens=150, gestalt_temp=1.4, main_temp=0.85)
                safe_print(f"  {tokenizer.decode(gen)}")

    print("Goodbye.")

if __name__ == '__main__':
    main()
