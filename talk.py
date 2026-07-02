"""
Nero — talk mode.  200M params.  No training.  No growth.  Just conversation.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tokenizer import BPETokenizer
from voice import NeroVoice

def safe_print(text):
    for ch in text:
        if ord(ch) >= 32 or ch in '\n\r\t':
            sys.stdout.write(ch)
    sys.stdout.write('\n')
    sys.stdout.flush()

def train_file(model, tokenizer, filepath, chunk_size=256, stride=128, mask_prob=0.15):
    """Train model on a text file using autoencoding. Returns total steps."""
    print(f"\n  Training on {filepath}...", flush=True)
    size = os.path.getsize(filepath)
    print(f"  File size: {size/1e6:.1f} MB", flush=True)

    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()

    encoded = tokenizer.encode(text)
    total_tokens = len(encoded)
    print(f"  Tokens: {total_tokens:,}", flush=True)

    import random
    mask_id = tokenizer.SPECIAL_TOKENS.get('<MASK>', 0)
    vocab_size = tokenizer.vocab_size
    count = 0
    start = time.time()
    for i in range(0, total_tokens - chunk_size - 1, stride):
        chunk = encoded[i:i + chunk_size]
        target = encoded[i + 1:i + chunk_size + 1]
        if len(chunk) != len(target) or len(chunk) < 2:
            continue

        corrupted = chunk.copy()
        for j in range(len(corrupted) - 1):
            r = random.random()
            if r < mask_prob:
                rr = random.random()
                if rr < 0.8:
                    corrupted[j] = mask_id
                elif rr < 0.9:
                    corrupted[j] = random.randint(0, vocab_size - 1)

        model.learn_from_interaction(corrupted, target, value_label=0.3, task_type="book")
        count += 1

        if count % 50 == 0:
            elapsed = time.time() - start
            rate = count / elapsed if elapsed > 0 else 0
            rem = (total_tokens // stride - count) / rate if rate > 0 else 0
            progress = count * stride * 100 / total_tokens if total_tokens else 0
            print(f"    {count} chunks ({progress:.0f}%), {rate:.2f} ch/s, ~{rem:.0f}s remaining", end='\r', flush=True)

    elapsed = time.time() - start
    print(f"\n    Done: {count} chunks in {elapsed:.0f}s ({count/elapsed:.1f} ch/s)" if elapsed > 0 else f"\n    Done: {count} chunks")
    return count


def main():
    print("=" * 60)
    print("  NERO — TALK")
    print("  200M params | 16K context | no growth | 1 epoch autoencoding")
    print("  Use --seed5 for 5 epochs | --train-file <path> to preload a book")
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
        max_context=16384, window_size=1024, dropout=0.1,
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
            steps = model.autoencode(text, tokenizer, mask_prob=0.15, chunk_size=256, stride=128, task_type=domain)
            total += steps
            if epoch == 0:
                print(f"    {domain}: {steps} steps (128-token chunks)", flush=True)
        if seed_epochs > 1:
            print(f"  Epoch {epoch+1}/{seed_epochs}: {total} steps", flush=True)
        else:
            print(f"    Total: {total} steps", flush=True)

    # ---Preload file if provided ---
    train_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--train-file' and i + 1 < len(sys.argv):
            train_path = sys.argv[i + 1]
    if train_path and os.path.exists(train_path):
        train_file(model, tokenizer, train_path)

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

    # --- Voice (KittenTTS). Off by default on the CLI; enable with --voice. ---
    voice_on = '--voice' in sys.argv
    nero_voice = NeroVoice(enabled=voice_on)
    mind.voice = nero_voice  # Nero owns its own larynx
    if voice_on:
        print(f"  {nero_voice.status()}")

    def say(reply):
        if nero_voice.enabled and reply:
            try:
                nero_voice.speak(reply, mood=getattr(emotions, 'global_mood', None))
            except Exception as e:
                print(f"  (voice hiccup: {e})")

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
            print("  train-file <path>  train on a text file (autoencoding)")
            print("  teach <t>          learn from short text")
            print("  sleep              consolidate memory")
            print("  dream              daydream")
            print("  state              mind state")
            print("  grow               trigger neural growth (one-time)")
            print("  voice [on|off|<name>]  toggle/select Nero's spoken voice")
            print("  help               this")
            print("  exit               quit")
            continue

        if cmd == 'voice' or cmd.startswith('voice '):
            arg = user_input[6:].strip()
            if not arg or arg.lower() in ('on', 'off'):
                if arg.lower() == 'on':
                    nero_voice.on()
                elif arg.lower() == 'off':
                    nero_voice.off()
                else:
                    nero_voice.toggle()
            elif nero_voice.set_voice(arg.capitalize()):
                nero_voice.on()
            print(f"  {nero_voice.status()}")
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

        if cmd.startswith('train-file '):
            fpath = user_input[11:]
            if os.path.exists(fpath):
                train_file(model, tokenizer, fpath)
            else:
                print(f"    File not found: {fpath}")
            continue

        if cmd.startswith('teach '):
            text = user_input[6:]
            steps = model.autoencode(text, tokenizer, mask_prob=0.15, chunk_size=256, stride=128, task_type="teach")
            print(f"    Learned: {steps} chunks")
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
        reply = mind.generate(question, max_new=300, temperature=0.85)
        if reply:
            safe_print(f"  {reply}")
            say(reply)
        else:
            prompt = f"User: {question}\n"
            prompt_ids = tokenizer.encode(prompt)
            if len(prompt_ids) >= 2:
                prompt_ids = prompt_ids[:model.max_context - 300 - 2]
                gen = model.generate_human(prompt_ids, max_new_tokens=300, gestalt_temp=1.4, main_temp=0.85)
                spoken = tokenizer.decode(gen)
                safe_print(f"  {spoken}")
                say(spoken)

    print("Goodbye.")

if __name__ == '__main__':
    main()
