"""
Nero — FULL GPU TRAINING.  200M.  5 seed epochs.  Train on any book.  Then chat.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import random
from tokenizer import BPETokenizer

def safe_print(text):
    for ch in text:
        if ord(ch) >= 32 or ch in '\n\r\t':
            sys.stdout.write(ch)
    sys.stdout.write('\n')
    sys.stdout.flush()

def train_on_file(model, tokenizer, filepath, chunk_size=256, stride=128, mask_prob=0.15, epochs=1):
    """Train model on a text file using denoising autoencoding. Returns total steps."""
    print(f"\n{'='*60}")
    print(f"  TRAINING ON: {os.path.basename(filepath)}")
    print(f"  {epochs} epoch{'s' if epochs > 1 else ''} | {chunk_size}-token chunks | stride {stride}")
    print(f"{'='*60}")
    
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    encoded = tokenizer.encode(text)
    total_tokens = len(encoded)
    print(f"  File size: {os.path.getsize(filepath)/1e6:.1f} MB | Tokens: {total_tokens:,}")
    
    mask_id = tokenizer.SPECIAL_TOKENS.get('<MASK>', 0)
    vocab_size = tokenizer.vocab_size
    total_chunks = 0
    grand_start = time.time()
    
    for epoch in range(epochs):
        count = 0
        epoch_start = time.time()
        step_times = []
        
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
            
            t0 = time.time()
            model.learn_from_interaction(corrupted, target, value_label=0.3, task_type="book")
            step_times.append(time.time() - t0)
            count += 1
            total_chunks += 1
            
            if count % 100 == 0:
                avg = sum(step_times[-100:]) / max(len(step_times[-100:]), 1)
                rem_chunks = ((total_tokens - chunk_size - 1) // stride) - count
                eta = rem_chunks * avg
                pct = count * stride * 100 / max(total_tokens, 1)
                print(f"  Epoch {epoch+1}/{epochs}: {count} chunks ({pct:.0f}%) | {avg*1000:.0f}ms/chunk | ETA: {eta/60:.1f}min", end='\r', flush=True)
        
        elapsed = time.time() - epoch_start
        print(f"\n  Epoch {epoch+1}/{epochs} done: {count} chunks in {elapsed:.0f}s ({count/elapsed:.1f} ch/s)" if elapsed > 0 else "")
    
    total_elapsed = time.time() - grand_start
    print(f"  TOTAL: {total_chunks} chunks in {total_elapsed:.0f}s")
    return total_chunks


def main():
    # Parse args
    filepath = None
    epochs = 3
    save_path = None
    load_path = None
    for i, arg in enumerate(sys.argv):
        if arg == '--train-file' and i + 1 < len(sys.argv):
            filepath = sys.argv[i + 1]
        if arg.startswith('--epochs') and i + 1 < len(sys.argv):
            try:
                epochs = int(sys.argv[i + 1])
            except: pass
        if arg == '--save' and i + 1 < len(sys.argv):
            save_path = sys.argv[i + 1]
        if arg == '--load' and i + 1 < len(sys.argv):
            load_path = sys.argv[i + 1]
    
    print("=" * 60)
    print("  NERO — FULL GPU TRAINING")
    print("  200M params | 16K context | 5 seed epochs + book training")
    print("=" * 60)
    
    # Check GPU compatibility
    import torch
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device='cuda')
            del t
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        except:
            print("  GPU incompatible with current PyTorch. Falling back to CPU.")
            import os as _os
            _os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            # Force DEVICE to cpu by re-importing
            import importlib
            import biologic_v2
            importlib.reload(biologic_v2)
    
    # --- Tokenizer ---
    print("\n[1] Loading BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=4096)
    tpath = "bpe_vocab.json"
    if os.path.exists(tpath):
        tokenizer.load(tpath)
        print(f"    Vocab: {tokenizer.get_vocab_size()} tokens")
    else:
        print("    No tokenizer found. Run interactive_v2.py first.")
        return
    
    # --- 200M model ---
    print("[2] Creating 200M model on GPU...")
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
    print(f"    {p:,} params ({p/1e6:.0f}M) on {DEVICE}")
    
    # --- 5-epoch seed learning ---
    print(f"\n[3] Seed learning (5 epochs)...")
    mask_id = tokenizer.SPECIAL_TOKENS.get('<MASK>', 0)
    vocab_size = tokenizer.vocab_size
    for epoch in range(5):
        total = 0
        for domain, text in SEED_TEXTS.items():
            encoded = tokenizer.encode(text)
            if len(encoded) < 4:
                continue
            for i in range(0, len(encoded) - 256 - 1, 128):
                chunk = encoded[i:i+256]
                target = encoded[i+1:i+257]
                if len(chunk) != len(target) or len(chunk) < 2:
                    continue
                corrupted = chunk.copy()
                for j in range(len(corrupted) - 1):
                    if random.random() < 0.15:
                        corrupted[j] = mask_id if random.random() < 0.8 else random.randint(0, vocab_size - 1)
                model.learn_from_interaction(corrupted, target, value_label=0.3, task_type=domain)
                total += 1
            if epoch == 0:
                print(f"    {domain}: complete", flush=True)
        print(f"  Epoch {epoch+1}/5: {total} steps", flush=True)
    print("  Seed learning complete.")
    
    # --- Load checkpoint if provided ---
    if load_path and os.path.exists(load_path):
        import torch
        print(f"\n[4] Loading checkpoint: {load_path}...")
        sd = torch.load(load_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(sd)
        print(f"    Loaded {sum(p.numel() for p in model.parameters()):,} params")
    
    # --- Book training ---
    if filepath and os.path.exists(filepath):
        train_on_file(model, tokenizer, filepath, epochs=epochs)
    
    # --- Save checkpoint if requested ---
    if save_path:
        import torch
        print(f"\n  Saving checkpoint to {save_path}...")
        torch.save(model.state_dict(), save_path)
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"    Saved ({size_mb:.0f} MB)")
    
    print(f"\n{'='*60}")
    print(f"  Training complete. Entering interactive mode.")
    print(f"{'='*60}")
    
    # --- Subsystems ---
    print("\n[5] Loading subsystems...")
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
    memory_graph = MemoryGraph(filepath="memory_graph_full.json")
    plasticity = PlasticityEngine(episodic, memory_graph)
    developmental = DevelopmentalSystem()
    social = SocialEmotionSystem()
    curiosity = CuriositySystem()
    narrative = NarrativeSelf()
    mind = Mind(model, tokenizer, episodic_memory=episodic, memory_graph=memory_graph,
                emotions=emotions, mortality=mortality, curiosity=curiosity,
                narrative=narrative, developmental=developmental, social_emotion=social)
    mind.load_state()
    
    print("\nReady. Commands: ask | teach | train-file | sleep | state | help | exit")
    last_tick = time.time()
    
    while True:
        try:
            user_input = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSaving...")
            mind.save_state()
            break
        if not user_input:
            continue
        cmd = user_input.lower()
        
        if cmd in ('exit', 'quit'):
            mind.save_state()
            break
        
        if cmd == 'help':
            print("  ask <q>         generate response")
            print("  teach <t>       learn from text")
            print("  train-file <p>  train on a text file")
            print("  save <path>     save model checkpoint")
            print("  load <path>     load model checkpoint")
            print("  sleep           consolidate")
            print("  state           mind state")
            print("  help / exit")
            continue
        
        if cmd == 'state':
            print(f"  Body: fatigue={mind.body.fatigue:.2f}, heart={mind.body.heart_rate:.0f}bpm")
            print(f"  Sleep: {mind.sleep_pressure.pressure:.2f}")
            print(f"  ToM: mood={mind.theory_of_mind.user_mood}, intent={mind.theory_of_mind.user_intent}")
            print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")
            continue
        
        if cmd.startswith('train-file '):
            fp = user_input[11:]
            if os.path.exists(fp):
                train_on_file(model, tokenizer, fp, epochs=epochs)
            else:
                print(f"  Not found: {fp}")
            continue
        
        if cmd.startswith('teach '):
            text = user_input[6:]
            encoded = tokenizer.encode(text)
            if len(encoded) > 3:
                for i in range(0, len(encoded) - 256, 128):
                    chunk = encoded[i:i+256]
                    target = encoded[i+1:i+257]
                    if len(chunk) == len(target):
                        corr = chunk.copy()
                        for j in range(len(corr)):
                            if random.random() < 0.15:
                                corr[j] = mask_id if random.random() < 0.8 else random.randint(0, vocab_size - 1)
                        model.learn_from_interaction(corr, target, value_label=0.5, task_type="teach")
                print(f"  Learned")
            continue
        
        if cmd.startswith('save '):
            spath = user_input[5:]
            import torch
            print(f"  Saving to {spath}...")
            torch.save(model.state_dict(), spath)
            print(f"    Done ({os.path.getsize(spath)/1e6:.0f} MB)")
            continue

        if cmd.startswith('load '):
            lpath = user_input[5:]
            if os.path.exists(lpath):
                import torch
                print(f"  Loading from {lpath}...")
                sd = torch.load(lpath, map_location=DEVICE, weights_only=True)
                model.load_state_dict(sd)
                print(f"    Loaded {sum(p.numel() for p in model.parameters()):,} params")
            else:
                print(f"  Not found: {lpath}")
            continue

        if cmd == 'sleep':
            model.consolidate_memory()
            print("  Consolidated")
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
        else:
            from biologic_v2 import generate_with_gestalt
            prompt = f"User: {question}\n"
            ids = tokenizer.encode(prompt)
            if len(ids) >= 2:
                ids = ids[:model.max_context - 300 - 2]
                gen = generate_with_gestalt(model, tokenizer, ids, max_new_tokens=300, gestalt_temp=1.4, main_temp=0.85)
                safe_print(f"  {tokenizer.decode(gen)}")
    
    print("Goodbye.")

if __name__ == '__main__':
    main()
