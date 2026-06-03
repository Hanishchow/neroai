"""
Nero — FULL GPU TRAINING with max CPU/GPU utilization.
- Mixed precision (AMP fp16)
- Async data pre-fetch
- Multi-CPU tokenization
- Large batch sizes on T4
"""
import sys, os, time, random, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from tokenizer import BPETokenizer

torch.backends.cudnn.benchmark = True
torch.set_num_threads(os.cpu_count())

def safe_print(text):
    for ch in text:
        if ord(ch) >= 32 or ch in '\n\r\t':
            sys.stdout.write(ch)
    sys.stdout.write('\n')
    sys.stdout.flush()

def prepare_all_chunks(encoded, chunk_size=1024, stride=512):
    """Pre-compute chunk arrays on CPU (single pass, fast with numpy)."""
    n = len(encoded)
    chunks = []
    for i in range(0, n - chunk_size - 1, stride):
        inp = np.array(encoded[i:i + chunk_size], dtype=np.int64)
        tgt = np.array(encoded[i + 1:i + chunk_size + 1], dtype=np.int64)
        if len(inp) == len(tgt):
            chunks.append((inp, tgt))
    return chunks


def train_batched(model, tokenizer, filepath, chunk_size=1024, stride=512, epochs=5, batch_size=32):
    """Full-throttle GPU training with async CPU pre-fetch + AMP."""
    print(f"\n{'='*60}")
    print(f"  GPU TRAINING: {os.path.basename(filepath)}")
    print(f"  {epochs} epochs | {chunk_size}-token chunks | batch_size={batch_size} | AMP=on | FlashAttn")
    print(f"{'='*60}")
    
    # Read and tokenize (CPU, single-threaded BPE)
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        text = f.read()
    fsize_mb = os.path.getsize(filepath) / 1e6
    
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        # Clear cached attention masks from previous runs
        for block in model.blocks:
            if hasattr(block.attention, '_mask_cache'):
                block.attention._mask_cache.clear()
            if hasattr(block.attention, '_rel_cache'):
                block.attention._rel_cache.clear()
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU: {torch.cuda.get_device_name(0)} | {(total-free)/1e9:.1f}/{total/1e9:.1f} GiB used")
    
    print(f"  File: {fsize_mb:.1f} MB | Encoding...", end=' ', flush=True)
    t0 = time.time()
    encoded = tokenizer.encode(text)
    t1 = time.time()
    print(f"Tokens: {len(encoded):,} ({t1-t0:.1f}s)")
    
    # Pre-compute chunk indices on CPU
    print(f"  Pre-computing chunks...", end=' ', flush=True)
    chunks = prepare_all_chunks(encoded, chunk_size, stride)
    n_chunks = len(chunks)
    print(f"Chunks: {n_chunks:,}")
    del text, encoded  # free memory
    
    # Setup model for training
    model.train()
    if getattr(model, '_optimizer', None) is None:
        model._create_optimizer()
    optimizer = model._optimizer
    device = model.device
    
    # Mixed precision
    scaler = torch.amp.GradScaler(device=device.type)
    
    total_processed = 0
    grand_start = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        indices = list(range(n_chunks))
        random.shuffle(indices)
        epoch_loss = 0.0
        n_batches = 0
        
        for bidx in range(0, n_chunks, batch_size):
            batch_idx = indices[bidx:bidx + batch_size]
            B = len(batch_idx)
            
            # Build batch tensors (async to GPU)
            inp_np = np.stack([chunks[i][0] for i in batch_idx])
            tgt_np = np.stack([chunks[i][1] for i in batch_idx])
            inp = torch.from_numpy(inp_np).long().to(device, non_blocking=True)
            tgt = torch.from_numpy(tgt_np).long().to(device, non_blocking=True)
            
            # Forward with AMP
            with torch.amp.autocast(device.type):
                logits, loss, _ = model(inp, targets=tgt, return_value=False)
            
            if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                if not (torch.isnan(grad_norm) or torch.isinf(grad_norm) or grad_norm < 1e-8):
                    scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss.item()
                total_processed += B
                n_batches += 1
            
            # Progress
            if (bidx // batch_size) % 25 == 0 and bidx > 0:
                pct = bidx * 100 / n_chunks
                elapsed = time.time() - epoch_start
                rate = n_batches / elapsed if elapsed > 0 else 0
                rem = (n_chunks - bidx) / (bidx / elapsed) if bidx > 0 else 0
                avg_loss = epoch_loss / max(n_batches, 1)
                gpu_mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == 'cuda' else 0
                print(f"  E{epoch+1}: {bidx}/{n_chunks} ({pct:.0f}%) | loss={avg_loss:.4f} | {rate:.0f}ch/s | ETA {rem:.0f}s | VRAM={gpu_mem:.1f}GB", flush=True)
        
        elapsed = time.time() - epoch_start
        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"  Epoch {epoch+1}/{epochs} done: {n_chunks} chunks in {elapsed:.0f}s | avg_loss={avg_loss:.4f}")
    
    total_elapsed = time.time() - grand_start
    avg_loss = total_processed / max(total_elapsed, 1)
    print(f"  TOTAL: {total_processed:,} tokens in {total_elapsed:.0f}s ({total_processed/total_elapsed:.0f} tok/s)")
    
    # Sanity check
    model.eval()
    print("\n  Sanity check — generating with prompt 'User: Hello'...")
    test_ids = tokenizer.encode("User: Hello\n")
    if len(test_ids) >= 2:
        test_ids = test_ids[:model.max_context - 50 - 2]
        gen = model.generate_human(test_ids, max_new_tokens=50, gestalt_temp=1.0, main_temp=0.5)
        sample = tokenizer.decode(gen)[:200]
        print(f"  Output: {sample}")
        readable = all(c in ' \n\r\t' or 32 <= ord(c) <= 126 for c in sample[:50])
        print(f"  {'OK: printable' if readable else 'WARN: non-printable chars'}")
    model.train()
    
    return n_chunks * epochs


def main():
    filepath = None
    epochs = 3
    save_path = None
    load_path = None
    small = False
    for i, arg in enumerate(sys.argv):
        if arg == '--train-file' and i + 1 < len(sys.argv):
            filepath = sys.argv[i + 1]
        if arg.startswith('--epochs') and i + 1 < len(sys.argv):
            try: epochs = int(sys.argv[i + 1])
            except: pass
        if arg == '--save' and i + 1 < len(sys.argv):
            save_path = sys.argv[i + 1]
        if arg == '--load' and i + 1 < len(sys.argv):
            load_path = sys.argv[i + 1]
        if arg == '--small':
            small = True
    
    embed_dim = 688 if small else 1408
    num_heads = 8
    num_layers = 8
    model_size = "50M" if small else "200M"
    
    print("=" * 60)
    print(f"  NERO — FULL GPU TRAINING")
    print(f"  {model_size} params | 16K context | {filepath or 'corpus.txt'} | {epochs} epochs")
    print("=" * 60)
    
    if torch.cuda.is_available():
        try:
            t = torch.zeros(1, device='cuda')
            del t
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        except:
            print("  GPU incompatible. Falling back to CPU.")
            import os as _os
            _os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            import importlib, biologic_v2
            importlib.reload(biologic_v2)
    
    print("\n[1] Loading BPE tokenizer...")
    tokenizer = BPETokenizer(vocab_size=4096)
    tpath = "bpe_vocab.json"
    if os.path.exists(tpath):
        tokenizer.load(tpath)
        print(f"    Vocab: {tokenizer.get_vocab_size()} tokens")
    else:
        print("    No tokenizer found. Run interactive_v2.py first.")
        return
    
    print(f"[2] Creating {model_size} model on GPU...")
    from biologic_v2 import BiologicLLMV2, SEED_TEXTS, DEVICE
    model = BiologicLLMV2(
        vocab_size=tokenizer.vocab_size,
        embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers,
        max_context=16384, window_size=1024, dropout=0.1,
        device=DEVICE
    )
    model.growth_enabled = False
    model.hebbian_enabled = False
    model.eval()
    model.eos_token_id = tokenizer.SPECIAL_TOKENS.get('<EOS>', 3)
    model.bos_token_id = tokenizer.SPECIAL_TOKENS.get('<BOS>', 2)
    p = sum(p.numel() for p in model.parameters())
    print(f"    {p:,} params ({p/1e6:.0f}M) on {DEVICE}")
    
    # Brief seed (direct fwd/bwd, no Hebbian)
    print(f"\n[3] Brief seed (1 epoch, direct fwd/bwd)...")
    seed_steps = 0
    model.train()
    if model._optimizer is None:
        model._create_optimizer()
    scaler = torch.amp.GradScaler(device=DEVICE.type)
    for domain, text in SEED_TEXTS.items():
        encoded = tokenizer.encode(text)
        if len(encoded) < 4:
            continue
        for i in range(0, len(encoded) - 256 - 1, 128):
            chunk = encoded[i:i+256]
            target = encoded[i+1:i+257]
            if len(chunk) != len(target) or len(chunk) < 2:
                continue
            inp = torch.tensor([chunk], dtype=torch.long, device=DEVICE)
            tgt = torch.tensor([target], dtype=torch.long, device=DEVICE)
            with torch.amp.autocast(DEVICE.type):
                _, loss, _ = model(inp, targets=tgt, return_value=False)
            if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                model._optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(model._optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(model._optimizer)
                scaler.update()
                model.loss_history.append(loss.item())
                seed_steps += 1
        print(f"    {domain}: {len(encoded)} tokens", flush=True)
    last_loss = model.loss_history[-1] if model.loss_history else 0
    print(f"  Seed done: {seed_steps} steps, loss {last_loss:.4f}")
    
    # Free seed memory before main training
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
        free, total = torch.cuda.mem_get_info()
        print(f"  GPU memory: {(total-free)/1e9:.1f}/{total/1e9:.1f} GiB used")
    
    # Load checkpoint
    if load_path and os.path.exists(load_path):
        print(f"\n[4] Loading checkpoint: {load_path}...")
        sd = torch.load(load_path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(sd)
        print(f"    Loaded {sum(p.numel() for p in model.parameters()):,} params")
    
    # Build corpus if missing
    if not filepath:
        filepath = "corpus.txt"
        if not os.path.exists(filepath):
            print(f"\n  Building corpus from Project Gutenberg...")
            ret = os.system(f"{sys.executable} prepare_corpus.py --fast --min-books 2")
            if ret != 0:
                ret = os.system(f"{sys.executable} prepare_corpus.py --min-books 2")
            if ret != 0:
                print(f"  Could not build corpus. Use --train-file <path>")
                return
    
    # Book training
    if filepath and os.path.exists(filepath):
        train_batched(model, tokenizer, filepath, epochs=epochs)
    
    # Save checkpoint
    if save_path:
        print(f"\n  Saving checkpoint to {save_path}...")
        torch.save(model.state_dict(), save_path)
        size_mb = os.path.getsize(save_path) / 1e6
        print(f"    Saved ({size_mb:.0f} MB)")
    
    print(f"\n{'='*60}")
    print(f"  Training complete. Entering interactive mode.")
    print(f"{'='*60}")
    
    # Interactive mode
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
                train_batched(model, tokenizer, fp, epochs=epochs)
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
                        inp = torch.tensor([chunk], dtype=torch.long, device=DEVICE)
                        tgt = torch.tensor([target], dtype=torch.long, device=DEVICE)
                        with torch.amp.autocast(DEVICE.type):
                            _, loss, _ = model(inp, targets=tgt, return_value=False)
                        if loss is not None and not (torch.isnan(loss) or torch.isinf(loss)):
                            model._optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                            model._optimizer.step()
                print(f"  Learned")
            continue
        
        if cmd.startswith('save '):
            spath = user_input[5:]
            print(f"  Saving to {spath}...")
            torch.save(model.state_dict(), spath)
            print(f"    Done ({os.path.getsize(spath)/1e6:.0f} MB)")
            continue

        if cmd.startswith('load '):
            lpath = user_input[5:]
            if os.path.exists(lpath):
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
