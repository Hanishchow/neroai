"""
Download multiple classic books from Project Gutenberg and combine into a single training corpus.

Usage:
    python prepare_corpus.py [--output corpus.txt] [--min-books 3]
"""
import sys, os, re, time, urllib.request

BOOKS = [
    (1661, "Sherlock Holmes"),
    (1342, "Pride and Prejudice"),
    (2701, "Moby Dick"),
    (1400, "Great Expectations"),
    (76,   "Huckleberry Finn"),
    (84,   "Frankenstein"),
    (345,  "Dracula"),
    (2600, "War and Peace"),
    (98,   "A Tale of Two Cities"),
    (730,  "Oliver Twist"),
    (174,  "The Picture of Dorian Gray"),
    (43,   "Dr Jekyll and Mr Hyde"),
    (1663, "Around the World in 80 Days"),
    (2814, "The Republic"),
    (1232, "The Prince"),
    (1250, "The Scarlet Letter"),
    (5827, "The Art of War"),
]

def strip_gutenberg(text):
    """Remove Gutenberg header/footer boilerplate."""
    start = text.find("*** START OF")
    if start == -1:
        start = text.find("***START OF")
    if start >= 0:
        text = text[start:]
    end_markers = ["*** END OF", "***END OF", "End of Project Gutenberg"]
    for m in end_markers:
        pos = text.find(m)
        if pos >= 0:
            text = text[:pos]
            break
    return text

def clean_text(text):
    """Remove Gutenberg-specific cruft while preserving book content."""
    # Remove _underscores_ used for italics
    text = re.sub(r'_([^_]+)_', r'\1', text)
    # Collapse multiple blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def download_book(book_id, title, output_dir, timeout=15):
    """Download a single book; returns (filename, char_count) or None."""
    alt_urls = [
        f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt",
        f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
        f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
    ]
    fpath = os.path.join(output_dir, f"book_{book_id}.txt")
    
    for attempt, u in enumerate(alt_urls):
        try:
            req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode('utf-8', errors='replace')
            stripped = strip_gutenberg(raw)
            cleaned = clean_text(stripped)
            if len(cleaned) < 500:
                continue
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            return fpath, len(cleaned)
        except Exception:
            pass
    # Try one more time with longer timeout
    try:
        u = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
        req = urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw = resp.read().decode('utf-8', errors='replace')
        stripped = strip_gutenberg(raw)
        cleaned = clean_text(stripped)
        if len(cleaned) >= 500:
            with open(fpath, 'w', encoding='utf-8') as f:
                f.write(cleaned)
            return fpath, len(cleaned)
    except Exception:
        pass
    return None, 0

def main():
    output = "corpus.txt"
    min_books = 3
    fast = False
    
    args = sys.argv[1:]
    for i, a in enumerate(args):
        if a == '--output' and i+1 < len(args):
            output = args[i+1]
        if a == '--min-books' and i+1 < len(args):
            try: min_books = int(args[i+1])
            except: pass
        if a == '--fast':
            fast = True
    
    books = BOOKS[:5] if fast else BOOKS

    tmpdir = "gutenberg_raw"
    os.makedirs(tmpdir, exist_ok=True)
    
    print(f"Downloading {len(books)} classic books to build training corpus...\n")
    
    all_text = []
    succeeded = 0
    total_chars = 0
    
    for book_id, title in books:
        print(f"  [{book_id:5d}] {title}...", end=' ', flush=True)
        fpath, chars = download_book(book_id, title, tmpdir)
        if fpath and chars > 0:
            with open(fpath, 'r', encoding='utf-8') as f:
                all_text.append(f.read())
            print(f"OK ({chars:,} chars)")
            succeeded += 1
            total_chars += chars
        else:
            print()
        time.sleep(0.3)  # be polite to Gutenberg
    
    if succeeded < min_books:
        print(f"\nFAIL: Only got {succeeded} books (need {min_books}). Aborting.")
        sys.exit(1)
    
    # Combine
    combined = "\n\n".join(all_text)
    with open(output, 'w', encoding='utf-8') as f:
        f.write(combined)
    
    print(f"\nOK: Saved {succeeded} books -> {output}")
    print(f"  Total: {total_chars:,} chars ({total_chars/1e6:.1f}M)")
    print(f"  Estimated tokens: ~{total_chars//4:,}")
    
    # Cleanup raw files
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
    
    print(f"  (removed temp files)")

if __name__ == '__main__':
    main()
