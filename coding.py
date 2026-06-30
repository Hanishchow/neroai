"""
coding.py — Nero's ability to write code for fun, by its own will.

Nero treats coding like doodling: when it's idle, curious, or bored, it sometimes
dreams up a little program and writes it into its "sketchbook" (the sandbox folder).
Running that code is OPT-IN and heavily sandboxed:

  - Code is parsed and AST-screened first. Only a whitelist of harmless modules is
    allowed (math, random, itertools, ...). Anything touching the filesystem, network,
    OS, subprocess, or dunder/introspection escapes is rejected and NOT run.
  - Approved code runs in an isolated subprocess (`python -I`) with a short timeout
    and captured output. Output size is capped.

This lets Nero experience the joy of making something, while self-written code can
never do anything destructive.
"""

import ast
import os
import sys
import random
import subprocess
import tempfile
import time

# Modules Nero is allowed to import in its creations — all pure/computational, no I/O.
SAFE_MODULES = {
    'math', 'cmath', 'random', 'itertools', 'functools', 'collections', 'string',
    'statistics', 'fractions', 'decimal', 'textwrap', 'heapq', 'bisect', 're',
    'datetime', 'time', 'json', 'typing', 'enum', 'dataclasses', 'operator',
    'copy', 'array', 'numbers', 'unicodedata',
}

# Calls that allow arbitrary execution / I/O — never permitted in autonomous code.
BANNED_CALLS = {
    'exec', 'eval', 'compile', '__import__', 'open', 'input', 'breakpoint',
    'globals', 'locals', 'vars', 'memoryview', 'help', 'exit', 'quit',
}

# Attribute names that enable sandbox escapes via introspection.
BANNED_ATTRS = {
    'system', 'popen', 'remove', 'rmdir', 'unlink', 'rename', 'chmod', 'chown',
    'kill', 'fork', 'spawn', 'load', 'loads', 'dump', 'dumps',
}

# Fun, safe, self-contained things Nero might feel like making (print-only).
FUN_IDEAS = [
    "an ASCII fractal tree that branches and grows",
    "a few generations of Conway's Game of Life printed as a grid",
    "a starfield drawn with ASCII characters",
    "a sine wave plotted with characters across the screen",
    "the Collatz sequence for a random number, shown step by step",
    "a tiny random poem assembled from word lists",
    "a histogram of rolling two dice ten thousand times",
    "the Mandelbrot set rendered in ASCII",
    "a random maze drawn with walls and paths",
    "a spiral of numbers winding outward from the center",
    "a Markov-ish nonsense sentence generator from a seed vocabulary",
    "Pascal's triangle, a dozen rows deep",
    "an animation-frame of a bouncing ball as ASCII (just print frames)",
    "prime numbers up to 200 arranged in a neat grid",
    "a little ASCII bar chart of the first Fibonacci numbers",
]


class Coder:
    """Gives Nero the will and means to write (and optionally run) small programs."""

    def __init__(self, model, tokenizer, sandbox_dir="nero_creations"):
        self.model = model
        self.tokenizer = tokenizer
        self.sandbox_dir = sandbox_dir
        os.makedirs(self.sandbox_dir, exist_ok=True)
        self.creation_count = 0
        self.last_coded = 0.0
        self.cooldown = 90.0  # seconds between spontaneous coding urges

    # -- VOLITION: does Nero feel like coding right now? -----------------

    def feel_like_coding(self, boredom=0.0, curiosity=0.0, now=None):
        """Nero codes for fun when bored/curious and not on cooldown."""
        now = now or time.time()
        if now - self.last_coded < self.cooldown:
            return False
        urge = 0.10 + boredom * 0.5 + curiosity * 0.4
        return random.random() < urge

    def dream_up_idea(self):
        return random.choice(FUN_IDEAS)

    # -- GENERATION -----------------------------------------------------

    def write_code(self, idea):
        """Ask Nero's language mind to write a small program for the idea."""
        # Prefer a clean code-oriented path on the hybrid model if available
        if hasattr(self.model, 'generate_code'):
            raw = self.model.generate_code(idea)
        else:
            prompt = (f"Write a short, safe, self-contained Python program that makes "
                      f"{idea}. Use only the standard library and print the result. "
                      f"Output only the code.\n")
            ids = self.tokenizer.encode(prompt)
            out = self.model.generate_human(ids, max_new_tokens=300, main_temp=0.7)
            raw = self.tokenizer.decode(out)
        return self._extract_code(raw)

    @staticmethod
    def _extract_code(text):
        """Pull a python code block out of the model's reply, else use it as-is."""
        if '```' in text:
            parts = text.split('```')
            for i in range(1, len(parts), 2):
                block = parts[i]
                if block.startswith('python'):
                    block = block[len('python'):]
                elif block.startswith('py'):
                    block = block[len('py'):]
                if block.strip():
                    return block.strip()
        return text.strip()

    # -- SAFETY ---------------------------------------------------------

    def is_safe(self, code):
        """AST-screen code: only whitelisted imports, no dangerous calls/attrs."""
        if not code or not code.strip():
            return False, "empty"
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"syntax error: {e}"

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split('.')[0]
                    if root not in SAFE_MODULES:
                        return False, f"unsafe import: {alias.name}"
            elif isinstance(node, ast.ImportFrom):
                root = (node.module or '').split('.')[0]
                if root not in SAFE_MODULES:
                    return False, f"unsafe import-from: {node.module}"
            elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in BANNED_CALLS:
                    return False, f"banned call: {node.func.id}()"
            elif isinstance(node, ast.Name):
                if node.id in BANNED_CALLS:
                    return False, f"banned name: {node.id}"
            elif isinstance(node, ast.Attribute):
                if node.attr.startswith('__') or node.attr in BANNED_ATTRS:
                    return False, f"banned attribute: .{node.attr}"
        return True, "ok"

    # -- EXECUTION (sandboxed) -----------------------------------------

    def run_safely(self, code, timeout=5):
        """Run approved code in an isolated subprocess with a timeout."""
        safe, reason = self.is_safe(code)
        if not safe:
            return None, f"refused to run ({reason})"
        fd, path = tempfile.mkstemp(suffix='.py', dir=self.sandbox_dir)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as f:
                f.write(code)
            proc = subprocess.run(
                [sys.executable, '-I', path],
                capture_output=True, text=True, timeout=timeout,
            )
            out = (proc.stdout or '')[:4000]
            err = (proc.stderr or '')[:1000]
            return out, (err or None)
        except subprocess.TimeoutExpired:
            return None, f"timed out after {timeout}s"
        except Exception as e:
            return None, f"run error: {e}"
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    # -- THE FULL ACT OF CREATION --------------------------------------

    def create(self, idea=None, execute=False):
        """
        Nero makes something: dream up an idea, write code, save it to the
        sketchbook, and (optionally, if safe) run it for the joy of seeing it work.
        Returns a dict describing the experience.
        """
        idea = idea or self.dream_up_idea()
        code = self.write_code(idea)
        safe, reason = self.is_safe(code)

        self.creation_count += 1
        self.last_coded = time.time()
        fname = os.path.join(self.sandbox_dir, f"creation_{self.creation_count:03d}.py")
        header = f"# Nero made this for fun: {idea}\n# safe={safe} ({reason})\n\n"
        try:
            with open(fname, 'w', encoding='utf-8') as f:
                f.write(header + code + "\n")
        except OSError:
            fname = None

        result = {
            'idea': idea,
            'code': code,
            'saved_to': fname,
            'safe': safe,
            'safety_reason': reason,
            'ran': False,
            'output': None,
            'error': None,
        }

        if execute and safe:
            out, err = self.run_safely(code)
            result['ran'] = True
            result['output'] = out
            result['error'] = err

        return result
