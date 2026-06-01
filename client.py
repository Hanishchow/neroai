"""
Nero client — chat with a remote Nero server.
Usage:
  python client.py --url http://localhost:8000
  python client.py --url https://xxxxx.gradio.live
"""
import sys, argparse, json, requests

parser = argparse.ArgumentParser()
parser.add_argument('--url', default='http://localhost:8000')
args = parser.parse_args()

print(f"Connecting to Nero at {args.url}")
print("Type 'exit' to quit.\n")

while True:
    try:
        msg = input("> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not msg:
        continue
    if msg.lower() in ('exit', 'quit'):
        break

    try:
        if 'gradio' in args.url:
            # Gradio API
            resp = requests.post(f"{args.url}/api/chat", json={
                "data": [msg, None]
            }, timeout=120)
            result = resp.json()
            if result.get('data'):
                print(f"  {result['data'][0]}")
        else:
            # FastAPI
            resp = requests.post(f"{args.url}/chat", json={
                "message": msg, "max_tokens": 300, "temperature": 0.85
            }, timeout=120)
            print(f"  {resp.json()['reply']}")
    except Exception as e:
        print(f"  Error: {e}")

print("Bye.")
