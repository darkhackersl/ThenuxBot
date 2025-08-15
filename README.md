# ThenuxBot — Your Own Local AI

This is a complete starter you can run locally:
- Transformer model (`distilgpt2`) you can fine-tune on your own chats.
- Persistent memory using TF‑IDF retrieval (remembers past chats).
- FastAPI backend + simple ChatGPT-like web UI.
- No third‑party API calls.

## 1) Setup
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) (Optional) Fine-tune your own model
Edit/extend `data/sample.jsonl` with your own conversations.
Format per line:
```json
{"turns":[{"role":"user","content":"Hii"},{"role":"assistant","content":"Hello!"}]}
```

Train:
```bash
python train.py --data data/sample.jsonl --out model --base distilgpt2 --epochs 3
```
After training, the server will automatically prefer `./model`.

## 3) Run the server
```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```
Open http://localhost:8000 in your browser.

## Tips
- The bot stores memory in `data/memory.json`. Delete it to forget.
- Use a bigger base model if you have GPU (e.g., `--base gpt2` or `openai-community/gpt2` alternatives).
- You can change the system prompt inside `server.py` to make the bot more “you”.

Enjoy building your own AI, Thenux! 🇱🇰

