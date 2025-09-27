# VideoSDK Agents – Voice AI Agent with Local RAG

> Demo: Watch the project in action here — https://drive.google.com/drive/folders/1VH1eNnr01HRomqC1A9wgEOZuuSDyhgmx?usp=drive_link

This project implements a voice AI agent using the VideoSDK Agents SDK with a Cascading Pipeline (VAD → STT → LLM → TTS → Turn Detection) and a local RAG pipeline (Chroma + Mistral embeddings). The agent:

- Listens to the user via STT.
- Responds with TTS.
- Uses a local vector store to retrieve relevant context from documents.
- Falls back to the base LLM when no relevant context is found.

If you are non-technical, follow the Quick Start below step-by-step.

## Project Structure

- `cascading_pipeline.py` – Configures the `CascadingPipeline` and starts the `AgentSession`.
- `conversational_flow.py` – `RAGConversationFlow` that injects RAG context or falls back to LLM.
- `rag_setup.py` – Builds/loads a Chroma vector store from files in `docs/`.
- `docs/` – Source documents (txt/pdf) used for RAG.
- `rag_store/` – Persisted Chroma store created after first run.
- `generate_meeting_id.py` – Helper to create a VideoSDK room via REST API.
- `custom_logger.py` – Structured JSON logs (to files in `logs/`).
- `requirements.txt` – Dependencies.
- `logs/` – Log files.

## Prerequisites

- Python 3.12 or higher
- A virtual environment recommended
- API keys for providers you choose:
  - `DEEPGRAM_API_KEY` (STT)
  - `ELEVENLABS_API_KEY` (TTS) or `GOOGLE_API_KEY` if using Google TTS
  - `CEREBRAS_API_KEY` (LLM)
  - `MISTRALAI_API_KEY` (Embeddings)
  - `VIDEOSDK_AUTH_TOKEN` (to create a room via REST if needed)
  - `VIDEOSDK_ROOM_ID` (optional; provide if you want to join an existing room)

Create a file named `.env` in the project folder (see example above). Add your provider API keys.

Where to get keys:
- Deepgram: https://developers.deepgram.com/
- ElevenLabs: https://elevenlabs.io/
- Google (optional for TTS): https://console.cloud.google.com/
- Cerebras (LLM): https://inference.cerebras.ai/
- Mistral (embeddings): https://console.mistral.ai/
- VideoSDK Auth Token: https://app.videosdk.live/

## Quick Start (Windows)

```
# 1) Clone this repository
git clone <your-repo-url>
cd VideoSDK_Task

# 2) Create and activate a virtual environment (Python 3.12+)
python -m venv .venv
.\.venv\Scripts\activate

# 3) Install dependencies
pip install -r requirements.txt
```

## Quick Start (macOS/Linux)

```
# 1) Clone this repository
git clone <your-repo-url>
cd VideoSDK_Task

# 2) Create and activate a virtual environment (Python 3.12+)
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
```

## Configure API Keys

Create a file named `.env` in the project folder (see example above). Add your provider API keys.

## Prepare/Inspect Documents

Place your text/PDF files into the `docs/` directory. On first run, the vector store is built and persisted to `rag_store/`.

## Create a Meeting (optional)

If you need to create a room id via REST:

```
python generate_meeting_id.py
```

Ensure `.env` has `VIDEOSDK_AUTH_TOKEN` set. The script prints the response with a room id (you can copy to `VIDEOSDK_ROOM_ID`).

## Run the Agent

You can run in two modes:

- Playground (browser):
  ```
  python cascading_pipeline.py
  ```
  The console will print a Playground URL. Click it, join, allow microphone permissions, and unmute the agent tile.

- Console Mode (local mic/speakers):
  ```
  python cascading_pipeline.py console
  ```
  Audio comes from your system’s default output device. Use headphones if possible.

By default, `RoomOptions` will use `VIDEOSDK_ROOM_ID` from `.env` if present; otherwise a default placeholder is used. You can also manually edit `make_context()` in `cascading_pipeline.py`.

## How RAG Works Here

- The conversation flow (`RAGConversationFlow.run`) first checks similarity scores from Chroma via `similarity_search_with_score`.
- If the best score indicates a relevant match, it retrieves more documents (k=10) and reranks them using `FlashrankRerank` for higher quality context.
- The selected documents are added to the chat as a system message (context) and `process_with_llm()` is called.
- If no relevant documents are found (below threshold), the flow skips adding context and calls `process_with_llm()` (fallback to base LLM).
- The flow avoids manually adding the user message because the `ConversationFlow` base class already handles chat context management.

You can adjust relevance via `similarity_threshold` in `RAGConversationFlow` (default `0.6`).

## Example Queries

- Matches documents (RAG should trigger):
  - "What is the main idea from the attention paper?" (if `docs/1706.03762v7.pdf` exists)
  - "Tell me about Elon Musk's early ventures."

- Not covered in docs (fallback to LLM):
  - "What's the capital of France?"
  - "Explain bubble sort."

## Troubleshooting

- Missing keys: Ensure `.env` has the required API keys.
- No audio in Playground: join the page, allow mic permission, unmute agent tile, set device in page settings, and ensure the tab/site isn’t muted.
- No audio in Console Mode: set your Windows/macOS default output device, check Volume Mixer for the Python process, disable “exclusive mode”, set output sample rate to 48kHz.
- RAG not triggering: Inspect logs in `logs/` for `RAG HIT`/`RAG MISS`. Tweak `similarity_threshold` or add more relevant docs.
- Rebuild vector store: Delete `rag_store/` and re-run to rebuild from `docs/`.

## Notes
- The current LLM is Cerebras (`llama3.3-70b`). You may switch to other providers exposed by `videosdk` plugins (OpenAI, Google, etc.) if you have keys.
