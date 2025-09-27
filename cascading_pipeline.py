"""Cascading voice AI agent entrypoint.

This script wires together the VideoSDK CascadingPipeline stages (VAD → STT → LLM → TTS →
Turn Detection) and starts an AgentSession. It also initializes the RAG conversation flow
so the agent can answer using local documents when relevant.

How to run:
- Playground (browser):
    python cascading_pipeline.py
- Console mode (local mic/speakers):
    python cascading_pipeline.py console

Environment variables are read from .env (see README or .env.example).
"""

import asyncio, os
from dotenv import load_dotenv
from videosdk.agents import Agent, AgentSession, CascadingPipeline, JobContext, RoomOptions, WorkerJob, ConversationFlow
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.cerebras import CerebrasLLM
from videosdk.plugins.google import GoogleTTS, GoogleVoiceConfig  # optional alternative TTS
from conversational_flow import RAGConversationFlow
from rag_setup import build_vector_store

# Pre-downloading the Turn Detector model
pre_download_model()

load_dotenv()

class MyVoiceAgent(Agent):
    """Minimal agent with entry/exit speech.

    You can customize the `instructions` to change the agent's persona.
    """

    def __init__(self):
        super().__init__(instructions="You are a helpful voice assistant that can answer questions and help with tasks.")

    async def on_enter(self):
        """Called when the session starts and the agent joins the room."""
        await self.session.say("Hello! How can I help?")

    async def on_exit(self):
        """Called when the session is closing."""
        await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    """Create vector store, compose pipeline and start the agent session.

    The RAG vector store is built/loaded first. Then we construct a CascadingPipeline
    (Deepgram STT → Cerebras LLM → ElevenLabs TTS → SileroVAD → TurnDetector) and attach
    a `RAGConversationFlow` for context injection. Finally we start the session.
    """
    vector_store, embedder = build_vector_store()

    # Create agent and conversation flow
    agent = MyVoiceAgent()

    conversation_flow = RAGConversationFlow(agent, vector_store, embedder)

    # Configure voice settings
    voice_config = GoogleVoiceConfig(
        languageCode="en-US",
        name="en-US-Chirp3-HD-Aoede",
        ssmlGender="FEMALE"
    )

    # Create pipeline
    pipeline = CascadingPipeline(
        stt=DeepgramSTT(model="nova-2", language="en", api_key=os.getenv("DEEPGRAM_API_KEY")),
        llm=CerebrasLLM(
            model="llama3.3-70b",
            temperature=0.7,
            api_key=os.getenv("CEREBRAS_API_KEY")
        ),

        tts=ElevenLabsTTS(model="eleven_flash_v2_5", api_key=os.getenv("ELEVENLABS_API_KEY")),
        # tts=GoogleTTS(api_key=os.getenv("GOOGLE_API_KEY"), speed=1.0, pitch=0.0, voice_config=voice_config),  # optional
        
        vad=SileroVAD(threshold=0.35),
        
        turn_detector=TurnDetector(threshold=0.8)
    )

    # Attach the conversation flow so RAG context can be injected automatically.
    pipeline.set_conversation_flow(conversation_flow)

    session = AgentSession(
        agent=agent,
        pipeline=pipeline,
        conversation_flow=conversation_flow
    )

    try:
        await context.connect()
        await session.start()
        # Keep the session running until manually terminated
        await asyncio.Event().wait()
    finally:
        # Clean up resources when done
        await session.close()
        await context.shutdown()

def make_context() -> JobContext:
    """Create the JobContext with meeting options.

    If `VIDEOSDK_ROOM_ID` is provided in `.env`, the agent will attempt to join that room.
    Otherwise, it will use a default placeholder or auto-create (depending on SDK behavior).
    Set `playground=True` to get a shareable Playground URL in the logs.
    """
    room_options = RoomOptions(
        # room_id="YOUR_MEETING_ID",  # Set to join a pre-created room; omit to auto-create
        room_id=os.getenv("VIDEOSDK_ROOM_ID", "ost3-r1ev-tgp2"),
        name="VideoSDK Cascaded Agent",
        playground=True
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()