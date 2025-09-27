import asyncio, os
from dotenv import load_dotenv
from videosdk.agents import Agent, AgentSession, CascadingPipeline, JobContext, RoomOptions, WorkerJob,ConversationFlow
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS
from videosdk.plugins.google import GoogleLLM
from videosdk.plugins.google import GoogleTTS, GoogleVoiceConfig
from videosdk.plugins.sarvamai import SarvamAILLM
from videosdk.plugins.cerebras import CerebrasLLM
from conversational_flow import RAGConversationFlow
from rag_setup import build_vector_store

# Pre-downloading the Turn Detector model
pre_download_model()

load_dotenv()

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="You are a helpful voice assistant that can answer questions and help with tasks.")
    async def on_enter(self): await self.session.say("Hello! How can I help?")
    async def on_exit(self): await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    
    vector_store, embedder = build_vector_store()

    # Create agent and conversation flow
    agent = MyVoiceAgent()

    conversation_flow = RAGConversationFlow(agent, vector_store, embedder)
    # pipeline.set_conversation_flow(conversation_flow)

    # Configure voice settings
    voice_config = GoogleVoiceConfig(
        languageCode="en-US",
        name="en-US-Chirp3-HD-Aoede",
        ssmlGender="FEMALE"
    )

    # Create pipeline
    pipeline = CascadingPipeline(
        stt=DeepgramSTT(model="nova-2", language="en", api_key=os.getenv("DEEPGRAM_API_KEY")),
        
        # llm=OpenAILLM(model="gpt-4o"),
        # llm=GoogleLLM(model="gemini-2.0-flash", api_key=os.getenv("GOOGLE_API_KEY")),
        # llm = SarvamAILLM(
        #     model="sarvam-m",
        #     api_key=os.getenv("SARVAM_API_KEY"),
        #     # temperature=0.7,
        #     # tool_choice="auto",
        #     # max_completion_tokens=1000
        # ),

        llm = CerebrasLLM(
            model="llama3.3-70b",
            temperature=0.7,
            api_key=os.getenv("CARTESIA_API_KEY")
        ),

        tts=ElevenLabsTTS(model="eleven_flash_v2_5", api_key=os.getenv("ELEVENLABS_API_KEY")),
        # tts=GoogleTTS(api_key=os.getenv("GOOGLE_API_KEY"), speed=1.0, pitch=0.0, voice_config=voice_config),
        
        vad=SileroVAD(threshold=0.35),
        
        turn_detector=TurnDetector(threshold=0.8)
    )

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
    room_options = RoomOptions(
        # room_id="YOUR_MEETING_ID",  # Set to join a pre-created room; omit to auto-create
        room_id="ost3-r1ev-tgp2",
        name="VideoSDK Cascaded Agent",
        playground=True
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()