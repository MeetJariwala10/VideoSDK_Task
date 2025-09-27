from videosdk.agents import ConversationFlow, ChatRole
from typing import AsyncIterator
from custom_logger import CustomLogger

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

logger = CustomLogger().get_logger(__file__)

class RAGConversationFlow(ConversationFlow):
    def __init__(self, agent, vector_store, embedder, similarity_threshold=0.6):
        super().__init__(agent)
        self.vector_store = vector_store
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

        # Wrap the base retriever with FlashRank reranker
        self.reranked_retriever = ContextualCompressionRetriever(
            base_compressor=FlashrankRerank(),
            base_retriever=self.vector_store.as_retriever(search_kwargs={"k": 10})
        )

    async def run(self, transcript: str) -> AsyncIterator[str]:
        # Use reranked retriever instead of direct similarity_search
        results = self.reranked_retriever.invoke(transcript)
        
        if results:
            context_text = "\n\n".join(doc.page_content for doc in results)
            
            logger.info(
                "RAG HIT",
                query=transcript,
                num_docs=len(results),
                docs_preview=[doc.page_content[:80] for doc in results],
            )
            
            self.agent.chat_context.add_message(
                ChatRole.SYSTEM, f"Context documents:\n{context_text}"
            )
        else:
            logger.warning(
                "RAG MISS",
                query=transcript,
                reason="No documents passed similarity threshold",
            )

        self.agent.chat_context.add_message(ChatRole.USER, transcript)

        async for chunk in self.process_with_llm():
            yield chunk
