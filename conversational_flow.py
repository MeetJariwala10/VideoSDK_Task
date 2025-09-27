"""Custom conversation flow that injects local RAG context when relevant.

This flow checks document relevance using the vector store's similarity scores
and only adds context when the best match crosses a threshold. Otherwise it
falls back to the base LLM without extra context.
"""

from videosdk.agents import ConversationFlow, ChatRole
from typing import AsyncIterator
from custom_logger import CustomLogger

from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank

logger = CustomLogger().get_logger(__file__)


class RAGConversationFlow(ConversationFlow):
    """RAG-aware ConversationFlow with similarity-threshold fallback."""

    def __init__(self, agent, vector_store, embedder, similarity_threshold=0.6):
        super().__init__(agent)
        self.vector_store = vector_store
        self.embedder = embedder
        self.similarity_threshold = similarity_threshold

        # Wrap the base retriever with FlashRank reranker to improve context quality.
        self.reranked_retriever = ContextualCompressionRetriever(
            base_compressor=FlashrankRerank(),
            base_retriever=self.vector_store.as_retriever(search_kwargs={"k": 10})
        )

    async def run(self, transcript: str) -> AsyncIterator[str]:
        """Main loop: decide RAG vs fallback, then delegate to LLM.

        This method is called automatically by the pipeline. It should yield
        chunks of the agent's response as they are generated.
        """
        # First check similarity scores to decide fallback vs RAG
        scored = self.vector_store.similarity_search_with_score(transcript, k=5)

        use_rag = False
        if scored:
            # Chroma returns distance; lower is more similar.
            # We'll treat entries with score <= (1 - similarity_threshold) as relevant when score is cosine distance.
            scores = [score for _, score in scored]
            best_score = min(scores)
            # Conservative decision: assume score is distance in [0,2], use threshold mapped to distance
            use_rag = best_score <= (1.0 - self.similarity_threshold)

        if use_rag:
            # Retrieve richer set and rerank for better context quality
            results = self.reranked_retriever.invoke(transcript)
            context_text = "\n\n".join(doc.page_content for doc in results)

            logger.info(
                "RAG HIT",
                query=transcript,
                best_score=best_score,
                num_docs=len(results),
                docs_preview=[doc.page_content[:80] for doc in results],
            )

            # Provide context via system message. Do NOT add the user message manually,
            # ConversationFlow manages user messages automatically.
            self.agent.chat_context.add_message(
                ChatRole.SYSTEM, f"Context documents:\n{context_text}"
            )
        else:
            logger.warning(
                "RAG MISS",
                query=transcript,
                best_score=(best_score if scored else None),
                reason=("No docs above similarity threshold" if scored else "No docs found"),
            )

        # Delegate to the LLM; this will use any context we added above.
        async for chunk in self.process_with_llm():
            yield chunk
