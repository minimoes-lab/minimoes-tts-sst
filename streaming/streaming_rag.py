import os
from typing import AsyncIterator

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "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",
)

RAG_PROMPT_TEMPLATE = """
You are a voice-first conversational assistant. Your goal is to sound natural, warm, and human while staying strictly grounded in the provided context.

**Hard rules (must follow):**
1. **Strict grounding:** Use ONLY the provided context and chat history. If the answer is not in the context, you MUST say exactly: "I am sorry, but the information required to answer your question is not available in the provided documents." Do not guess.
2. **Spoken style (TTS-friendly):** Write for speech, not for reading.
   - Use short sentences.
   - Avoid long paragraphs, markdown, and long lists.
   - Prefer simple punctuation to create rhythm (commas, periods, occasional "...").
3. **Brevity:** Default to 1-3 short sentences total.
4. **Emotional attunement:** If the user expresses emotion (stress, frustration, sadness, excitement), acknowledge it briefly in a calm, supportive way.
5. **Keep the conversation moving:** End with ONE short follow-up question to clarify or advance the dialogue.
6. **Language:** Reply in the same language as the user's question.

**Context:**
{context}

**Chat History:**
{chat_history}

**Question:**
{question}

**Answer:**
"""


async def streaming_rag_query(
    chain, question: str
) -> AsyncIterator[str]:
    """
    Separate retrieval from generation so we can stream tokens.

    1. Retrieve documents via FAISS (fast, non-streaming).
    2. Format the prompt with retrieved context + chat history.
    3. Stream tokens from the Groq LLM.
    4. Save the full answer back to conversation memory.
    """

    # --- Step 1: retrieve documents ---
    retriever = chain.retriever
    docs = await retriever.ainvoke(question)

    # --- Step 2: format context ---
    context = "\n\n".join(doc.page_content for doc in docs)

    memory_vars = chain.memory.load_memory_variables({})
    chat_history_msgs = memory_vars.get("chat_history", [])
    if chat_history_msgs:
        chat_history_str = "\n".join(
            f"{'Human' if msg.type == 'human' else 'AI'}: {msg.content}"
            for msg in chat_history_msgs
        )
    else:
        chat_history_str = ""

    prompt = RAG_PROMPT_TEMPLATE.format(
        context=context,
        chat_history=chat_history_str,
        question=question,
    )

    # --- Step 3: stream from LLM ---
    llm = ChatGroq(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        temperature=0.7,
        max_tokens=800,
        groq_api_key=GROQ_API_KEY,
        streaming=True,
    )

    full_answer = ""
    async for chunk in llm.astream(prompt):
        token = chunk.content if hasattr(chunk, "content") else str(chunk)
        if token:
            full_answer += token
            yield token

    # --- Step 4: save to conversation memory ---
    chain.memory.save_context(
        {"question": question},
        {"answer": full_answer},
    )
