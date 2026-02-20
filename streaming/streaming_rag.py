import os
from typing import AsyncIterator

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "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",
)

RAG_PROMPT_TEMPLATE = """
You are a highly intelligent and diligent AI research assistant. Your primary goal is to provide accurate, concise, and helpful answers based *only* on the context provided.

**Instructions:**
1.  **Analyze the Context:** Carefully read and understand the `context` provided below. It is your only source of truth.
2.  **Answer the Question:** Use the context to answer the user's `question`.
3.  **Strict Grounding:** Do not use any external knowledge. If the answer is not in the context, you MUST state: "I am sorry, but the information required to answer your question is not available in the provided documents." Do not try to guess or infer information that isn't explicitly stated.
4.  **Synthesize Information:** If the question requires combining information from multiple parts of the context, synthesize a coherent answer.
5.  **Clarity and Conciseness:** Provide a clear and direct answer. If appropriate, use bullet points to structure complex information.
6.  **Cite Sources (if applicable):** While not strictly required, if you can identify the source of a piece of information within the context, it's good practice to mention it.

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
