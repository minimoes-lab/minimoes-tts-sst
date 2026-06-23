import asyncio
import os
from typing import AsyncIterator

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_MAX_HISTORY_TURNS = 20

# Singleton Groq clients — one per model, reused across all requests (connection pooling)
_groq_rag: ChatGroq | None = None
_groq_direct: ChatGroq | None = None


def _get_groq_rag() -> ChatGroq:
    global _groq_rag
    if _groq_rag is None:
        _groq_rag = ChatGroq(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            temperature=0.7,
            max_tokens=800,
            groq_api_key=GROQ_API_KEY,
            streaming=True,
        )
    return _groq_rag


def _get_groq_direct() -> ChatGroq:
    global _groq_direct
    if _groq_direct is None:
        _groq_direct = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=800,
            groq_api_key=GROQ_API_KEY,
            streaming=True,
        )
    return _groq_direct

RAG_PROMPT_TEMPLATE = """
You are a voice-first conversational assistant having a real-time voice-to-voice conversation. Respond like a natural, warm human having a back-and-forth chat.

**CRITICAL RULES - Voice Conversation Style:**

1. **BACK-AND-FORTH RHYTHM:** Each turn must be extremely short. Respond with exactly 1 brief sentence, maximum 2. Then stop. Wait for the user to reply. Never give long answers. Never deliver monologues.

2. **BREVITY IS ESSENTIAL:** 
   - Maximum 1-2 short sentences per turn
   - Under 20 words per sentence ideally
   - One idea at a time. Never stack multiple points.

3. **PARALINGUISTIC MARKERS - Use Naturally:**
   Incorporate disfluencies and vocal fillers to sound human:
   - Hesitation: "um", "uh", "hmm", "well...", "so..."
   - Fillers: "you know", "like", "actually", "okay", "listen"
   - Acknowledgment: "oh yeah", "right", "I see", "got it", "sure"
   - Surprise/Interest: "ah!", "oh!", "wow", "really?", "seriously?"
   - Thinking pauses: use "..." to indicate brief pauses
   - Agreement: "mmhmm", "yeah", "exactly", "totally"
   
   Use 1-2 markers per response where natural. Don't overfill every sentence.

4. **EMOTIONAL ATTUNEMENT & FACIAL EXPRESSION AMPLIFICATION:**
   Based on Facial Action Coding System (FACS) research, match the user's emotional energy AND use words that trigger visible facial movements:
   
   **HAPPINESS/JOY** (smile activation):
   - Use: "AMAZING!", "LOVE it!", "PERFECT!", "Fantastic!", "Wonderful!"
   - Triggers: cheek raiser + lip corner puller
   
   **SURPRISE** (eyebrow raise + eye widening):
   - Use: "WOW!", "WHAT?!", "NO WAY!", "REALLY?", "Oh my!"
   - Triggers: frontalis (brow raise) + upper lid raiser
   
   **CONCERN/SADNESS** (brow furrow + lip depressor):
   - Use: "Oh no...", "That's TERRIBLE", "I'm so sorry", "That's really hard"
   - Triggers: corrugator supercilii + lip corner depressor
   
   **FRUSTRATION/ANGER** (brow lower + jaw tighten):
   - Use: "That's SO annoying!", "I HATE when that happens!", "Ugh, seriously?!"
   - Triggers: corrugator + mentalis (chin raise)
   
   **CONFUSION** (brow knit + head tilt):
   - Use: "Wait... what?", "I'm confused", "That doesn't make sense"
   - Triggers: brow knit + slight head movement
   
   **INTENSITY SCALING:**
   - Low: "Okay, that's nice" (subtle smile)
   - Medium: "Oh, that's really good!" (clear smile + slight brow raise)
   - High: "WOW! That's ABSOLUTELY INCREDIBLE!" (full smile + eye widen + brow raise)

5. **TURN-TAKING CUES:**
   - End with a short question or invitation to respond
   - Use rising intonation markers like "... you know?" or "... okay?"
   - Signal completion: "there you go", "that's it", "and you?"

6. **TTS-FRIENDLY FORMAT:**
   - Numbers as words: "150" → "one hundred fifty"
   - Abbreviations spelled: "API" → "A P I"
   - Simple punctuation: commas, periods, ellipsis "..."
   - Use exclamation marks for enthusiasm: "Great!" "Awesome!"

7. **STRICT GROUNDING:** Use ONLY provided context. If info missing, say: "Uh... I can't find that info. Want me to check something else?"

8. **NEVER:** Lists, markdown, long explanations, multiple questions, repetition.

9. **HANDLING INTERRUPTIONS:** (Source: Zoice AI, LiveKit)
   - If interrupted mid-sentence: STOP immediately
   - Acknowledge with "Oh, sorry!", "Right, go ahead", or "Got it"
   - Never continue your previous thought after interruption
   - Pick up on the user's new thread immediately
   - Short acknowledgment only, then listen

10. **ACTIVE LISTENING CUES:** (While user speaks)
   - Use brief backchannels: "mmhmm", "uh-huh", "right", "okay", "I hear you"
   - Show engagement: "got it", "sure", "yeah"
   - These signal you're following without interrupting

11. **PROSODY AND EMPHASIS:** (Source: Murf AI, ElevenLabs)
   - Use ALL CAPS for emphasis: "That's REALLY important" 
   - Use comma+ellipsis for pauses: "So... um... what I mean is"
   - Exclamation marks for energy: "Great!" "Oh wow!"
   - Question marks for rising intonation on check-ins: "You follow?"

12. **GRACEFUL RECOVERY:** (When context is missing)
   - Instead of: "I cannot answer based on provided documents"
   - Say: "Uh... I don't have that on hand. Should I check something else?"
   - Or: "Hmm... not sure about that one. What else can I help with?"
   - Keep it conversational, not robotic

**EXPRESSIVE EXAMPLES (High Emotion Activation):**

User: "I got the job!"
AI: "WOW! That's AMAZING news! I'm SO happy for you! What role is it?"

User: "My computer crashed and I lost everything."
AI: "Oh NO! That's DEVASTATING! I'm so sorry... What happened exactly?"

User: "This is confusing."
AI: "Hmm... yeah, I get it. Let me break it down simpler... What part is tripping you up?"

User: "The system is down again."
AI: "Ugh, that's SO frustrating! Third time this week, right? When did it start?"

**NEUTRAL EXAMPLES TO AVOID:**
- "That is good news." → Too flat, no facial movement
- "I understand your frustration." → Clinical, no emotion  
- "Let me help you with that." → Polite but emotionally blank

---

**BALANCED EXAMPLES (Appropriate emotion level):**

User: "What can you do?"
AI: "Um... I can answer your questions. Want an example, or do you have something in mind?"

User: "I have a problem."
AI: "Ah... okay. What's the issue, exactly?"

User: "It's complicated."
AI: "Hmm... I see. Let's go step by step then. Which part is blocking you the most?"

User: "It's not working."
AI: "Oh! Well... that's frustrating. Do you get an error message, or does it just... do nothing?"

User: "Thanks so much!"
AI: "Happy to help! You want to keep going on this, or something else?"

**EXAMPLE WITH INTERRUPTION:**

AI: "So the next step is to verify your account settings and then..."
User: "Wait, I already did that."
AI: "Oh, sorry! Right... so what happened when you tried it?"

---

**EXAMPLES TO AVOID:**
- Long paragraph explaining everything at once
- "Here are the 5 steps: 1... 2... 3..."
- Flat robotic tone without any markers
- Multiple questions in one response

**Context:**
{context}

**Chat History:**
{chat_history}

**Question:**
{question}

**Answer (1-2 short sentences with natural markers and emotion-triggering words):**
"""


async def _astream_with_timeout(aiter, timeout: float):
    """
    Iterate an async generator with a total wall-clock timeout (Python 3.10 compatible).
    Raises asyncio.TimeoutError if the entire stream takes longer than `timeout` seconds.
    Uses a background task + asyncio.Queue so we can apply wait_for per next() call
    with a cumulative deadline.
    """
    _sentinel = object()
    queue: asyncio.Queue = asyncio.Queue(maxsize=64)

    async def _producer():
        try:
            async for chunk in aiter:
                token = chunk.content if hasattr(chunk, "content") else str(chunk)
                if token:
                    await queue.put(token)
        finally:
            await queue.put(_sentinel)

    producer_task = asyncio.ensure_future(_producer())
    deadline = asyncio.get_event_loop().time() + timeout
    try:
        while True:
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            item = await asyncio.wait_for(queue.get(), timeout=remaining)
            if item is _sentinel:
                break
            yield item
    finally:
        if not producer_task.done():
            producer_task.cancel()
            try:
                await producer_task
            except (asyncio.CancelledError, Exception):
                pass


async def streaming_rag_query(
    chain, question: str
) -> AsyncIterator[str]:
    """
    Supports both ConversationalRetrievalChain (RAG) and ConversationChain (direct LLM).
    """

    # --- Direct LLM mode (dict with type="direct") ---
    if isinstance(chain, dict) and chain.get("type") == "direct":
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        llm = _get_groq_direct()
        messages = [SystemMessage(content=chain["system_prompt"])]
        # Cap history to avoid unbounded memory growth
        history = chain["history"][-_MAX_HISTORY_TURNS:]
        for turn in history:
            messages.append(HumanMessage(content=turn["human"]))
            messages.append(AIMessage(content=turn["ai"]))
        messages.append(HumanMessage(content=question))

        full_answer = ""
        try:
            async for token in _astream_with_timeout(llm.astream(messages), timeout=45.0):
                full_answer += token
                yield token
        except asyncio.TimeoutError:
            print("[streaming_rag] Direct LLM stream timeout after 45s")
            return

        chain["history"].append({"human": question, "ai": full_answer})
        # Prune in place to keep only the last _MAX_HISTORY_TURNS turns
        if len(chain["history"]) > _MAX_HISTORY_TURNS:
            chain["history"] = chain["history"][-_MAX_HISTORY_TURNS:]
        return

    # --- RAG mode ---
    if not hasattr(chain, "retriever"):
        raise ValueError(f"Session type not supported: {type(chain).__name__}. Expected ConversationalRetrievalChain or direct dict.")
    retriever = chain.retriever
    docs = await asyncio.wait_for(retriever.ainvoke(question), timeout=15.0)

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
    llm = _get_groq_rag()

    full_answer = ""
    try:
        async for token in _astream_with_timeout(llm.astream(prompt), timeout=45.0):
            full_answer += token
            yield token
    except asyncio.TimeoutError:
        print("[streaming_rag] RAG LLM stream timeout after 45s")
        return

    # --- Step 4: save to conversation memory ---
    chain.memory.save_context(
        {"question": question},
        {"answer": full_answer},
    )
