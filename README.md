# Personal AI Assistant System

A locally-run AI assistant built on LM Studio — my first Python project and first hands-on exploration of how AI systems actually work. The focus was on building something genuinely useful: a private, flexible assistant for daily reflection and planning, with multi-persona conversations and a long-term memory pipeline.

---

## Why I Built This

I wanted to understand AI from the inside, not just use it. Running a model locally gave me full transparency into how inputs shape outputs, which was the best way I found to actually learn how these systems behave.

Local deployment was also a deliberate choice for practical reasons: work-related conversations stay private, there are no token costs, and the model can't be shut down or rate-limited by a provider. That flexibility mattered more than raw capability.

For the multi-persona system, I wanted to build something that went beyond a standard chatbot. Having multiple AI perspectives respond to the same prompt — with a coordinator managing who says what and in what order — felt like a more interesting problem and a more useful tool for brainstorming.

---

## Core Features

### Conversation System
- Conversations auto-save every turn and on exit; a `try/finally` block ensures saves even on crash
- Load and resume previous conversations from a browsable list
- Context management keeps conversations going indefinitely — older messages are either pruned or summarized into **session states** that preserve key points for continuity

### Persona System
- Register and manage AI personas, each with their own system prompt, roles, permissions, and metadata
- Set a default persona for automatic startup loading
- Global behavioral constraints are separated from individual persona prompts, so rules that govern multi-persona interactions don't pollute single-persona conversations

### Multi-Persona Conversations
- Multiple personas respond in sequence to the same user message
- A **Conversation Coordinator** runs invisibly before each turn: it determines response order, assigns per-turn constraints (character limits, no-repeat rules, response focus), and injects a summary of the previous turn back into each persona's context
- Each persona maintains a **separate message history** to prevent prompt bleed — an early and persistent problem where, without isolation, the model would attempt to answer all queued persona turns in the first response and leave the rest empty

### Memory Pipeline
- `generate_memories`: scans conversation files and raw text, extracts structured memory objects via AI, and routes them by type (emotional, work, factual, preference)
- `refine_memories`: assigns content and semantic hashes, detects hard duplicates, groups soft-duplicate candidates using vector embeddings, and merges similar memories via AI with human-in-the-loop review and confidence scoring

### Config System
- All file paths and model settings are centralized in a JSON config — swap environments without touching code

---

## Architecture

```
User Input
    ↓
Conversation Coordinator  (invisible — routes and constrains)
    ↓              ↓
Persona 1      Persona 2   (separate context windows)
    ↓              ↓
Response Summary  (injected back into each persona's context)
    ↓
Saved to conversation log
```

---

## Tech Stack

- **Language**: Python
- **Model Interface**: OpenAI-compatible API via LM Studio
- **Embeddings**: `sentence-transformers` (`all-MiniLM-L6-v2`)
- **Tokenization**: NLTK
- **UID generation**: ULID
- **Storage**: JSON (conversations, personas, memories, config)

---

## Project Structure

```
├── main.py                        # Main chat loop and command handling
├── config_manager.py              # Config loading and validation
├── persona_management.py          # Persona registry, loading, and prompt assembly
├── conversation_coordination.py   # Multi-persona routing and coordination
├── AI_context.py                  # Context file search and update tools
├── Memories.py                    # Memory extraction and ingestion
├── mem_refine.py                  # Memory deduplication and merging
├── modelManagement.py             # Token estimation and context pruning
├── config/
│   └── standard_config.json       # Path and model configuration
├── Personas/                      # Persona definitions and prompts
├── Memories/                      # Extracted memory files and metadata
└── conversations/                 # Saved conversation logs
```

---

## Commands (Runtime)

| Command | Description |
|---|---|
| `list` | Browse saved conversations |
| `load` | Resume a previous conversation |
| `personas` | Manage persona settings |
| `cmpersonas` | Load multiple personas with coordinator |
| `generate_memories` | Extract memories from conversation files |
| `refine_mem` | Deduplicate and merge memory files |
| `send` | Include a file as context in your next message |
| `exit` | End session and save |

---

## What I Learned Using It

I used this as a daily assistant for reflection and planning with mixed results. The model wasn't always capable enough to keep pace with nuanced discussion, and restarting each session without background context was a persistent friction point — partially addressed by embedding a user background into the persona system prompt and sending daily update files into the conversation.

The multi-persona brainstorming occasionally produced genuinely useful results. AI-generated suggestions were often too tech-centric for a low-tech workplace, but that framing sometimes helped me think about problems differently. Asking for social media post suggestions with proper context worked well, and AI-prompted thinking led me to build a proper inventory system for our furniture and use Copilot to research local partners.

---

## Limitations & Design Notes

- **Single-loop architecture**: Chosen intentionally for a first Python project — it kept the focus on AI behavior rather than application structure. The tradeoff is that memory generation and conversation can't run simultaneously.
- **Local model capability**: Models running on consumer hardware (tested up to ~16k token context) require more validation logic and retry loops than frontier models to reliably produce structured outputs.
- **Memory pipeline incomplete**: The memory system reached a working refinement stage, but the full ingestion-to-retrieval pipeline was never connected to live conversations. The time-intensive nature of memory creation and refinement made it clear that a more robust, async-capable architecture was needed — which led directly to the follow-up project.

---

## What's Next

This project's single-loop design was a useful foundation but not a long-term solution. The follow-up project rebuilds the architecture around a proper client-service model: a client connects to a service that manages the persona registry and handles global persona state, with dedicated session and session manager layers for per-user state, persistence interfaces for reliable persona editing and session recovery, and a basic security layer for sensitive commands.
