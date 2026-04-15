# Memories.py
"""
For all memory related functionality
Desired end product:
    1. Memory ingestion
    2. Memory processing
    3. Memory search
    4. Memory merge
"""
import time
from pathlib import Path
import json
import os
from typing import Optional, Tuple
from datetime import datetime #validate_memory_object
from string import Template
from typing import Dict, List, Tuple, Any #build_text_chunks
from pyexpat.errors import messages
from collections import defaultdict #generate memories reporting
import re #get_date_from_chat_filename
import nltk #build_raw_text_chunks
nltk.download("punkt")
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
from collections import Counter # ChunkStats
from dataclasses import dataclass, field #ChunkStats
from collections import defaultdict # generate_memories

EXCLUDED_DIRS = {
    "Memories",
    ".git",
    "__pycache__",
    "venv",
    ".venv"
    "mistralai"
}

EXCLUDED_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif",
    ".pdf", ".zip", ".exe", ".bin",
    ".py", ".yaml", "yml",
    ".pyc", ".pyo"
}

# Assumes generate_search_path() and read_file() are defined in this module
# Assumes extract_memories_with_ai(chunk_text) is defined elsewhere

def append_to_file(memory_file: str, memories: list):
    """Append a list of memory objects to a JSON file."""
    file_path = Path(memory_file)
    if file_path.exists():
        with open(file_path, "r+", encoding="utf-8") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
            existing.extend(memories)
            f.seek(0)
            json.dump(existing, f, indent=2)
            f.truncate()
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2)

HARD_MAX_FILES = 100
HARD_MAX_RUNTIME_SECONDS = 8 * 60 * 60  # 8 hours
DEFAULT_MAX_FILES = HARD_MAX_FILES
DEFAULT_MAX_RUNTIME_SECONDS = HARD_MAX_RUNTIME_SECONDS

def generate_memories(
    search_dir: str,
    CMemotional_file: str,
    CMuser_file: str,
    CMwork_file: str,
    CMfact_file: str,
    memories_file: str,
    client,
    model_name : str,
    char_limit: int = 2000,
    overlap: int = 200,
    max_files: int | None = None,
    max_runtime_seconds: int | None = None
) -> None:

    #start timer for time limit
    start_time = time.monotonic() # never use time.time()

    #load already processed files
    processed_files_path = Path("Memories/meta/processed_files.json")
    if processed_files_path.exists():
        with open(processed_files_path, "r", encoding="utf-8") as f:
            processed_files = json.load(f)
    else:
        processed_files = {}

    #create list of files to search
    search_paths = generate_search_path(search_dir)
    total_files = len(search_paths)
    total_memories = 0
    errors = []

    # Load tags
    tags_file_path = Path("Memories/meta/tags.json")
    if not tags_file_path.exists():
        raise FileNotFoundError(f"Tags file not found at {tags_file_path}")

    with open(tags_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        tags_list = data.get("tags", [])
    print(f"DEBUG: Valid tags: {tags_list}")
    if not tags_list:
        raise ValueError("Tags list is empty in tags.json")

    #Report generation
    report: Dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "total_files": total_files,
        "files_processed": 0,
        "memories_created": 0,
        "errors": [], #AI errors / file read errors
        "invalid_memories": [],
        "invalid_memories_count": 0,
        "conversation_json": 0,
        "raw_text": 0
    }
    invalid_memories = []
    chunk_stats = ChunkStats() #chunk diagnostics

    invalid_tag_stats = defaultdict(lambda: {
        "count": 0,
        "repaired_to": defaultdict(int),
        "memory_types": defaultdict(int),
    })
    print(f"max files: {max_files}")
    #resolve file and runtime limits
    resolved_max_files, resolved_max_runtime = _normalize_limits(
        max_files,
        max_runtime_seconds
    )

    print(f"beginning file search for {len(search_paths)} files")
    fileno = 0
    processed_file_no = 0
    conv_json_no = 0
    raw_text_no = 0
    for file_path in search_paths:
        #check file and time cap before proceeding
        # ---------- hard file cap ----------
        if processed_file_no >= resolved_max_files:
            print(f"⚠️ File limit reached ({resolved_max_files}), stopping.")
            break

        # ---------- hard time cap ----------
        elapsed = time.monotonic() - start_time
        if elapsed >= resolved_max_runtime:
            print(
                f"⏱️ Time limit reached "
                f"({elapsed:.1f}s / {resolved_max_runtime}s), stopping."
            )
            break

        fileno += 1
        print(f"\nprocessing {file_path}, file {fileno} / {total_files}, processed files: {processed_file_no}")

        file_mtime = os.path.getmtime(file_path)  # Last modified timestamp

        # Skip if file unchanged since last processed
        if processed_files.get(file_path) == file_mtime:
            print(f"Skipping unchanged file: {file_path}")
            continue

        #Determine source type
        source_check = check_mem_source(file_path)
        is_conversation_json = source_check["is_valid"]
        #system_ranges = source_check.get("system_message_indices", [])

        # Resolve memory date ONCE per file
        source_date = resolve_memory_date(
           file_path=file_path,
           is_conversation=is_conversation_json
        )
        print(f"source_date: {source_date}")
        source_check = check_mem_source(file_path)

        if is_conversation_json:
            # Structured conversation path
            conv_json_no += 1
            conversation = source_check["conversation"]
            system_ranges = source_check["system_message_indices"]

            chunks = build_text_chunks(
                conversation=conversation,
                system_message_indices=system_ranges,
                char_limit=char_limit,
                stats = chunk_stats,
                file_id=file_path
            )

        else:
            # Fallback: unstructured text
            raw_text_no += 1
            try:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    raw_text = f.read()
            except Exception as e:
                report["errors"].append({
                    "file": file_path,
                    "error_type": "file_read",
                    "message": str(e)
                })
                continue

            chunks = build_raw_text_chunks(
                raw_text,
                char_limit=char_limit,
                stats = chunk_stats,
                file_id=file_path
            )

        for chunk in chunks:
            try:
                #print(f"[DEBUG] Sending chunk to AI from file: {file_path}")
                # Call AI to extract memories from this chunk
                #TODO: when multiple personas are implemented - make persona dynamic

                extracted_memories = extract_memories_with_ai(chunk_text=chunk,
                        persona="Luna",
                        client=client,
                        model_name=model_name,     # pass the model name
                        tags_list=tags_list)

                # validate each memory object here
                valid_memories = []
                if len(extracted_memories) == 0:
                    print("no memories extracted from chunk")
                for memory in extracted_memories:
                    print(f"testing memory: {memory}")

                    if validate_memory_object(memory, tags_list, strict=True):
                        print(f"valid memory created")
                        valid_memories.append(memory)
                    elif validate_memory_object(memory, tags_list, strict=False):
                        print("Invalid memory created")
                        if repair_tags(memory, tags_list, invalid_tag_stats):
                            print("Invalid memory repaired")
                            valid_memories.append(memory)
                        else:
                            print("Invalid memory not salvageable, adding to invalid memories")
                            invalid_memories.append(memory)
                    else:
                        print("Invalid memory created")
                        invalid_memories.append(memory)
                #add dates after validation
                for memory in valid_memories:
                    memory["memory_date"] = source_date
                # Route memories to appropriate file
                core_emotional = [m for m in valid_memories if m['memory_type'] == 'emotional']
                core_work = [m for m in valid_memories if m['memory_type'] == 'work']
                core_user = [m for m in valid_memories if m['memory_type'] == 'preference']  # example mapping
                core_fact = [m for m in valid_memories if m['memory_type'] == 'factual']  # example mapping
                misc = [m for m in valid_memories if m['memory_type'] not in ['emotional', 'work', 'preference', 'factual']]

                if core_emotional:
                    append_to_file(CMemotional_file, core_emotional)
                if core_work:
                    append_to_file(CMwork_file, core_work)
                if core_user:
                    append_to_file(CMuser_file, core_user)
                if core_fact:
                    append_to_file(CMfact_file, core_fact)
                if misc:
                    append_to_file(memories_file, misc)

                #file_memories.extend(valid_memories) - use in the future for more detailed reporting?
                total_memories += len(valid_memories)

            except Exception as e_ai:
                    #errors.append(f"AI error on file {file_path}: {str(e_ai)}") - revised for consistency
                errors.append({
                  "file": file_path,
                  "error_type": "AI" or "File",
                  "message": str(e_ai),
                })

                print(f"File #{fileno}, {file_path}, error, {e_ai} skipping current file") #skip current file
                break
        #if file_pointer is None:
         #   break  # EOF
        processed_file_no += 1
        processed_files[file_path] = file_mtime
        report['files_processed'] += 1

    report["chunk_diagnostics"] = chunk_stats.finalize()
    report['memories_created'] = total_memories
    report['errors'] = errors
    report['invalid_memories'] = invalid_memories
    report['invalid_memories_count'] = len(invalid_memories)
    report['conversation_json'] = conv_json_no
    report['raw_text'] = raw_text_no
    report["invalid_tags"] = normalize_stats(invalid_tag_stats)
    report["proposed_tags"] = [
        tag for tag, data in report["invalid_tags"].items()
        if data["count"] >= 10
    ]

    # Save report
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    report_path = Path(f"Memories/meta/generation_report_{timestamp}.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Save the updated file tracking info
    processed_files_path.parent.mkdir(parents=True, exist_ok=True)
    with open(processed_files_path, "w", encoding="utf-8") as f:
        json.dump(processed_files, f, indent=2)
    print(f"Memory generation complete. Files processed: {report['files_processed']}, Memories created: {total_memories}")
    if errors:
        print(f"Errors encountered: {len(errors)} (see {report_path})")

#Helper function for generate_memories
def extract_memories_with_ai(
    chunk_text: str,
    persona: str,
    client,
    model_name: str,
    tags_list: list
) -> list:
    #print("[DEBUG] extract_memories_with_ai() entered")
    # Build prompt dynamically
    prompt_file = "Memories/meta/prompts/extract_memories_prompt.txt"
    prompt = load_prompt(prompt_file, persona, tags_list, chunk_text)
    print(f"[DEBUG] prompt loaded; calling AI to evaluate")
    # Call the AI client
    try:
        # The exact call depends on your client; e.g., client.chat() or client.complete()

        # send to AI
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": prompt}],
            max_tokens=1500,
            temperature=0.0
        )
        text_output = response.choices[0].message.content
        print("AI Response: ", text_output)
        # Parse JSON
        memories = json.loads(text_output)
        # Optional: basic sanity check on each memory
        if not isinstance(memories, list):
            raise ValueError("AI did not return a JSON array")

        for m in memories:
            if not isinstance(m, dict):
                raise ValueError("Memory item is not a dictionary")

        return memories

    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error from AI output: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Error extracting memories: {str(e)}") from e




ALLOWED_MEMORY_TYPES = ["emotional", "work", "factual", "preference"]
ALLOWED_MEMORY_DOMAINS = ["personal", "project", "system", "meta"]
ALLOWED_ORIGINS = ["user", "assistant", "persona"]

def validate_memory_object(memory: dict,tags_list: list = None, strict: bool = True) -> bool:
    """
    Validate a single memory object against the schema.

    Args:
        memory: dict returned from AI
        tags_list: list of allowed tags

    Returns:
        True if valid, False otherwise
    """
        # ---------- Structural checks ----------
    required_fields = {
        "id": int,
        "schema_version": str,
        "memory_type": str,
        "memory_domain": str,
        "primary_tag": str,
        "tags": list,
        "owner": (str, type(None)),
        "origin": str,
        "shared": bool,
        "salience": int,
        "stability": int,
        "certainty": float,
        "keywords": list,
        "content": str,
        "merged_from": list,
    }

    optional_fields = {
        "memory_date": str #injected later
    }

    # Check all required fields exist and have correct typing
    #debug
    #print("required_fields =", required_fields)
    #print("type(required_fields) =", type(required_fields))

    #ensure required fields are present and correct type
    for field, expected_type in required_fields.items():
        if field not in memory:
            print(f"Invalid Memory: field {field} not found in memory")
            return False
        if not isinstance(memory[field], expected_type):
            print(f"Invalid Memory: field {field} not of type {expected_type}")
            return False

    # Check for typing if AI entered something in optional fields
    for field, expected_type in optional_fields.items():
        if field in memory and not isinstance(memory[field], expected_type):
            print(f"Invalid Memory: optional field '{field}' wrong type")
            return False

    # ---------- Enum checks ----------
    if memory["memory_type"] not in {"emotional", "work", "factual", "preference"}:
        print("Invalid Memory - Enum check failed")
        return False

    if memory["memory_domain"] not in {"personal", "project", "system", "meta"}:
        print("Invalid Memory - Enum check failed")
        return False

    if memory["origin"] not in {"user", "assistant", "persona"}:
        print("Invalid Memory - Enum check failed")
        return False

    # ---------- Range checks ----------
    if not (1 <= memory["salience"] <= 100):
        print("Invalid Memory - Range check failed")
        return False
    if not (1 <= memory["stability"] <= 100):
        print("Invalid Memory - Range check failed")
        return False
    if not (0.0 <= memory["certainty"] <= 1.0):
        print("Invalid Memory - Range check failed")
        return False

    # ---------- Content quality ----------
    if len(memory["content"].strip()) < 20:
        print("Invalid Memory - Content length check failed")
        return False
    if not (3 <= len(memory["keywords"]) <= 5):
        print("Invalid Memory - Keyword length check failed")
        return False

    # ---------- Tag checks ----------
    tag_violations = False

    if memory["primary_tag"] and memory["primary_tag"] not in tags_list:
        tag_violations = True

    for t in memory["tags"]:
        if t not in tags_list:
            tag_violations = True

    if strict and tag_violations:
        return False

    # No longer validate date format - entered by program

    return True


#helper for generate_memories()
def generate_search_path(root_dir: str) -> List[str]:
    """
    Recursively traverse root_dir and return a list of file paths
    suitable for offline memory ingestion.

    Excludes system, memory, binary, and hidden files.
    """

    search_paths: List[str] = []

    root_dir = os.path.abspath(root_dir)

    for current_root, dirnames, filenames in os.walk(root_dir):
        # Modify dirnames in-place to prevent descending into excluded dirs
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDED_DIRS and not d.startswith(".")
        ]

        for filename in filenames:
            if filename.startswith("."):
                continue

            _, ext = os.path.splitext(filename)
            if ext.lower() in EXCLUDED_EXTENSIONS:
                continue

            full_path = os.path.join(current_root, filename)

            # Final sanity check
            if os.path.isfile(full_path):
                search_paths.append(full_path)

    return search_paths

# Memories.py

#helper for generate_memories
def read_file(
    file_path: str,
    char_limit: int,
    overlap: int,
    file_pointer: Optional[int]
) -> Tuple[str, Optional[int]]:
    """
    Read a chunk of a file with character limits and overlap.

    Args:
        file_path: Absolute path to file
        char_limit: Max number of characters to read
        overlap: Number of characters to overlap with next read
        file_pointer: Current file offset (None means start of file)

    Returns:
        content: The text read from the file
        next_pointer: Next file offset, or None if EOF reached
    """

    if char_limit <= 0:
        raise ValueError("char_limit must be > 0")

    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        # Seek to start or provided pointer
        if file_pointer is not None:
            f.seek(file_pointer)
        else:
            f.seek(0)

        content = f.read(char_limit)

        if not content:
            # Empty file or pointer already at EOF
            return "", None

        current_pos = f.tell()

        # Check for EOF
        f.seek(0, 2)  # Move to end of file
        eof_pos = f.tell()

        if current_pos >= eof_pos:
            # EOF reached
            return content, None

        # Calculate next pointer with overlap
        next_pointer = max(current_pos - overlap, 0)

        return content, next_pointer

from typing import List, Dict, Any, Tuple

@dataclass
class ChunkStats:
    total_chunks: int = 0
    total_chars: int = 0
    min_chunk_size: int = float("inf")
    max_chunk_size: int = 0

    small_chunks: int = 0     # < MIN_CHUNK_CHARS
    large_chunks: int = 0     # > char_limit

    overlap_chars: int = 0
    overlap_samples: int = 0

    #new
    messages_consumed = 0
    remainders_created = 0
    max_overlap_seen = 0

    chunks_per_file: Counter = field(default_factory=Counter)

    def record_chunk(self, chunk_len: int):
        self.total_chunks += 1
        self.total_chars += chunk_len
        self.min_chunk_size = min(self.min_chunk_size, chunk_len)
        self.max_chunk_size = max(self.max_chunk_size, chunk_len)

    def record_overlap(self, overlap_len: int):
        self.overlap_chars += overlap_len
        self.overlap_samples += 1

    def finalize(self):
        return {
            "total_chunks": self.total_chunks,
            "avg_chunk_size": (
                self.total_chars // self.total_chunks
                if self.total_chunks else 0
            ),
            "min_chunk_size": self.min_chunk_size if self.total_chunks else 0,
            "max_chunk_size": self.max_chunk_size,
            "small_chunks": self.small_chunks,
            "large_chunks": self.large_chunks,
            "avg_overlap_chars": (
                self.overlap_chars // self.overlap_samples
                if self.overlap_samples else 0
            ),
            "chunks_per_file": dict(self.chunks_per_file)
        }

from typing import List, Dict, Any, Tuple

from typing import List, Dict, Tuple, Any
import re
from typing import List, Dict, Tuple, Any


def build_text_chunks(
    conversation: List[Dict[str, Any]],
    system_message_indices: List[Tuple[int, int]],
    char_limit: int = 2000,
    stats: "ChunkStats | None" = None,
    file_id: str | None = None,
):
    """
    Build semantically stable text chunks from a conversation,
    excluding system messages, using bounded, message-aligned overlap.

    Overlap is explicitly marked so the AI can avoid duplicate memories.

    Args:
        conversation: List of dicts with 'content' field
        system_message_indices: List of (start_idx, end_idx) marking system messages
        char_limit: maximum characters per chunk
        stats: optional ChunkStats for diagnostics
        file_id: optional file identifier for diagnostics

    Returns:
        List[str]: chunks ready for AI
    """

    # ---------- configuration ----------
    MIN_CHUNK_CHARS = 500
    TARGET_CHUNK_CHARS = 1000
    MAX_CHUNK_CHARS = 2000

    MIN_OVERLAP_CHARS = 50
    MAX_OVERLAP_CHARS = 250

    MAX_CHUNKS = 500

    # ---------- helper ----------
    def is_system_message(idx: int) -> bool:
        for start, end in system_message_indices:
            if start <= idx <= end:
                return True
        return False

    # ---------- filter usable messages ----------
    usable_messages = [msg.get("content", "").strip()
                       for idx, msg in enumerate(conversation)
                       if not is_system_message(idx) and msg.get("content", "").strip()]
    #for msg in usable_messages:
        #print("DEBUG: {msg}".format(msg=msg))
    if not usable_messages:
        return []

    chunks: List[str] = []
    prev_overlap: List[str] = []  # carryover for overlap

    idx = 0
    print(f"Debug: len(usable_messages) = {len(usable_messages)}")
    while idx < len(usable_messages):
        print(f"Debug: idx = {idx} < len(usable_messages) = {len(usable_messages)}")
        if len(chunks) >= MAX_CHUNKS:
            print("⚠️ Chunk limit reached, stopping.")
            #append remaining content
            if idx < len(usable_messages):
                chunks.append("\n".join(usable_messages[idx:]))
            break
        oversize = False
        # ---------- start new chunk ----------
        current_messages: List[str] = list(prev_overlap)
        current_len = sum(len(m) + 1 for m in current_messages)
        start_idx = idx
        # Add messages until chunk is full
        while idx < len(usable_messages):
            msg = usable_messages[idx]
            msg_len = len(msg) + 1
            oversize = False

            if current_len + msg_len > MAX_CHUNK_CHARS + MAX_OVERLAP_CHARS:
                # --------- oversized message handling ---------
                if msg_len > MAX_CHUNK_CHARS:
                    oversize = True
                    space_left = MAX_CHUNK_CHARS - current_len
                    if space_left > 0:
                        # Take slice up to space left
                        slice_text = msg[:space_left]

                        # Find last sentence or line break
                        last_break_match = list(re.finditer(r'(\.|\n)', slice_text))
                        if last_break_match:
                            last_break = last_break_match[-1].end()
                        else:
                            last_break = space_left  # no break, cut at limit

                        # Add text up to last logical break to current chunk
                        current_messages.append(msg[:last_break])

                        # Remainder is everything after last break
                        remainder = msg[last_break:]
                    else:
                        remainder = msg  # no space left in current chunk

                    # Replace current message with remainder for next iteration
                    usable_messages[idx] = remainder.lstrip()
                    break  # assemble chunk now

                else:
                    # Message fits better in next chunk
                    break

            else:
                current_messages.append(msg)
                current_len += msg_len
                idx += 1

            if current_len >= TARGET_CHUNK_CHARS:
                break  # stop early if near target

        # assemble chunk
        if prev_overlap:
            overlap_text = "\n".join(prev_overlap)
            new_text = "\n".join(current_messages[len(prev_overlap):])
            chunk_text = (
                "<OVERLAP_CONTEXT>\n"
                "The following text may repeat information from a previous chunk.\n"
                "Do NOT extract duplicate memories unless new information is present.\n"
                "</OVERLAP_CONTEXT>\n\n"
                "<OVERLAP_TEXT>\n"
                f"{overlap_text}\n"
                "</OVERLAP_TEXT>\n\n"
                "<NEW_TEXT>\n"
                f"{new_text}\n"
                "</NEW_TEXT>"
            )
        else:
            chunk_text = "\n".join(current_messages)
        if idx == start_idx and oversize is False:
            # No new messages consumed — overlap-only chunk
            break
        chunks.append(chunk_text)

        # ---------- compute overlap ----------
        overlap_messages = []
        overlap_len = 0

        for m in reversed(current_messages):
            remaining = MAX_OVERLAP_CHARS - overlap_len
            if remaining <= 0:
                break

            m_len = len(m) + 1  # newline

            if m_len <= remaining:
                # Entire message fits
                overlap_messages.insert(0, m)
                overlap_len += m_len
                continue

            # --------- partial message overlap (semantic) ---------
            # Take up to remaining chars from the END, but expand to sentence boundary
            tail = m[-remaining:]

            # Find sentence endings inside tail
            sentence_matches = list(re.finditer(r"[\.!\?\n]", tail))

            if sentence_matches:
                # Choose the EARLIEST sentence boundary that keeps overlap large
                cut = sentence_matches[0].start()
                semantic_tail = tail[cut:].lstrip()
            else:
                semantic_tail = tail.lstrip()

            if len(semantic_tail) > 20:  # avoid punctuation-only or tiny overlaps
                overlap_messages.insert(0, semantic_tail)
                overlap_len += len(semantic_tail) + 1

            break  # partial message handled, stop overlap

        #remove non-ideal overlaps
        if overlap_len >= current_len * 0.8:
            #print("dominant overlap removed")
            overlap_messages = []
        elif overlap_len < MIN_OVERLAP_CHARS:
            overlap_messages = []

        if overlap_len > 0:
            print(f"Created overlap {overlap_len}")

        # ---------- diagnostics ----------
        if stats:
            stats.record_chunk(len(chunk_text))
            stats.record_overlap(overlap_len)
            stats.messages_consumed += (idx - start_idx)
            stats.remainders_created += 1
            stats.max_overlap_seen = max(stats.max_overlap_seen, overlap_len)
            if file_id:
                stats.chunks_per_file[file_id] += 1

        prev_overlap = overlap_messages

    # ---------- handle last small chunk ----------
    if len(chunks) >= 2 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        chunks[-2] += "\n\n" + chunks[-1]
        chunks.pop()

    print(f"Created {len(chunks)} conversation chunks")
    for chunk in chunks:
        print(f"Debug:chunk length: {len(chunk)}; START CHUNK:\n \n{chunk}\n END CHUNK\n")
    return chunks


"""
def build_raw_text_chunks(
    text: str,
    char_limit: int = 2000,
    stats=None,
    file_id: str | None = None,
):
    ""
    Build semantically stable chunks from raw text.

    - Explicit overlap signaling
    - Sentence-aligned chunking
    - Bounded overlap
    - No overlap-only chunks
    ""

    if not text.strip():
        return []

    # ---------- configuration ----------
    MIN_CHUNK_CHARS = 500
    TARGET_CHUNK_CHARS = 1200
    MAX_CHUNK_CHARS = char_limit

    MIN_OVERLAP_CHARS = 100
    MAX_OVERLAP_CHARS = 300

    MAX_CHUNKS = 500

    # ---------- sentence tokenization ----------
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    chunks: list[str] = []
    prev_overlap: list[str] = []

    idx = 0

    while idx < len(sentences):
        if len(chunks) >= MAX_CHUNKS:
            print("⚠️ Raw chunk limit reached, stopping.")
            break

        # ---------- start new chunk ----------
        current_sentences = list(prev_overlap)
        current_len = sum(len(s) + 1 for s in current_sentences)

        # ---------- fill chunk ----------
        while idx < len(sentences):
            sent = sentences[idx]
            sent_len = len(sent) + 1

            if current_len + sent_len > MAX_CHUNK_CHARS:
                # ---------- oversized sentence handling ----------
                if sent_len > MAX_CHUNK_CHARS:
                    space_left = MAX_CHUNK_CHARS - current_len

                    if space_left > 0:
                        slice_text = sent[:space_left]

                        # Prefer logical break
                        matches = list(re.finditer(r'[.!?\n]', slice_text))
                        if matches:
                            cut = matches[-1].end()
                        else:
                            cut = space_left

                        current_sentences.append(sent[:cut])
                        remainder = sent[cut:]
                    else:
                        remainder = sent

                    # Replace sentence with remainder and assemble chunk
                    sentences[idx] = remainder.strip()
                    break
                else:
                    break

            current_sentences.append(sent)
            current_len += sent_len
            idx += 1

            if current_len >= TARGET_CHUNK_CHARS:
                break

        # ---------- assemble chunk ----------
        if prev_overlap:
            overlap_text = " ".join(prev_overlap)
            new_text = " ".join(current_sentences[len(prev_overlap):])

            chunk_text = (
                "<OVERLAP_CONTEXT>\n"
                "The following text may repeat information from a previous chunk.\n"
                "Do NOT extract duplicate memories unless new information is present.\n"
                "</OVERLAP_CONTEXT>\n\n"
                "<OVERLAP_TEXT>\n"
                f"{overlap_text}\n"
                "</OVERLAP_TEXT>\n\n"
                "<NEW_TEXT>\n"
                f"{new_text}\n"
                "</NEW_TEXT>"
            )
        else:
            chunk_text = " ".join(current_sentences)

        if not chunk_text.strip():
            break

        chunks.append(chunk_text)

        # ---------- diagnostics ----------
        if stats:
            stats.record_chunk(len(chunk_text))
            if file_id:
                stats.chunks_per_file[file_id] += 1

        # ---------- compute sentence-aligned overlap ----------
        overlap_sentences = []
        overlap_len = 0

        for sent in reversed(current_sentences):
            sent_len = len(sent) + 1  # space/newline

            if overlap_len + sent_len > MAX_OVERLAP_CHARS:
                break

            overlap_sentences.insert(0, sent)
            overlap_len += sent_len

        # Enforce minimum overlap
        if overlap_len < MIN_OVERLAP_CHARS:
            overlap_sentences = []

        prev_overlap = overlap_sentences

    # ---------- merge final small chunk ----------
    if len(chunks) >= 2 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        chunks[-2] += "\n\n" + chunks[-1]
        chunks.pop()

    print(f"Created {len(chunks)} raw text chunks")
    for chunk in chunks:
        print(f"chunk length: {len(chunk)} start chunk:\n{chunk}\nend chunk:\n")
    return chunks
"""

import nltk
nltk.download("punkt", quiet=True)
from nltk.tokenize import sent_tokenize

def build_raw_text_chunks(
    text: str,
    char_limit: int = 2000,
    overlap_ratio: float = 0.1,
    min_overlap: int = 50,
    max_overlap: int = 300,
    *,
    min_chunk_chars: int = 500,
    stats: "ChunkStats | None" = None,
    file_id: str | None = None
) -> list[str]:
    """
    Build semantically stable chunks from raw text.

    - Explicit overlap signaling
    - Sentence-aligned chunking
    - Bounded overlap
    - No overlap-only chunks
    """

    if not text.strip():
        return []

    # ---------- configuration ----------
    MIN_CHUNK_CHARS = 500
    TARGET_CHUNK_CHARS = 1200
    MAX_CHUNK_CHARS = char_limit

    MIN_OVERLAP_CHARS = 100
    MAX_OVERLAP_CHARS = 300

    MAX_CHUNKS = 500

    # ---------- sentence tokenization ----------
    sentences = sent_tokenize(text)
    if not sentences:
        return []

    chunks: list[str] = []
    prev_overlap: list[str] = []

    idx = 0

    while idx < len(sentences):
        if len(chunks) >= MAX_CHUNKS:
            print("⚠️ Raw chunk limit reached, stopping.")
            break

        # ---------- start new chunk ----------
        current_sentences = list(prev_overlap)
        current_len = sum(len(s) + 1 for s in current_sentences)

        # ---------- fill chunk ----------
        while idx < len(sentences):
            sent = sentences[idx]
            sent_len = len(sent) + 1

            if current_len + sent_len > MAX_CHUNK_CHARS:
                # ---------- oversized sentence handling ----------
                if sent_len > MAX_CHUNK_CHARS:
                    space_left = MAX_CHUNK_CHARS - current_len

                    if space_left > 0:
                        slice_text = sent[:space_left]

                        # Prefer logical break
                        matches = list(re.finditer(r'[.!?\n]', slice_text))
                        if matches:
                            cut = matches[-1].end()
                        else:
                            cut = space_left

                        current_sentences.append(sent[:cut])
                        remainder = sent[cut:]
                    else:
                        remainder = sent

                    # Replace sentence with remainder and assemble chunk
                    sentences[idx] = remainder.strip()
                    break
                else:
                    break

            current_sentences.append(sent)
            current_len += sent_len
            idx += 1

            if current_len >= TARGET_CHUNK_CHARS:
                break

        # ---------- assemble chunk ----------
        if prev_overlap:
            overlap_text = " ".join(prev_overlap)
            new_text = " ".join(current_sentences[len(prev_overlap):])

            chunk_text = (
                "<OVERLAP_CONTEXT>\n"
                "The following text may repeat information from a previous chunk.\n"
                "Do NOT extract duplicate memories unless new information is present.\n"
                "</OVERLAP_CONTEXT>\n\n"
                "<OVERLAP_TEXT>\n"
                f"{overlap_text}\n"
                "</OVERLAP_TEXT>\n\n"
                "<NEW_TEXT>\n"
                f"{new_text}\n"
                "</NEW_TEXT>"
            )
        else:
            chunk_text = " ".join(current_sentences)

        if not chunk_text.strip():
            break

        chunks.append(chunk_text)

        # ---------- diagnostics ----------
        if stats:
            stats.record_chunk(len(chunk_text))
            if file_id:
                stats.chunks_per_file[file_id] += 1

        # ---------- compute sentence-aligned overlap ----------
        overlap_sentences = []
        overlap_len = 0

        for sent in reversed(current_sentences):
            sent_len = len(sent) + 1  # space/newline

            if overlap_len + sent_len > MAX_OVERLAP_CHARS:
                break

            overlap_sentences.insert(0, sent)
            overlap_len += sent_len

        # Enforce minimum overlap
        if overlap_len < MIN_OVERLAP_CHARS:
            overlap_sentences = []

        prev_overlap = overlap_sentences

    # ---------- merge final small chunk ----------
    if len(chunks) >= 2 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        chunks[-2] += "\n\n" + chunks[-1]
        chunks.pop()

    # ---------- post-chunk diagnostics ----------
    if stats:
        total_text_len = len(text)
        total_chunk_len = 0

        for chunk in chunks:
            chunk_len = len(chunk)
            total_chunk_len += chunk_len

            stats.record_chunk(chunk_len)

            if chunk_len < min_chunk_chars:
                stats.small_chunks += 1
            if chunk_len > char_limit:
                stats.large_chunks += 1

        #revise class to include
        #inflated = total_chunk_len - total_text_len
        #stats.total_inflated_chars += max(inflated, 0)

        if file_id:
            stats.chunks_per_file[file_id] += len(chunks)
    if stats and file_id:
        print(
            f"[RAW CHUNKS] {file_id}: "
            f"{len(chunks)} chunks | "
            f"small={stats.small_chunks} | "
            f"large={stats.large_chunks}"
        )
    return chunks


#helper for generate_memories
def check_mem_source(file_path: str) -> Dict[str, Any]:
    """
    Check whether a file is a valid conversation JSON file and
    identify system-message index ranges.

    Returns:
        {
            "is_valid": bool,
            "reason": str | None,
            "conversation": list | None,
            "system_message_indices": list[tuple[int, int]]
        }
    """
    path = Path(file_path)

    # 1. Load JSON
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        return {
            "is_valid": False,
            "reason": f"invalid_json: {e}",
            "conversation": None,
            "system_message_indices": []
        }
    except Exception as e:
        return {
            "is_valid": False,
            "reason": f"file_error: {e}",
            "conversation": None,
            "system_message_indices": []
        }

    # 2. Validate conversation structure
    if not isinstance(data, dict) or "conversation" not in data:
        return {
            "is_valid": False,
            "reason": "missing_conversation_key",
            "conversation": None,
            "system_message_indices": []
        }

    conversation = data["conversation"]
    if not isinstance(conversation, list):
        return {
            "is_valid": False,
            "reason": "conversation_not_list",
            "conversation": None,
            "system_message_indices": []
        }

    # 3. Validate messages + collect system ranges
    system_ranges: List[Tuple[int, int]] = []
    current_start = None

    for idx, msg in enumerate(conversation):
        if not isinstance(msg, dict):
            return {
                "is_valid": False,
                "reason": f"invalid_message_format_at_{idx}",
                "conversation": None,
                "system_message_indices": []
            }

        role = msg.get("role")
        content = msg.get("content")

        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return {
                "is_valid": False,
                "reason": f"invalid_message_fields_at_{idx}",
                "conversation": None,
                "system_message_indices": []
            }

        if role == "system":
            if current_start is None:
                current_start = idx
        else:
            if current_start is not None:
                system_ranges.append((current_start, idx - 1))
                current_start = None

    if current_start is not None:
        system_ranges.append((current_start, len(conversation) - 1))

    return {
        "is_valid": True,
        "reason": None,
        "conversation": conversation,
        "system_message_indices": system_ranges
    }

#for generate memories' prompt
def load_prompt(prompt_path: str, persona: str, tags_list: list, chunk_text: str) -> str:
    """
    Load the extraction prompt from a text file and replace placeholders.
    """
    #print("Debug: Loading prompt...")
    #print(f"Debug: Prompt parameters: path: {prompt_path} , persona: {persona} , tags_list: {tags_list}, chunk_text: {chunk_text}")
    path = Path(prompt_path)
    if not path.exists():
        print("Prompt not found")
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    #print("Debug: opening prompt file...")
    with open(path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    #print(f"Debug: prompt loaded: {prompt_template}")
    #print("Debug: replacing placeholders...")
    # Replace placeholders - did not work
    """
    prompt_filled = prompt_template.format(
        persona=persona,
        tags_list=json.dumps(tags_list),
        chunk_text=chunk_text
    )
    """
    prompt_template = Template(prompt_text)
    prompt_filled = prompt_template.substitute(
        persona=persona,
        tags_list=json.dumps(tags_list),
        chunk_text=chunk_text
    )
    print(f"Debug: Prompt loaded")
    return prompt_filled

def repair_tags(memory: dict, allowed_tags: list, invalid_tag_stats) -> bool:
    """
    Repairs tags and primary_tag in-place.
    Returns True if memory is salvageable.
    """
    original_tags = memory.get("tags", [])
    valid_tags = []
    invalid_tags = []

    for t in original_tags:
        if t in allowed_tags:
            valid_tags.append(t)
        else:
            invalid_tags.append(t)

    for bad_tag in invalid_tags:
        stat = invalid_tag_stats[bad_tag]
        stat["count"] += 1
        stat["memory_types"][memory["memory_type"]] += 1

    if not valid_tags:
        memory["tags"] = []
        memory["primary_tag"] = ""
        return False

    memory["tags"] = valid_tags

    if memory.get("primary_tag") not in valid_tags:
        repaired_to = valid_tags[0]
        for bad_tag in invalid_tags:
            invalid_tag_stats[bad_tag]["repaired_to"][repaired_to] += 1
        memory["primary_tag"] = repaired_to

    return True

#helper for report tag normalization for json dump
def normalize_stats(stats):
    return {
        tag: {
            "count": data["count"],
            "repaired_to": dict(data["repaired_to"]),
            "memory_types": dict(data["memory_types"]),
        }
        for tag, data in stats.items()
    }

CHAT_DATE_RE = re.compile(r"chat_(\d{4})_(\d{2})_(\d{2})_\d{2}_\d{2}_\d{2}\.json")

#helper that calls both file date functions to improve readability
def resolve_memory_date(file_path: str, is_conversation: bool) -> str:
    if is_conversation:
        date = get_date_from_chat_filename(file_path)
        if date:
            return date
    return get_date_from_mtime(file_path)

#conversation file
def get_date_from_chat_filename(file_path: str) -> str | None:
    name = Path(file_path).name
    match = CHAT_DATE_RE.match(name)
    if not match:
        return None

    year, month, day = match.groups()
    return f"{year}-{month}-{day}"

#return file modified time (non-conversation file)
def get_date_from_mtime(file_path: str) -> str:
    ts = os.path.getmtime(file_path)
    return datetime.fromtimestamp(ts).date().isoformat()

#for generate_memories limit checks
def _normalize_limits(max_files, max_runtime_seconds):
    resolved_max_files = (
        DEFAULT_MAX_FILES if max_files is None
        else min(max_files, HARD_MAX_FILES)
    )

    resolved_max_runtime = (
        DEFAULT_MAX_RUNTIME_SECONDS if max_runtime_seconds is None
        else min(max_runtime_seconds, HARD_MAX_RUNTIME_SECONDS)
    )

    if resolved_max_files <= 0:
        raise ValueError("max_files must be > 0")

    if resolved_max_runtime <= 0:
        raise ValueError("max_runtime_seconds must be > 0")

    return resolved_max_files, resolved_max_runtime


if __name__ == "__main__":
    paths = generate_search_path(".")
    print(f"Found {len(paths)} files:")
    for p in paths[:10]:
        print(p)