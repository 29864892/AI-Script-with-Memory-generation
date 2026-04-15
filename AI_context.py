"""
    Functionality related to AI context calls and updates
    Desired Functionality:
        1.Send AI list of current context files with summaries - Done in load_conversation()
        2.Allow AI to search through specific context files as needed - Done in getContext()
        3.Allow AI to change content of context files:
            1.Update entries - update context
            2.Add entries
"""
from openai import OpenAI
import json #Create JSON files for conversations
import os
import re
import hashlib #for checking for duplicates in json
from datetime import datetime
import os
import copy


def getContext(filename, keyword, base_dir="conversations", max_chars=3000, context_lines = 2):
    """
    Searches a context file (JSON or text) for keyword matches.
    If keyword == "all", returns the entire file (safely truncated).
    """

    filepath = resolve_conversation_path(filename, base_dir)
    if not filepath:
        return f"⚠️ Context file '{filename}' not found in '{base_dir}'."

    ext = os.path.splitext(filepath)[1].lower()

    # ------------------------------------------------
    # TEXT FILES
    # ------------------------------------------------
    if ext != ".json":
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            return f"⚠️ Error reading '{filename}': {e}"

        if keyword.lower() == "all":
            return (
                f"📄 Context file: {filename}\n"
                f"Type: Text\n\n"
                f"{content[:max_chars]}"
                + ("\n\n... (truncated)" if len(content) > max_chars else "")
            )

        # Keyword search in text
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        lines = content.splitlines()
        matches = []

        used_indices = set()

        for i, line in enumerate(lines):
            if pattern.search(line):
                start = max(0, i - context_lines)
                end = min(len(lines), i + context_lines + 1)

                block = []
                for j in range(start, end):
                    if j not in used_indices:
                        block.append(f"{j + 1}: {lines[j]}")
                        used_indices.add(j)

                matches.append("\n".join(block))

        if not matches:
            return f"🔍 No matches found for '{keyword}' in '{filename}'."

        return (
                f"📄 Context matches in {filename} (keyword: '{keyword}'):\n\n"
                + "\n\n---\n\n".join(matches)
        )

    # ------------------------------------------------
    # JSON FILES
    # ------------------------------------------------
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return f"⚠️ Error: '{filename}' is not valid JSON."

    #if not isinstance(data, list):
     #   return f"⚠️ '{filename}' is not a list-based context file."
    #
    #Normalize structure
    if isinstance(data, dict) and "entries" in data:
        entries = data["entries"]
        template = data.get("template")
    elif isinstance(data, list):
        # Backward compatibility
        entries = data
        template = None
    else:
        return f"⚠️ '{filename}' is not a valid context file."

    # Return entire JSON
    if keyword.lower() == "all":
        return (
            f"📂 Full context file: {filename}\n"
            f"Total entries: {len(entries)}\n\n"
            f"{json.dumps(entries, indent=2, ensure_ascii=False)}"
        )
    elif keyword.lower() == "template":
        if template:
            return {
                "type": "template",
                "file": filename,
                "template": template
            }
        else:
            return f"⚠️ '{filename}' does not define a template."
    # Keyword search
    pattern = re.compile(re.escape(keyword), re.IGNORECASE)
    matches = []

    for entry in entries:
        entry_text = json.dumps(entry, ensure_ascii=False)
        if pattern.search(entry_text):
            matches.append(entry)

    if not matches:
        return f"🔍 No matches found for '{keyword}' in '{filename}'."

    return (
        f"📂 Context matches in {filename} (keyword: '{keyword}'):\n"
        f"Matches found: {len(matches)}\n\n"
        f"{json.dumps(matches, indent=2, ensure_ascii=False)}"
    )

def handle_getContext_command(user_input):
    """
    Detects and executes a getContext(filename, keyword) command in user input.
    Returns the output of getContext() or None if not a valid command.
    """

    # Match: getContext(filename, keyword)
    # Allows optional quotes, spaces, and case-insensitivity
    pattern = re.compile(
        r"getContext\s*\(\s*['\"]?([^,'\"\)]+)['\"]?\s*,\s*['\"]?([^,'\"\)]+)['\"]?\s*\)",
        re.IGNORECASE
    )

    match = pattern.search(user_input)
    if not match:
        return None  # Not a getContext call

    filename = match.group(1).strip()
    keyword = match.group(2).strip()

    # Call your existing getContext function
    try:
        result = getContext(filename, keyword)
    except Exception as e:
        result = f"⚠️ Error running getContext({filename}, {keyword}): {e}"

    return result

def handle_updateContext_command(text):
    """
    Detects and prepares proposed edits from AI.
    Returns (handled, action_dict) — action_dict includes all details.
    """
    pattern = re.compile(
        r"updateContext\s*\(\s*['\"]?([^,'\"\)]+)['\"]?\s*,\s*(\{.*\})\s*\)",
        re.IGNORECASE | re.DOTALL
    )
    match = pattern.search(text)
    if not match:
        return False, None

    filename = match.group(1).strip()
    json_data_str = match.group(2).strip()

    try:
        new_data = json.loads(json_data_str)
    except Exception as e:
        return True, {"error": f"⚠️ Invalid JSON format in updateContext: {e}"}

        # Ensure path inside conversations/ - TODO: create separate folder for context files, and update this
    #base_dir = "conversations"
    #os.makedirs(base_dir, exist_ok=True)
    #if not filename.startswith(base_dir):
        #filename = os.path.join(base_dir, filename)

        #replaces commented code above to accommodate folders in conversations directory
    filename = resolve_conversation_path(filename)

    return True, {"filename": filename, "new_data": new_data}

#WIP 2

#conversations\Context\staffContext.json
CONTEXT_DIR = "Conversations/Context"

def AI_proposal(proposal: dict, base_dir="Conversations/Context"):
    """
    Handles AI-proposed edits/additions to context JSON files.
    Requires user confirmation before applying changes.
    """
    filepath = resolve_conversation_path(proposal["filename"])
    print("Debug: ", filepath)
    #filepath = os.path.join(CONTEXT_DIR, proposal["filename"])
    action = proposal["action"]
    new_entry = proposal["new_entry"]
    match_criteria = proposal.get("match_criteria")

    #For new file creation
    if filepath is None:
        if action != "add":
            print(f"❌ Cannot perform '{action}' — context file does not exist.")
            return

        print("\n📁 NEW CONTEXT FILE DETECTED")
        print(f"File: {proposal['filename']}")
        print("\n🧩 Proposed TEMPLATE:")
        print(json.dumps(new_entry, indent=2))

        if input("\nCreate new context file with this template? (y/n): ").lower() != "y":
            print("❌ File creation cancelled.")
            return

        context = {
            "template": new_entry,
            "entries": []
        }
        new_file = CONTEXT_DIR + "/" + proposal["filename"]
        with open(new_file, "w", encoding="utf-8") as f:
            json.dump(context, f, indent=2)

        print(f"✅ New context file '{proposal['filename']}' created.")
        return
    # Load or initialize context file
    elif os.path.exists(filepath):
        with open(filepath, "r", encoding="utf-8") as f:
            context = json.load(f)
    else:
        #Create new context file and return
        print("ERROR: file not handled properly: ", filepath)
        return


    template_keys = set(context["template"].keys())
    entry_keys = set(new_entry.keys())

    # Template compatibility check
    if template_keys != entry_keys:
        print("⚠️ Proposed entry does not match template.")
        print("Template keys:", template_keys)
        print("Proposed keys:", entry_keys)
        return

    # Handle ADD
    if action == "add":
        print("\n📌 PROPOSED ADDITION:")
        print(json.dumps(new_entry, indent=2))

        if input("\nConfirm add? (y/n): ").lower() == "y":
            context["entries"].append(new_entry)
        else:
            print("❌ Addition cancelled.")
            return

    # Handle EDIT
    elif action == "edit":
        if not match_criteria:
            raise ValueError("Edit action requires match_criteria")

        keyword = next(iter(match_criteria.values()))
        target = None

        for entry in context["entries"]:
            if keyword.lower() in json.dumps(entry).lower():
                target = entry
                break

        if not target:
            print("❌ No matching entry found.")
            return

        before = copy.deepcopy(target)

        print("\n✏️ PROPOSED EDIT:")
        print("\n--- BEFORE ---")
        print(json.dumps(before, indent=2))
        print("\n--- AFTER ---")
        print(json.dumps(new_entry, indent=2))

        if input("\nConfirm edit? (y/n): ").lower() == "y":
            context["entries"].remove(target)
            context["entries"].append(new_entry)
        else:
            print("❌ Edit cancelled.")
            return

    else:
        raise ValueError("Invalid action type")

    # Save changes
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(context, f, indent=2)

    print("✅ Context updated successfully.")


PROPOSAL_START = "<<PROPOSE_CONTEXT_UPDATE>>"
PROPOSAL_END = "<<END>>"

def check_proposal(ai_text: str) -> dict | None:
    """
    Extracts and validates a proposed context update from an AI response.
    Returns the proposal dict or None if no valid proposal is found.
    """

    if PROPOSAL_START not in ai_text or PROPOSAL_END not in ai_text:
        return None

    pattern = re.escape(PROPOSAL_START) + r"(.*?)" + re.escape(PROPOSAL_END)
    match = re.search(pattern, ai_text, re.DOTALL)

    if not match:
        return None

    raw_json = match.group(1).strip()

    try:
        proposal = json.loads(raw_json)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in proposal: {e}")

    required_keys = {"filename", "action", "new_entry"}
    if not required_keys.issubset(proposal):
        raise ValueError("Proposal missing required fields")

    return proposal

#WIP #2



#WIP #2 END
def review_context_update(action):
    """Ask the user to approve or reject an AI-suggested context update."""
    if "error" in action:
        print(action["error"])
        return

    filename = action["filename"]
    new_data = action["new_data"]

    print(f"\n🤖 Proposed context update for '{filename}':")
    print(json.dumps(new_data, indent=2, ensure_ascii=False))

    confirm = input("\nApply this change? (y/n): ").strip().lower()
    if confirm == "y":
        if not os.path.exists(filename):
            print(f"📄 '{filename}' does not exist. Creating new file...")
            create_context_file(filename, new_data)
        else:
            append_to_context_file(filename, new_data)
    else:
        print("❌ Change discarded.")

def create_context_file(filename, data):
    """Create a new context file with a template and optional first entry."""
    # Build initial data array
    if isinstance(data, dict) and "template" in data:
        file_data = [data["template"]]
        if "new_entry" in data:
            file_data.append(data["new_entry"])
    else:
        # Fallback: treat it as a single entry
        file_data = [data]

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(file_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Created new context file: {filename}")

def append_to_context_file(filename, new_entry):
    """Append a new JSON object to an existing context file."""
    try:
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            data.append(new_entry)
        else:
            data = [data, new_entry]
    except Exception:
        data = [new_entry]

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"📝 Appended new entry to {filename}")

def resolve_conversation_path(filename, base_dir="conversations"):
    """
    Resolve a filename safely within the conversations directory.
    Supports subdirectories like Context/, backup/, etc.
    Prevents path traversal.
    """

    # Normalize separators
    filename = filename.strip().replace("\\", "/")

    # Prevent path traversal
    if ".." in filename or filename.startswith("/"):
        return None

    # check if already resolved
    if "conversations" in filename:
        if os.path.exists(filename):
            return filename
        else:
            print(
                "DEBUG - resolve_conversation_path: file is said to be in conversations directory, but does not exist")
            return None

    # Direct relative path (e.g., Context/file.json)
    candidate = os.path.join(base_dir, filename)
    if os.path.exists(candidate):
        return candidate

    # Bare filename → search subdirectories
    for root, _, files in os.walk(base_dir):
        if filename in files:
            return os.path.join(root, filename)

    return None

def AIRead_raw(filename, base_dir="PythonProject"):
    """
    Returns raw structured data (dict or str).
    Use AIRead for structured data
    """
    print("DEBUG: Running AIRead_raw")
    filepath = (resolve_conversation_path(filename, base_dir))
    print("filepath = ", filepath)
    if filepath.lower().endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            print("DEBUG: Reading JSON...")
            return json.load(f)
    else:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()

def AIRead(filename, max_items=20, max_chars=5000):
    """
    Primarily usage is to send files to AI and provide relevant info
    Read JSON or text files anywhere inside conversations/.
    PROBLEM: AVOID USING THIS WHEN ONLY A JSON IS WANTED
    Returns an object:
        filename
        filepath
        Type
        Top-level keys
        Json Object
    """

    filepath = resolve_conversation_path(filename)
    if not filepath:
        return f"⚠️ AIRead error: '{filename}' not found in conversations directory."

    ext = os.path.splitext(filepath)[1].lower()

    # -------------------------------
    # JSON FILES
    # -------------------------------
    if ext == ".json":
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return f"⚠️ AIRead error: '{filename}' contains invalid JSON."

        if isinstance(data, list):
            output = [
                f"📂 AIRead: {filename}",
                f"Location: {filepath}",
                "Type: JSON List",
                f"Total entries: {len(data)}",
                ""
            ]

            for i, entry in enumerate(data[:max_items]):
                output.append(f"Entry {i + 1}:")
                output.append(json.dumps(entry, indent=2, ensure_ascii=False))
                output.append("")

            if len(data) > max_items:
                output.append(f"... {len(data) - max_items} more entries not shown.")

            return "\n".join(output)

        elif isinstance(data, dict):
            return (
                f"📂 AIRead: {filename}\n"
                f"Location: {filepath}\n"
                f"Type: JSON Object\n"
                f"Top-level keys: {', '.join(data.keys())}\n\n"
                f"{json.dumps(data, indent=2, ensure_ascii=False)}"
            )

        else:
            return f"⚠️ AIRead warning: Unsupported JSON structure in '{filename}'."

    # -------------------------------
    # TEXT FILES
    # -------------------------------
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        return f"⚠️ AIRead error: Could not read '{filename}': {e}"

    preview = content[:max_chars]

    result = (
        f"📄 AIRead: {filename}\n"
        f"Location: {filepath}\n"
        f"Type: Text\n"
        f"Total characters: {len(content)}\n\n"
        f"Preview:\n{preview}"
    )

    if len(content) > max_chars:
        result += "\n\n... (content truncated)"

    return result