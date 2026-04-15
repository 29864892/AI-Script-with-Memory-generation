"""
    Code for implementing dynamic conversation coordination - for when more than 1 persona is present
    Step 1: Build coordinator prompt

        This is not merged with persona prompts.

        COORDINATOR SYSTEM PROMPT =
        [Core System Constraints]
        +
        [Coordinator Instructions]
    Step 2: For EACH persona response turn

        You dynamically assemble:

        PERSONA SYSTEM PROMPT =
        [Core System Constraints]
        +
        [Persona System Prompt (file)]
        +
        [Roles Injection]
        +
        [Coordinator-Imposed Constraints (TEMPORARY)]
    coordinator output: MUST match exactly, with no extra information
        {
          "conversation_mode": "<mode>",
          "responders": [<persona_name>, ...],
          "response_order": [<persona_name>, ...],
          "persona_constraints": {
            "<persona_name>": [<constraint>, ...]
          },
          "meta_message": <string or null>
        }
"""

import json
from typing import List, Dict, Any, Optional
import modelManagement as mM
from modelManagement import get_effective_context_limit, auto_prune_messages_enhanced

ALLOWED_CONVERSATION_MODES = {
    "casual",
    "planning",
    "analysis",
    "decision",
    "brainstorming",
    "clarification_required",
    "precision",
    "creative"
}
#coordinator_decision = cc.persona_coordination(coordinator_prompt, user_input, active_personas, messages_context, client, model_name)
def run_coordinator(client, model_name, coordinator_prompt: str, user_message:str, personas: List[Dict[str,Any]], conversation_state: Optional[Dict[str, Any]] = None, max_retries: int = 2
)->Dict[str, Any]:
    """
        Run the invisible coordinator to determine response routing.

        Returns a validated decision object:
        {
          "conversation_mode": str,
          "responders": [persona_name, ...],
          "response_order": [persona_name, ...],
          "persona_constraints": { persona_name: [constraint, ...] },
          "meta_message": str | None
        }
        """

    persona_names = {p["persona_name"] for p in personas}

    # ---------------------------
    # Build coordinator input
    # ---------------------------

    coordinator_input = {
        "user_message": user_message,
        "personas": [
            {
                "persona_name": p["persona_name"],
                "roles": p.get("roles", []),
                "permissions": p.get("permissions", []),
                "rank": p.get("rank"),
                "performance_points": p.get("performance_points"),
            }
            for p in personas
        ],
        "conversation_state": conversation_state or {},
    }
    print("coordinator_input: ", coordinator_input)
    messages = [
        {"role": "system", "content": coordinator_prompt},
        {
            "role": "user",
            "content": (
                "Produce a coordination decision object ONLY.\n\n"
                "INPUT CONTEXT:\n"
                f"{json.dumps(coordinator_input, indent=2)}\n\n"
                "REMINDER:\n"
                "- Output MUST be valid JSON\n"
                "- Do NOT include explanations\n"
                "- Do NOT include markdown\n"
            ),
        },
    ]

    last_error = None
    max_tokens = get_effective_context_limit(model_name)
    cur_tokens = mM.estimate_tokens(messages)
    print("cur_tokens before coordinator call: ", cur_tokens)
    if cur_tokens > max_tokens:
        messages_context, prune_info = auto_prune_messages_enhanced(messages, model_name, client)
        if prune_info["freed"] > 0:
            print(
                f"Context pruned: freed ~{prune_info['freed']} tokens \n"
                f"({prune_info['before']} -> {prune_info['after']})"
                f"Waiting for AI Response..."
            )
    # ---------------------------
    # Attempt / Retry Loop
    # ---------------------------
    #print("Attempting to get response")
    for attempt in range(max_retries + 1):
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.2,  # low creativity for structure
        )

        raw_text = response.choices[0].message.content.strip()
        #print("Received response: ", raw_text)
        # ---------------------------
        # Parse JSON
        # ---------------------------

        try:
            decision = json.loads(raw_text)
        except json.JSONDecodeError as e:
            last_error = f"Invalid JSON: {e}"
            messages.append({
                "role": "user",
                "content": f"ERROR: {last_error}\nRe-output ONLY valid JSON."
            })
            continue

        # ---------------------------
        # Validate structure
        # ---------------------------
        print("validating decision")
        error = _validate_coordinator_decision(decision, persona_names)

        if error:
            print("Coordinator generated an invalid decision: ", decision)
            print("Error message: ", last_error)
            last_error = error
            messages.append({
                "role": "user",
                "content": f"ERROR: {error}\nFix and re-output ONLY the corrected JSON."
            })
            continue
        print("decision successful!")
        # ---------------------------
        # Success
        # ---------------------------
        return decision

    # ---------------------------
    # Hard failure
    # ---------------------------

    raise RuntimeError(
        "Coordinator failed to produce a valid decision object.\n"
        f"Last error: {last_error}"
    )

def _validate_coordinator_decision(
    decision: Dict[str, Any],
    valid_persona_names: set,
) -> Optional[str]:
    """
    Returns None if valid, otherwise an error string.
    """

    required_keys = {
        "conversation_mode",
        "responders",
        "response_order",
        "persona_constraints",
        "meta_message",
    }

    missing = required_keys - decision.keys()
    if missing:
        return f"Missing required keys: {missing}"

    if decision["conversation_mode"] not in ALLOWED_CONVERSATION_MODES:
        return f"Invalid conversation_mode: {decision['conversation_mode']}"

    responders = decision["responders"]
    order = decision["response_order"]

    if not isinstance(responders, list) or not responders:
        return "responders must be a non-empty list"

    if not isinstance(order, list) or not order:
        return "response_order must be a non-empty list"

    if set(order) != set(responders):
        return "response_order must contain the same personas as responders"

    if not set(responders).issubset(valid_persona_names):
        return "responders contains unknown persona(s)"

    constraints = decision["persona_constraints"]
    if not isinstance(constraints, dict):
        return "persona_constraints must be an object"

    for name, rules in constraints.items():
        if name not in valid_persona_names:
            return f"persona_constraints references unknown persona: {name}"
        if not isinstance(rules, list):
            return f"Constraints for {name} must be a list"

    if decision["meta_message"] is not None and not isinstance(
        decision["meta_message"], str
    ):
        return "meta_message must be string or null"

    return None

def update_persona_prompt(
    base_sys_prompt: str,
    constraints: list[str],
) -> str:
    """
    Inject temporary, per-response constraints into a persona system prompt.

    Constraints are ephemeral and apply ONLY to the next response.
    """

    if not constraints:
        return base_sys_prompt

    translated = []

    for c in constraints:
        if c.startswith("max_chars:"):
            limit = c.split(":", 1)[1]
            translated.append(
                f"- Limit your response to approximately {limit} characters."
            )

        elif c == "avoid_repetition":
            translated.append(
                "- Do not restate or closely mirror points already made by other personas."
            )

        elif c == "no_new_arguments":
            translated.append(
                "- Do not introduce new primary arguments or reframe the task."
            )

        elif c == "build_on_previous":
            translated.append(
                "- Build directly on the discussion so far rather than restarting analysis."
            )

        elif c == "ambient_only":
            translated.append(
                "- You may add emotional, environmental, or reflective commentary only."
            )

        elif c == "precision_mode":
            translated.append(
                "- Be precise. Clearly qualify uncertainty and avoid speculation."
            )

        else:
            # Fallback: safe, generic phrasing
            translated.append(f"- Constraint: {c.replace('_', ' ')}.")

    constraint_block = (
        "\n\n"
        "────────────────────────────\n"
        "TEMPORARY RESPONSE CONSTRAINTS\n"
        "(APPLY ONLY TO THIS RESPONSE)\n"
        "────────────────────────────\n"
        + "\n".join(translated)
    )

    return base_sys_prompt + constraint_block

def get_next_persona(response_order, current_persona):
    """
    Returns:
      - next persona name (str) if one exists
      - None if current_persona is the last responder
    """
    try:
        idx = response_order.index(current_persona)
    except ValueError:
        return None  # safety fallback

    if idx + 1 < len(response_order):
        return response_order[idx + 1]

    return None

#Removes only system message labelled as identity anchor
def remove_identity_anchor(context_window):
    return [
        msg for msg in context_window
        if not (msg["role"] == "system" and msg.get("meta", {}).get("type") == "identity_anchor")
    ]

import re

def refine_user_input(user_input, active_personas, persona_names):
    """
    Parses user input into persona-targeted segments.

    Args:
        user_input (str): raw user message
        active_personas (list[str]): persona names (EXCLUDES coordinator)

    Returns:
        dict with keys:
            - "routed_input": dict[str, str]
            - "error": None or error message
    """

    persona_set = set(persona_names)

    # Regex: @{persona_name}
    tag_pattern = re.compile(r'@\{([^}]+)\}')

    matches = list(tag_pattern.finditer(user_input))

    routed = {p: [] for p in active_personas}
    routed["all"] = []

    # No tags → everything goes to @all
    if not matches:
        routed["all"].append(user_input.strip())
        return {"routed_input": finalize_routed(routed), "error": None}

    # Append sentinel end
    spans = []
    for i, match in enumerate(matches):
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(user_input)
        spans.append((match.group(1).strip(), user_input[start:end].strip()))

    # Handle text before first tag
    if matches[0].start() > 0:
        preamble = user_input[:matches[0].start()].strip()
        if preamble:
            routed["all"].append(preamble)

    # Route tagged spans
    for persona_name, text in spans:
        if persona_name not in persona_set:
            return {
                "routed_input": None,
                "error": (
                    f"Unknown persona tag '@{{{persona_name}}}'.\n\n"
                    f"Available personas: {', '.join(sorted(persona_set))}\n"
                    "Please correct the tag and try again.\n"
                    "Reminder: the coordinator cannot be tagged."
                )
            }
        routed[persona_name].append(text)

    return {"routed_input": finalize_routed(routed), "error": None}

def finalize_routed(routed):
    """
    Joins routed segments into clean strings.
    Removes empty entries.
    """
    return {
        k: "\n\n".join(v).strip()
        for k, v in routed.items()
        if v and any(seg.strip() for seg in v)
    }
