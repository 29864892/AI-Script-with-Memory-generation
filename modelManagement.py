"""
    Functionality related to maintaining model integrity during runtime
"""
MODEL_CONTEXT_LIMITS = {
    "mistralai/magistral-small-2509": 4000,
    "llama-3-8b": 8192,
    "llama-3-70b": 8192,
    "meta-llama-3.1-8b-instruct": 17000,
    "ministral-3-8b-instruct-2512@q5_k_m": 7000
}

#never exceed 80% of context -> changed to 75% to test increased stability (update: was too low, prune calls too frequent)
SAFETY_MARGIN = 0.80
#when 80% is exceeded, prune to 50% of context limit (to test and adjust)
PRUNE_RATIO = 0.50

def get_effective_context_limit(model_name):
    limit = MODEL_CONTEXT_LIMITS.get(model_name, 17000)
    return int(limit * SAFETY_MARGIN)

#Rough, conservative estimate of word count * 1.3
def estimate_tokens(messages):
    total = 0
    for msg in messages:
        total += int(len(msg["content"].split()) * 1.3)
    return total

#Rough, conservative estimate of word count checking a string
def estimate_message_tokens(message):
    return int(len(message)*1.3)

#Automatically prune conversations to limit context overload
#Returns the pruned messages list and a report
def auto_prune_messages(messages, model_name):
    max_tokens = get_effective_context_limit(model_name)
    before_tokens = estimate_tokens(messages)
    if before_tokens < max_tokens:
        return messages,    {"before": before_tokens, "after": before_tokens, "freed": 0}
    # Always keep the first system message
    system_msgs = [m for m in messages if m["role"] == "system"][:1]

    # Separate other messages
    non_system = [m for m in messages if m not in system_msgs]

    # Start with everything
    pruned = system_msgs + non_system
    ideal_tokens = max_tokens * PRUNE_RATIO
    while estimate_tokens(pruned) > ideal_tokens and len(non_system) > 1:
        # Remove oldest non-system message
        non_system.pop(0)
        pruned = system_msgs + non_system

    after_tokens = estimate_tokens(pruned)

    return pruned, {"before": before_tokens, "after": after_tokens, "freed": max(0, before_tokens - after_tokens)}

#Automatically prune conversations to limit context overload
#Returns the pruned messages list and a report
#WIP - want to prune extra information while maintaining important context information
def auto_prune_messages_enhanced(messages, model_name, client):
    max_tokens = get_effective_context_limit(model_name)
    before_tokens = estimate_tokens(messages)
    if before_tokens < max_tokens:
        return messages, {"before": before_tokens, "after": before_tokens, "freed": 0, "summarized_messages": 0}
    # Always keep the first system message
    system_msgs = [m for m in messages if m["role"] == "system" and not is_session_state(m)][:1]
    session_msgs = [m for m in messages if is_session_state(m)]
    # Separate other messages
    non_system = [
        m for m in messages
        if m not in system_msgs and m not in session_msgs]
    # Start with everything
    pruned = system_msgs + non_system
    # Messages to be summarized by AI
    popped_messages = []
    #avoid constantly being near token limit ~2600 tokens
    ideal_tokens = max_tokens*0.80
    while estimate_tokens(pruned) > ideal_tokens and len(non_system) > 1:
        # Remove oldest non-system message
        popped = non_system.pop(0)
        popped_messages.append(popped)
        pruned = system_msgs + non_system

    after_tokens = estimate_tokens(pruned)
    #calculate tokens freed and summarize if 200 or more
    freed_tokens = max(0, before_tokens - after_tokens)
    if freed_tokens > 200:
        #import summary prompt
        popped_messages = format_messages_for_summary(popped_messages)
        #TODO - create dedicated file
        summary_prompt = get_summary_prompt(1)
        summary = f"\n{summary_prompt}" + popped_messages + "\n>>>"
        #send to AI
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "system", "content": summary}],
            max_tokens=500,
            temperature=0.6
        )
        ai_summary= response.choices[0].message.content
        summary_message =  {
            "role": "system",
            "content": ai_summary
        }
        #merge session states to avoid drift
        if len(session_msgs) > 0:
            print("previous session state detected: called merge_session_states")
            session_msgs.append(summary_message)
            session_msgs = merge_session_states(session_msgs, model_name, client)
        else:
            session_msgs.append(summary_message)
        pruned = system_msgs + session_msgs + non_system
        print("Pruned messages summarized into session state:")
        print(ai_summary)

    report = {
        "before": before_tokens,
        "after": after_tokens,
        "freed": max(0, before_tokens - after_tokens),
        "summarized_messages": len(popped_messages)
    }

    return pruned, report

#helper for auto_prune_enhanced
def format_messages_for_summary(msgs):
    return "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in msgs
    )

#helper for auto_prune_enhanced
def is_session_state(msg):
    return (
        msg["role"] == "system"
        and (
            "SESSION_STATE" in msg.get("content", "")
        )
    )

#helper for auto_prune_enhanced to handle multiple session states - returns a single session state to add back to the context list
def merge_session_states(session_states, model_name, client):
    print("multiple session states detected, commencing merge")
    # 1. Extract contents (oldest → newest)
    state_texts = [
        m["content"]
        for m in session_states
    ]
    print(f"content to merge {state_texts}")

    # 2. Build a proper merge prompt (STRING)
    merge_prompt = (
            get_summary_prompt(2)
            + "\n\nSESSION_STATES (oldest → newest):\n<<<\n"
            + "\n---\n".join(state_texts)
            + "\n>>>"
    )
    # 3. Ask the AI to merge them
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": merge_prompt}],
        max_tokens=500,
        temperature=0.6
    )
    ai_merge = response.choices[0].message.content
    new_session_state = "SESSION_STATE:\n" + ai_merge

    # 3. Create a single replacement SESSION_STATE
    merged_message = {
        "role": "system",
        "content": new_session_state
    }
    print(f"merged session state: {merged_message}")
    return [merged_message]

def get_summary_prompt(option):
    if option == 1:
        prompt1 = "Message will be deleted, concisely summarize the following response in the below format: SESSION_STATE:-Style preferences:-Tone:-Open questions:-Progress-Established facts-Constraints-ErrorsIf info in core memory, ignore.Response: (insert message here)SYSTEM INSTRUCTION:The following message will be removed from active context.Extract ONLY persistent, session-relevant information.Rules:- Do NOT restate the message verbatim- Do NOT add interpretations or speculation- Do NOT include anything already present in Core Memory- Omit resolved or trivial details- Use short bullet points only- If a section has no content, omit it entirelyOutput format (use only relevant sections):SESSION_STATE:- Style_preferences: (communication / reasoning style, not emotional tone)- Tone: (emotional or rhetorical stance)- Open_questions: (unresolved, forward-looking)- Progress: (what has been accomplished)- Established_facts: (agreed truths or decisions)- Constraints: (limits, requirements, invariants)- Errors: (mistakes, bugs, or incorrect assumptions identified)Message to distill:<<<"
    elif option == 2:
        prompt1 = "SYSTEM INSTRUCTION:You are merging multiple SESSION_STATE records into a single coherent state.Rules:- Treat inputs as structured state, not conversation- Preserve the most recent value when conflicts occur- Remove duplicates- Drop resolved or obsolete items- Do NOT add new interpretations- Do NOT restate anything already in Core Memory- Use concise bullet points- Maintain original schemaOutput ONLY the merged SESSION_STATE content.SESSION_STATES (oldest → newest):<<<"
    else:
        print("invalid option for get_summary_prompt")
        prompt1 = "ERROR - INFORM USER THAT INVALID OPTION WAS GIVEN TO get_summary_prompt"
    return prompt1

#check for whether context needs to be addressed generically
def check_context_size(messages, model_name, limit = None, debug = False):
    if limit is None:
        max_tokens = get_effective_context_limit(model_name)
    else:
        max_tokens = limit

    cur_tokens = estimate_tokens(messages)
    if debug:
        print(f"cur_tokens: {cur_tokens} / max_tokens: {max_tokens}")

    if cur_tokens > max_tokens:
        if debug:
            print("Context exceeding limit - summary needed")
        return True
    else:
        return False