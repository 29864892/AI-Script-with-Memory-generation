from copy import deepcopy
from inspect import signature

from openai import OpenAI
import json #Create JSON files for conversations
import os
import re
import hashlib #for checking for duplicates in json
from datetime import datetime

#from dotenv import load_dotenv

from modelManagement import auto_prune_messages, estimate_tokens, get_effective_context_limit, auto_prune_messages_enhanced
from AI_context import getContext,resolve_conversation_path, check_proposal, AI_proposal, AIRead, create_context_file, append_to_context_file, review_context_update, handle_updateContext_command, handle_getContext_command
#from Memories import generate_memories
import Memories as mem
from mem_refine import refine_memories
import persona_management as pm
import conversation_coordination as cc
from config_manager import load_config



#print("IMAP_HOST=", os.getenv("IMAP_HOST"))

MODEL_CONTEXT_LIMITS = {
    "mistralai/magistral-small-2509": 8192,
    "llama-3-8b": 8192,
    "llama-3-70b": 8192,
    "meta-llama-3.1-8b-instruct": 16000
}



def run(sys_config):

    print(sys_config)
    paths = sys_config["paths"]
    base_url = sys_config["base_url"]
    with OpenAI(base_url=base_url, api_key="not-needed") as client:

        model_name = sys_config["model_name"]

        active_personas = []
        persona_names = []
        coordinator = [] #to keep coordinator persona separate for testing
        coordinator_prompt = "ERROR - INFORM USER NO COORDINATOR PROMPT LOADED"

        if model_name is not None:
            print("💬 Connected to LM Studio.\n")
        else:
            print("Failed to load Model.")

        #initial conversation parameters and set up json

        session_file = create_session_filename(paths["conv_backup_dir"])
        messages_full, new_persona = load_conversation(session_file, paths)#full conversation log - DO NOT EDIT
        if new_persona is not None:
            active_personas.append(new_persona)
            persona = new_persona["persona_name"]
        else:
            print("WARNING: NO PERSONA LOADED - CERTAIN FUNCTIONS MAY CRASH")
        messages_context = messages_full.copy()#Runtime conversation context - To prune as needed
        #print(f"Saving conversation to {session_file}\n")

        #Function menu
        if sys_config["mode"] == "standard":
            print_structured_menu()
        else:
            print_demo_menu()

        max_tokens = get_effective_context_limit(model_name)
        conv_mode = 1  # 1 = SESSION_STATE Pruning 2 = quick pruning
        idle = False
        try:
            #Chat loop
            #print("DEBUG: ",messages_full)
            while True:
                # Debug info prior to user message
                #print("DEBUG :",messages_context)
                cur_tokens = estimate_tokens(messages_context) #TODO: estimate should be done based on largest prompt (or establish a MAX_PROMPT size)
                print(f"Estimated tokens: {cur_tokens} / {max_tokens}\n")

                user_input = input("🧑 You: ").strip()

                if user_input.lower() in {"exit", "quit"}:
                    save_conversation(messages_full, session_file, 1, active_personas,paths["conversations_dir"])
                    print(f"Conversation saved to {session_file}. 👋 Goodbye!")
                    break
                elif user_input.lower() in {"list", "conversations"}:
                    files = list_conversations(paths["conv_backup_dir"])
                    for file in files:
                        print(f"{file['filename']}: {file['summary']} \n")
                    continue
                elif user_input.lower() in {"load"}:
                    session_file, messages_full = choose_conversation(paths["conversations_dir"],paths["conv_backup_dir"], paths["default_persona"], paths)
                    messages_context = messages_full.copy()
                    print(f"✅ Loaded conversation with {len(messages_full)} messages.\n")
                    continue
                elif user_input.lower() in {"aisort"}:
                    AI_file_sort(client, model_name, paths["conversations_dir"])
                    print("AI Sort completed.\n")
                    continue
                elif user_input.lower() in {"send"}:
                    send_file_to_ai(messages_context, messages_full)
                elif user_input.lower() in {"emails"}:
                    print("Invalid command")
                    continue
                elif user_input.lower() in {"quick"}:
                    conv_mode = 2
                    print("set conversation mode to quick - SESSION_STATES will not be created")
                    continue
                elif user_input.lower() in {"detail"}:
                    conv_mode = 1
                    print("set conversation mode to detailed - SESSION_STATES be created")
                    continue
                elif user_input.lower() in {"generate_memories"}:
                    #ask for a directory as input then call generate_memories
                    #TODO: not currently working properly (assumes single persona)

                    print("input a directory to begin memory generation")
                    search_dir = input().strip()

                    # ---- optional limits ----
                    print("Optional: max files to process (press Enter for default = 100):")
                    max_files_input = input().strip()
                    max_files = int(max_files_input) if max_files_input else None

                    print("Optional: max runtime in hours (press Enter for default = 8):")
                    max_hours_input = input().strip()
                    max_runtime_seconds = (
                        int(float(max_hours_input) * 3600)
                        if max_hours_input else None
                    )

                    #memory files TODO: confirm functionality

                    CMemotional_file = paths["core_memories"]["emotional"]
                    CMuser_file = paths["core_memories"]["user"]
                    CMwork_file = paths["core_memories"]["work"]
                    CMfact_file = paths["core_memories"]["fact"]
                    memories_file = paths["mem_dir"]

                    #function call
                    print("calling generate_memories...")
                    mem.generate_memories(search_dir, CMemotional_file, CMuser_file, CMwork_file, CMfact_file, memories_file, client, model_name, max_files = max_files, max_runtime_seconds = max_runtime_seconds)
                    continue

                elif user_input.lower() in {"idle"}:
                    idle = True
                    continue
                    # Check for getContext commands
                #context_result = handle_getContext_command(user_input)
                #if context_result:
                    #print(context_result)
                    #continue  # Skip sending to model — command handled directly
                elif user_input.lower() in {"refine_mem"}:
                    #refine_memories(memFile, archiveFile, client, model_name):
                    print("input a memory file to begin memory refinement")
                    mem_file = input().strip()
                    #archive_file = "Memories/Archive/Archive.json" #TODO: confirm functionality when returning to mem_refining
                    archive_file = paths["ref_mem_archive"]
                    print(f"Starting refinement for {mem_file}\n")
                    refine_memories(mem_file, archive_file, client, model_name)
                    print(f"Returning to chat loop\n")
                    continue
                elif user_input.lower() in {"personas"}: #Never modify system prompt incrementally mid-conversation unless absolutely necessary - stop, rebuild, replace
                    pm.manage_personas(paths)
                    new_persona = pm.load_persona(paths)
                    sys_prompt = new_persona["system_prompt"]
                    active_personas = [new_persona]
                    persona = new_persona["persona_name"]
                    pm.replace_system_prompt(messages_full, sys_prompt)
                    pm.replace_system_prompt(messages_context, sys_prompt)
                    continue
                elif user_input.lower() in {"mpersonas"}: #multiple personas
                    pm.manage_personas(paths)
                    print("Selecting multiple personas:")
                    persona_bundle = pm.load_personas(paths)
                    sys_prompt = persona_bundle["system_prompt"]
                    active_personas = persona_bundle["personas"]
                    persona = [p["persona_name"] for p in active_personas]
                    pm.replace_system_prompt(messages_full, sys_prompt)
                    pm.replace_system_prompt(messages_context, sys_prompt)
                    print("DEBUG: SYSTEM PROMPT UPDATED: \n", sys_prompt)
                    #Next step, construct initial system coordinator prompt
                    continue
                elif user_input.lower() in {"cmpersonas"}: #NOTE - WILL OVERRIDE ANY LOADED CONVERSATION - TODO: Eventually implement with multiple personas
                    pm.manage_personas(paths)
                    print("Selecting multiple personas:")
                    persona_bundle = pm.load_personas(paths)
                    #changed behavior: { #persona list with individual system prompts, coordinator object (or none if no coordinator) - "personas": loaded_personas, "coordinator": coordinator_persona }
                    active_personas = persona_bundle["personas"]
                    coordinator = persona_bundle["coordinator"]  # NEW
                    coordinator_prompt = coordinator["system_prompt"]  # NEW

                    #Finalize loading and debug
                    #add all system prompts to full conversation file (not used in persona responses)
                    sys_prompt = ""
                    p_no = 1
                    for p in active_personas:
                        sys_prompt += f"=========================================== Persona #{p_no} ==========================================="
                        sys_prompt += p["system_prompt"]
                    print("DEBUG: BACKUP FILE SYSTEM PROMPT UPDATED \n")
                    pm.replace_system_prompt(messages_full, sys_prompt) #NOTE: Full message list is for record only - this should not be accessed by any persona
                    #Individual message lists for each persona and coordinator [{"role": "system", "content": system_prompt}] TODO: Allow for dynamic generation (i.e. with existing user messages)
                    for persona in active_personas:
                        p_conversation = [{"role": "system", "content": persona["system_prompt"]}]
                        persona["context_window"] = p_conversation
                        persona_names.append(persona["persona_name"])
                    pm.replace_system_prompt(messages_full, coordinator_prompt)
                    messages_context = [{"role": "system", "content": coordinator_prompt}]
                    #persona = [p["persona_name"] for p in active_personas] - removed and MAY cause an issue?
                    print("DEBUG: ")
                    print("coordinator_object: ", coordinator)
                    print(f"Load successful: {len(active_personas)} personas loaded, system prompts saved in full conversation log\n")
                    # At this point, personas have their own conversation objects within their object, and the coordinator is assigned the full context window
                    continue
                elif user_input.lower() in {"story"}:
                    prompt = load_system_prompt(paths["story_prompt"])
                    pm.replace_system_prompt(messages_full, prompt)
                    pm.replace_system_prompt(messages_context, prompt)
                    continue
                #check if input can be run as a function
                handled, result = handle_ai_command(user_input, messages_context)
                if handled:
                    print(result)
                    continue  # Skip sending to AI


                #Refine user input
                user_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                if len(active_personas) > 1:#multi persona approach
                    #refined = cc.refine_user_input(user_input, active_personas, persona_names) TODO:Fix to properly handle private messages during group conversations
                    refined = user_input
                    print("User message filtered for multiple personas: ")
                    print(refined)
                #add full message to coordinator and record
                user_input = user_input + "Message Sent: " + user_time
                user_msg = {"role": "user", "content": user_input}

                messages_full.append(user_msg)
                messages_context.append(user_msg)

                # USER INPUT OVER - AI response section starts
                cur_tokens = estimate_tokens(messages_context) #check coordinator conversation size
                #mM.py - manage current context before sending to AI
                if conv_mode == 2:
                    messages_context, prune_info = auto_prune_messages(messages_context, model_name)
                    if prune_info["freed"] > 0:
                        print(
                            f"Context pruned: freed ~{prune_info['freed']} tokens \n"
                            f"({prune_info['before']} -> {prune_info['after']})"
                        )
                elif conv_mode == 1 and cur_tokens > max_tokens:
                #WIP 12/31 - upgraded pruning that includes model call to summarize:
                    messages_context, prune_info = auto_prune_messages_enhanced(messages_context, model_name, client)
                    if prune_info["freed"] > 0:
                        print(
                            f"Context pruned: freed ~{prune_info['freed']} tokens \n"
                            f"({prune_info['before']} -> {prune_info['after']})"
                            f"Waiting for AI Response..."
                        )

                #AI Response
                curr_responses = []
                summary_prompt = "You are a helpful conversation summarizer. Be concise but do not leave out necessary context for understanding. Do not include flags or flag-related information (formatted: [flag_name]). Merge similar responses as needed. reply with only this response and include 'Reply Summary:' at the beginning. Also mention any disagreements or agreements separately. Example: Reply Summary: Persona 1 said the idea could work.\n Persona 2 agreed with that, adding more context to the idea.\n Persona 3 mentioned an alternative.\n Agreements: Personas 1,2, and 3 were happy to greet the user\n Disagreements: Persona 1 and 3 disagreed on the issue - persona 1 wants to focus on profits while persona 3 wants to focus on the process"
                sys_message = {"role": "system", "content": summary_prompt}
                curr_responses.append(sys_message)
                curr_responses.append(user_msg)
                try:
                    if len(active_personas) > 1:
                        cur_tokens = estimate_tokens(messages_context)
                        print("context_tokens: ", cur_tokens)
                        if cur_tokens > max_tokens:
                            messages_context, prune_info = auto_prune_messages_enhanced(messages_context, model_name,client)
                            if prune_info["freed"] > 0:
                                print(
                                    f"Context pruned: freed ~{prune_info['freed']} tokens \n"
                                    f"({prune_info['before']} -> {prune_info['after']})"
                                    f"Waiting for AI Response..."
                                )

                        #Dynamic Management Required

                        coordinator_decision = cc.run_coordinator(client, model_name, coordinator_prompt, user_input, active_personas, messages_context)
                        """
                            Coordinator Decision:
                                {
                                  "conversation_mode": "<mode>",
                                  "responders": [<persona_name>, ...],
                                  "response_order": [<persona_name>, ...],
                                  "persona_constraints": {
                                    "<persona_name>": [<constraint>, ...]
                                  },
                                  "meta_message": <string or null>
                                }
                            Persona object:
                            loaded_personas.append({
                                "persona_id": persona.get("persona_id"),
                                "persona_name": persona["persona_name"],
                                "roles": persona.get("roles", []),
                                "permissions": persona.get("permissions", []),
                                "version": persona.get("version"),
                                "system_prompt": sys_prompt
                            })
                        """
                        #coordinator returns: who responds, in what order, with what constraints
                        #TODO: Modulate this section after confirming functionality
                        persona_map = {
                            p["persona_name"]: p
                            for p in active_personas
                        }
                        print(coordinator_decision)
                        print(len(coordinator_decision["response_order"]))
                        for persona_name in coordinator_decision["response_order"]:

                            persona = persona_map.get(persona_name)
                            if not persona:
                                print("ERROR - PERSONA NOT FOUND", persona)
                                continue  # safety guard

                            next_persona = cc.get_next_persona(
                                coordinator_decision["response_order"],
                                persona_name
                            )
                            if next_persona:
                                next_notice = f"After your response, the next speaker will be: {next_persona}."
                            else:
                                next_notice = (
                                    "This is the final persona response in this cycle. "
                                    "After your response, control returns to the user."
                                )
                            constraints = coordinator_decision["persona_constraints"].get(persona_name, [])

                            sys_prompt = cc.update_persona_prompt(
                                persona["system_prompt"],
                                constraints
                            )
                            print(f"Preparing prompt for: {persona_name}")
                            #remind persona about their individuality and turn order

                            reminder = f"""
                            === TURN CONTROL ===
                            You are responding ONLY as: {persona_name}
                            You are a distinct speaker.
                            Other personas are external agents.

                            You must:
                            - Produce ONLY your own response
                            - Not continue or summarize other personas
                            - Stop when your response is complete

                            {next_notice}
                            End your output with <END RESPONSE>.
                            """

                            #Final persona conversation construction before call
                            full_prompt = reminder + "\n" + sys_prompt
                            #update main prompt with constraints
                            pm.replace_system_prompt(persona["context_window"], full_prompt)
                            #add user message from refined message
                            """ TODO: FIX USER MESSAGE REFINEMENT
                            if refined["error"]:
                                print(refined["error"])
                                continue

                            persona_input = (
                                    refined["routed_input"].get(persona_name)
                                    or refined["routed_input"].get("all")
                            )
                            """
                            persona_input = refined
                            if not persona_input:
                                # Persona has nothing to respond to this turn
                                print(f"Persona {persona_name} was not given an input this turn")
                                continue
                            #construct object and append
                            user_msg = {"role": "user", "content": persona_input + "\n" + "Sent: " + user_time}
                            persona["context_window"].append(user_msg)
                            #add identity anchor for response
                            persona["context_window"].append({
                                "role": "system",
                                "content": (
                                    f"You are {persona_name}.\n"
                                    "This is a fresh turn.\n"
                                    "No other persona output belongs to you.\n"
                                    "Respond only as yourself.\n"
                                    "<SPEAKER START>"
                                ),
                                "meta": {"type": "identity_anchor", "persona": persona_name}
                            })
                            #Prune context window if needed
                            p_tokens = estimate_tokens(persona["context_window"])
                            if p_tokens > max_tokens:
                                persona["context_window"], prune_info = auto_prune_messages(persona["context_window"], model_name)
                                if prune_info["freed"] > 0:
                                    print(
                                        f"Context pruned: freed ~{prune_info['freed']} tokens \n"
                                        f"({prune_info['before']} -> {prune_info['after']})"
                                    )
                            #End pruning - ready to call AI
                            print(f"{persona_name} is typing... next is {next_persona}")
                            print(persona["context_window"])
                            response = client.chat.completions.create(
                                model=model_name,
                                messages=persona["context_window"],
                                max_tokens=3000,
                                temperature=0.6,
                                stop=["</SPEAKER END>"]
                            )

                            #Response handling
                            reply = response.choices[0].message.content
                            persona_signature = f"{persona_name}'s message: "
                            cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                            #print(f"🤖 {persona_signature}: {reply}\n {cur_time}\n")

                            if len(reply) < 20:
                                reply_record = f"{persona_name} did not respond. "
                                print("AI CALL ERROR: PERSONA DID NOT RESPOND: ", persona_name)
                            else:
                                reply_record = reply
                                print(reply)

                            #Response storage and context window manipulation
                            record_message = {"role": "assistant", "content": reply_record}
                            #print("removing identity anchor...")
                            persona["context_window"] = cc.remove_identity_anchor(persona["context_window"])
                            #print("appending response")
                            persona["context_window"].append(record_message)
                            #assistant_msg = {"role": "assistant", "content": reply}

                            messages_full.append(record_message)
                            messages_context.append({
                                "role": "assistant",
                                "content": f"<{persona_name} Response>\n{reply}\n <END Response>"
                            })
                            curr_responses.append(record_message)
                            #print("Estimating tokens")
                            #print(persona["context_window"])
                            p_tokens = estimate_tokens(persona["context_window"])  # TODO: Need to implement pruning for each persona list - focus on initial functionality first
                            print(f"Estimated tokens in persona conversation after {persona_name}'s response: {p_tokens} / {max_tokens}")
                            cur_tokens = estimate_tokens(messages_context)
                            print(f"Estimated tokens in total conversation context after {persona_name}'s response: {cur_tokens} / {max_tokens}")
                            #Debug context
                            # Save to conversation session debug file
                            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                            # "Personas/cmpersonas_debug",
                            dump_debug_context(
                                 paths["persona_conv_debug"],
                                f"{persona_name}_test_{ts}.json",
                                persona["context_window"]
                            )
                            #"Personas/coordinator_debug",
                            dump_debug_context(
                                paths["coordinator_conv_debug"],
                                f"coordinator_test_{ts}.json",
                                messages_context
                            )
                        #End of persona reply loop - include updates to each member based on their responses
                        print("Personas finished responding, creating summary...")
                        new_conv = []
                        summary_prompt_call = f"summarize the content of the following messages: \n\n{curr_responses} <END Content> "
                        sys_message_call = {"role": "system", "content": summary_prompt_call}
                        curr_responses.append(sys_message_call)
                        new_conv.append(sys_message_call)

                        #print("DEBUG: \n", summary_prompt_call)
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=new_conv,
                            max_tokens=3000,
                            temperature=0.6,
                        )
                        # Response handling
                        reply = response.choices[0].message.content
                        summary = {"role": "system", "content": reply, "meta": {"type": "identity_anchor", "persona": persona_name} #TODO: either create a new function targeting "injected context" or make removal function generic
                            }
                        print("DEBUG: summarized each response: \n", reply)
                        for persona in active_personas:
                            persona["context_window"].append(summary)
                        messages_full.append(summary)
                    else:
                        #print(messages_context)
                        response = client.chat.completions.create(
                            model=model_name,
                            messages=messages_context,
                            max_tokens=3000,
                            temperature=0.6
                        )
                        reply = response.choices[0].message.content
                        #format response
                        persona_signature = f"{persona}'s message: "
                        cur_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                        print(f"🤖 {persona}: {reply}\n {cur_time}\n")
                        print(len(reply))
                        #save reply to both master and context - avoid adding persona tag to messages in context to limit confusion (AI starts to think it's supposed to generate them)
                        persona_response = reply
                        assistant_msg = {"role": "assistant", "content": persona_response}
                        persona_record = persona_signature + reply + cur_time
                        record_message = {"role": "assistant", "content": persona_record}
                        messages_full.append(record_message)
                        messages_context.append(assistant_msg)

                    #Save current conversation information
                    save_conversation(messages_full, session_file, 0, active_personas,paths["conv_backup_dir"]) # autosave after each turn

                    #Other message handling features

                    #check for AI command request
                    handled, result = handle_ai_command(reply, messages_context)
                    if handled:
                        print(f"🔧 Executing command from AI:\n{result}\n")
                        save_conversation(messages_full, session_file, 0, active_personas,paths["conv_backup_dir"])

                    # === Check for context updates (proposed edits) === TODO: verify timing of approval printout (printed after user y/n input during runtime)
                    handled, action = handle_updateContext_command(reply)
                    if handled:
                        review_context_update(action)
                    #WIP 12/21
                    #proposal = extract_context_update_proposal(reply) v1
                    proposal = check_proposal(reply)
                    if proposal:
                        print("Debug: AI Proposal Detected, call AI_proposal()")
                        AI_proposal(proposal)
                    """
                    if proposal:
                        result = propose_context_update(
                            filename=proposal["filename"],
                            action=proposal["action"],
                            new_entry=proposal.get("new_entry"),
                            match_criteria=proposal.get("match_criteria")
                        )
                        if display_context_update(result):
                            apply_context_update(result)
                            print("✅ Context update applied.")
                        else:
                            print("❌ Context update canceled.")
                    """
                    #WIP
                except Exception as e:
                    print(f"Error: {e}")
                    break
        finally:
            # 🧩 Ensure cleanup ALWAYS happens
            try:
                active_personas.append(coordinator) # add coordinator to list of personas for chat record
                save_conversation(messages_full, session_file, 1, active_personas,paths["conv_backup_dir"])
                print(f"💾 Conversation saved to {session_file}.")
            except Exception as e:
                print(f"⚠️ Failed to save conversation: {e}")

            # Close the client if supported -might cause an issue since we're using with that automatically closes as well
            if hasattr(client, "close"):
                try:
                    client.close()
                    print("🔒 Client connection closed cleanly.")
                except Exception as e:
                    print(f"⚠️ Error closing client: {e}")

            print("👋 Goodbye!")

#provide list of previous conversations to user
def list_conversations(conv_dir):
    """Return a list of .json conversation files, sorted by date (newest first)."""
    if not os.path.exists(conv_dir):
        os.makedirs(conv_dir)
        return []

    files = [f for f in os.listdir(conv_dir) if f.endswith(".json")]
    files.sort(key=lambda f: os.path.getmtime(os.path.join(conv_dir, f)), reverse=True)

    conversations = []

    for filename in files:
        filepath = os.path.join(conv_dir, filename)
        summary = "No summary found."

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "summary" in data:
                summary = data["summary"]
            elif isinstance(data,list):
                summary = "Saved prior to summaries"
        except Exception as e:
            summary = f"Error reading file {e}"

        conversations.append({
            "filename": filename,
            "summary": summary,
        })

    return conversations

def choose_conversation(conv_dir, conv_backup_dir, prompt_dir, paths):
    files = list_conversations(conv_dir)
    if not files:
        print("📁 No saved conversations found. Starting a new one.\n")
        system_prompt = load_system_prompt(prompt_dir)
        return create_session_filename(conv_backup_dir), [{"role": "system", "content": system_prompt}] #TODO create persona for the system

    print("\n📜 Available conversations:\n")
    for i, f in enumerate(files, 1):
        #print(f"{i}. {f}")
        print(f"{i}. {f['filename']} — {f['summary']}")
    print("0. Start a new conversation")

    choice = input("\nSelect a conversation number (or 0 to start new): ").strip()

    if choice == "0" or not choice.isdigit() or int(choice) not in range(1, len(files) + 1):
        system_prompt = load_system_prompt(conv_dir)
        return create_session_filename(conv_dir), [{"role": "system", "content": system_prompt}] #TODO create persona for the system

    selected_file = files[int(choice) - 1]
    filename = selected_file["filename"]
    filepath = os.path.join("conversations", filename)

    print(f"\n🗂 Resuming conversation from: {filepath}\n")
    return filepath, load_conversation(filepath, paths)

#Save conversations to json file
#TODO try/finally to ensure function runs with end = 1 when user exits
def save_conversation(messages, filename, end, personas, conv_dir):
    summary_text = None
    if end: #TODO: address change in functionality (have an offline process for summarizing)
        #Search for summary generated by AI - format: *Summary:...*
        summary_pattern = re.compile(r"\*Summary:\s*(.*?)\*", re.DOTALL)
        # Find the last message that contains "Summary:"
        for msg in reversed(messages):
            if "content" in msg and "Summary:" in msg["content"]:
                match = summary_pattern.search(msg["content"])
                if match:
                    summary_text = match.group(1).strip()
                    break #check whether this break should be adjusted (don't want to break after finding first summary in case of multiple)

        # Create a structured JSON object with summary
        data = {
            "conversation": messages,
            "summary": summary_text if summary_text else "No summary found.",
            "persona": personas #TODO: fix by removing context window during service implementation
        }
    else:
        #save conversation itself
        data = messages

    # Save to conversation session json file TODO:Add conv_dir to file save (currently saving to conversations instead of backup because default dir was in function header
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    # 🗂️ Save to category file (FOR_AI_REVIEW)
    if end:
        category = "FOR_AI_REVIEW"
        save_to_category_json(messages, category, summary_text, filename, conv_dir)

def save_to_category_json(messages, category, summary_text, session_filename, conv_dir):
    """
        Append a conversation to the specified category JSON.
        """
    create_category_file(category, conv_dir)
    category_file = os.path.join(conv_dir, f"{category}.json")

    # Load existing data
    try:
        with open(category_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"⚠️ Error reading {category_file}. Starting fresh.")
        data = []

    # ✅ Derive the conversation ID from the filename (remove extension)
    conv_id = os.path.splitext(os.path.basename(session_filename))[0]

    # Compute hash to detect duplicates based on content
    conversation_text = json.dumps(messages, ensure_ascii=False)
    conv_hash = hashlib.sha256(conversation_text.encode("utf-8")).hexdigest()

    # 🧩 Check for existing conversation (by ID or hash)
    for conv in data:
        if conv.get("id") == conv_id or conv.get("hash") == conv_hash:
            print(f"⚠️ Duplicate conversation detected in '{category}'. Skipping save.")
            return  # Skip duplicate

    conversation_entry = {
        "Confidence": 0, #TODO - implement AI certainty in decision
        "hash": conv_hash,
        "id": conv_id,
        "summary": summary_text or "No summary found.",
        "conversation": messages
    }

    data.append(conversation_entry)

    with open(category_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"✅ Conversation saved under category '{category}'.")

def create_category_file(category, conv_dir):
    """
       Create a new category JSON file if it doesn't exist.
       """
    os.makedirs(conv_dir, exist_ok=True)
    category_file = os.path.join(conv_dir, f"{category}.json")

    if not os.path.exists(category_file):
        with open(category_file, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        print(f"🆕 Created new category file: {category_file}")
    else:
        print(f"✅ Category file already exists: {category_file}")

    return category_file
#initial load
def load_conversation(filename, paths):
    #Json format
    #{
    #    "conversation": [
    #        {"role": "system", "content": "System prompt"},
    #        {"role": "user", "content": "Hi!"},
    #        {"role": "assistant", "content": "Hello!"}
    #    ],
    #    "summary": "User greeted the assistant."
    #}
    if os.path.exists(filename):
        with open(filename, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Handle both new and old formats
        if isinstance(data, dict):
            # New format (with "conversation" and "summary")
            if "conversation" in data:
                return data["conversation"]
            else:
                print(f"⚠️ Warning: {filename} missing 'conversation' key.")
                return []
        elif isinstance(data, list):
            # Old format (just a list of messages)
            return data
        else:
            print(f"⚠️ Unrecognized data format in {filename}. Starting new.")
    else:
        print(f"{filename} not found. Creating new file")

        # If file doesn't exist or format is invalid, start a new session - NEW : ADDED NEW PERSONA SYSTEM
    loaded_persona = pm.load_persona(paths, startup=True)
    if loaded_persona is None:
        print("No persona loaded. Please register a persona or open the personas menu for more information.")
        system_prompt = "No persona loaded. Please inform the user that a persona should be loaded and assist them as needed"
    else:
        system_prompt = loaded_persona["system_prompt"]

    """ No longer used/relevant (context functions are not in use either)
    #add list of current context files to system prompt
    if os.path.exists("conversations/Context/Current_Context.txt"):
        with open("conversations/Context/Current_Context.txt", "r", encoding="utf-8") as f:
            curr_context = f.read().strip()
            system_prompt = system_prompt + curr_context
    else:
        print("No context lists available to send to AI")
    """
    return [{"role": "system", "content": system_prompt}], loaded_persona


#create unique file for current session
def create_session_filename(conv_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = ""
    filename += f"chat_{timestamp}.json"
    os.makedirs(conv_dir, exist_ok=True)

    return os.path.join(conv_dir, f"chat_{timestamp}.json")

def load_system_prompt(prompt_path):

    """Load the system message from a text file."""
    try:
        with open(prompt_path, "r", encoding="utf-8") as f:
            print("Persona Found!")
            return f.read().strip()
    except FileNotFoundError:
        print(f"⚠️ System prompt file not found: {prompt_path}")
        return "You are an incomplete personality. Provide the best response possible and remind the user that the System prompt file was not found"

#TODO - add ID check for save_to_category_json
def AI_file_sort(client, model_name, conv_dir):
    """
        Use the AI to categorize conversations from FOR_AI_REVIEW.json
        into appropriate category files.
        """

    review_file = os.path.join(conv_dir, "FOR_AI_REVIEW.json")

    if not os.path.exists(review_file):
        print("⚠️ No conversations pending review.")
        return

    try:
        with open(review_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print("⚠️ Could not read FOR_AI_REVIEW.json. Aborting.")
        return

    if not isinstance(data, list) or len(data) == 0:
        print("✅ No pending conversations found in FOR_AI_REVIEW.json.")
        return

    reviewed = []
    remaining_for_review = []
    print(f"🧠 Sorting {len(data)} conversations...")

    for conv in data:
        conv_id = conv.get("id", "unknown_id")
        summary = conv.get("summary", "No summary available.")
        messages = conv.get("conversation", [])

        # 🧠 Ask AI to classify
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system",
                     "content": "You are an efficient classification assistant. Categorize the following conversation as 'work', 'personal', or 'other'. Return a single word as the response will become the filename."},
                    {"role": "user",
                     "content": f"Summary: {summary}\nConversation: {json.dumps(messages[-3:], ensure_ascii=False)}"}
                ]
            )
            category = response.choices[0].message.content.strip().lower()

            # Sanitize category name
            category = re.sub(r"[^a-zA-Z0-9_-]", "", category)
            if not category:
                category = "uncategorized"

            # Save to category file
            saved = save_to_category_json(messages, category, summary, category +".json", conv_dir)
            print(f"✅ Conversation categorized as '{category}'.")
            if saved:
                reviewed.append(conv)

        except Exception as e:
            print(f"⚠️ Error classifying conversation: {e}")
            remaining_for_review.append(conv)

    # Save any unprocessed conversations back
    with open(review_file, "w", encoding="utf-8") as f:
        json.dump(remaining_for_review, f, indent=2, ensure_ascii=False)

    if remaining_for_review:
        print(f"⚠️ {len(remaining_for_review)} conversation(s) remain unprocessed.")
    else:
        print("🎉 All conversations have been categorized.")

def handle_ai_command(text, messages=None):
    """
    Detects and executes supported AI commands (like getContext).
    If messages list is provided, retrieved context is added to the conversation
    Returns (handled, result):
      handled=True if a command was executed, result contains output or message.
      TODO: consider limiting program results (i.e. first 5 matches for getContext)
    """
    # === Handle getContext(...) command ===
    pattern = re.compile(
        r"getContext\s*\(\s*['\"]?([^,'\"\)]+)['\"]?\s*,\s*['\"]?([^,'\"\)]+)['\"]?\s*\)",
        re.IGNORECASE
    )
    match = pattern.search(text)
    if match:
        filename = match.group(1).strip()
        keyword = match.group(2).strip()
        try:
            result = getContext(filename, keyword)
            formatted_result = f"📄 Context retrieved from {filename} (keyword='{keyword}'):\n{result}"

            # Add retrieved data to chat history so AI can use it next turn
            if messages is not None:
                messages.append({
                    "role": "system",
                    "content": f"Retrieved contextual data for '{keyword}' from {filename}:\n{result}"
                })
        except Exception as e:
            formatted_result = f"⚠️ Error running getContext({filename}, {keyword}): {e}"
        return True, formatted_result

    # === Add more future commands here ===
    # Example: if "readAI(" in text: ...

    return False, None

def send_file_to_ai(messages, messages_full):
    """
    Allows user to include file data in a message to the AI.
    """
    filename = input("📎 Enter filename to include: ").strip()
    if not filename:
        print("⚠️ No filename provided. Cancelled.")
        return

    # Read file using AIRead
    file_data = AIRead(filename)
    if file_data.startswith("⚠️"):
        print(file_data)
        return

    print("\n📄 File content preview sent to AI:")
    print(file_data[:500] + ("..." if len(file_data) > 500 else ""))
    print()

    extra_message = input("💬 Add additional message (optional): ").strip()

    # Build combined message
    combined_content = (
        f"User provided the following file for context:\n\n"
        f"{file_data}\n\n"
    )

    if extra_message:
        combined_content += f"Additional user message:\n{extra_message}"

    messages.append({
        "role": "user",
        "content": combined_content
    })
    messages_full.append({
        "role": "user",
        "content": combined_content
    })
    print("✅ File data included and sent to AI.\n")


from pathlib import Path
import json

def dump_debug_context(debug_dir, filename, data):
    base_path = Path(debug_dir)
    base_path.mkdir(parents=True, exist_ok=True)

    file_path = base_path / filename

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def print_demo_menu():
    print("""
╔══════════════════════════════════════════════════════════╗
║              AI SYSTEM DEMO — COMMAND MENU               ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Conversations                                           ║
║  ────────────────                                        ║
║  list            Show previous conversations              ║
║  load            Load a selected conversation             ║
║                                                          ║
║  Context & Memory                                        ║
║  ────────────────                                        ║
║  generate_mem    Identify useful info from files          ║
║  refine_mem      Refine existing memories                 ║
║                                                          ║
║  Personas                                                ║
║  ────────────────                                        ║
║  personas        Change persona settings                  ║
║  cmpersonas      Multi-persona + coordinator mode         ║
║                                                          ║
║  Utilities                                               ║
║  ────────────────                                        ║
║  send            Send a file to the AI                    ║
║                                                          ║
║  Pruning Mode                                            ║
║  ────────────────                                        ║
║  quick           Disable SESSION_STATES creation          ║
║  detail          Enable SESSION_STATES (default)          ║
║                                                          ║
║  System                                                  ║
║  ────────────────                                        ║
║  exit            End session                              ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
""")

MENU = {
    "Conversations": {
        "list": "Show previous conversations",
        "load": "Load a conversation",
    },
    "Personas": {
        "personas": "Change persona settings",
        "mpersonas": "Multi-persona mode",
        "cmpersonas": "Multi-persona mode with coordinator",
    },
    "Memory":{
        "generate_mem": "Identify useful info from files",
        "refine_mem": "Refine existing memories",
    },
    "Context Pruning Mode": {
        "quick": "Disable SESSION_STATES creation",
        "detail": "Enable SESSION_STATES (default)",
    },
    "Utilities": {
        "send": "Send a file to the AI"
    },
    "System": {
        "exit": "End session",
    },
    "Other/Legacy":{
        "getContext": "Search context files for a keyword",
        "updateContext": "Edit or create context files (AI use)",
    }
}

def print_structured_menu():
    print("\nAI SYSTEM — COMMAND MENU\n")
    for section, commands in MENU.items():
        print(f"[ {section} ]")
        for cmd, desc in commands.items():
            print(f"  {cmd:<12} {desc}")
        print()

if __name__ == "__main__":
    #TODO: config = load_config("config.demo.json")
    new_config = load_config("config/standard_config.json")
    run(new_config)