"""
    functions related to personas in AI system
    Goals:
    1.Allow for multiple personas
        1.Persona memory boundaries
            1.mixture of private and shared memories depending on context (most memories will be shared)
            2.will become applicable once memories are being actively injected into the conversation
        2.Persona scope - for temporary personas/one-off specialists/experimental agents not added to registry
            1.conversation
            2.session
            3.global
    2.At the start of a conversation, choose between (implement dynamic addition/removal of personas in the future)
        1.Single Persona Conversation
        2.Multiple Personas (combine persona prompts and include group instructions in system prompt)
        3.Every conversation should have exactly one "conversation controller" - can be treated as invisible, but should be architecturally present
    3.Register each persona and store in a list for quick reference
        {
          "persona_name": "Name",
          "persona_id": "ID",
          "version": "0.0",
          "date_registered": "mm/dd/yyyy",
          "system_prompt_path": "prompt_path",
          "default": "False",
          "rank": 0,
          "performance_points": 0,
          "roles": [],
          "permissions": [],
          "status": "active | experimental | disabled | archived",
          "visibility": "user | internal | system-only",
          "background_file_path": "background_file_path"
        }
        1.Version - system prompt version (keep old versions for records)
        2.Roles - what the persona is meant to do (e.g. conversational, secretary, researcher)
        3.Permissions - what the persona can access, and what the system will allow it to do (e.g. specific directories normally inaccessible to personas (if personas are allowed to access files, the default would be only conversations they participate in - expand as role requires, perhaps by having the persona submit an information request)
        4.Default - whether this is the default persona to load when no specific persona is requested
        5.Detailed background/dialogue data file (not based on conversations, but rather any relevant information related to persona formation) MUST be immutable during conversation, consider versioning when integrating feedback
        6.Rank - rank based on performance_points relative to other personas
        7.performance_points - to be calculated at the end of a conversation and only shared at the start
        8.persona_id - immutable to avoid keying by name (subject to change)
        9.status - whether in use or not to keep old personas in registry, prevent accidental loading, and test safely
        10.visibility - who the persona interacts with
    4.Load Personas
        1.Load system prompt(s)
        2.Include access to additional dialogue files
        3.Append to each persona's system prompt their feedback report from their last conversation, and their current score and rank
    5.During a conversation
        1.Allow each persona in the conversation a response in order of rank or prioritize a mentioned persona's response (i.e. @bot2 will have bot2 respond first instead of bot1)
        2.Turn order resolution
            1.Explicit mention order
            2.role priority
            3.rank
            4.fallback to controller (if not invisible)
        3.consider a "trainer" persona that can view all persona's feedback reports and help guide their responses during a conversation, also requesting input from the user as needed
            Trainer should never
                1.modify scores directly
                2.override user intent
                3.suppress another persona's output
    6.At the end of a conversation (NOT IMPLEMENTED)
        1.Run a neutral "grader" that will assign points based on performance and generate a feedback report - consider a "trainer" persona that can view all persona's feedback reports and help guide their responses during a conversation, also requesting input from the user as needed
            1.Ability to answer user questions
            2.Ability to create user engagement
            3.Ability to generate meaningful conversation - asking and answering complex questions
            4.Amount of overlap between a previous persona's response (i.e. bot2 should lose points if their response is too similar to bot1's - avoid simply restating information)
            5.Generate a summary of the persona's contribution to the conversation (unique points made)
            6.Generate a summary of ways that the persona can improve in the next conversation (avoid mentioning specific loss of points or sharing points here to limit stress)
            7.Ensure grading is partially hidden
            8.Occasional randomization to prevent overfitting
            Scoring system
                1.Each criterion gets
                    1.A randomized max between 15 - 35
                    2.An earned score between 0 and that max (~60- ~140, which normalizes to 100)
                1.Task Fulfillment & Accuracy - "Did the persona do what was asked?"
                2.Contribution Quality & Insight - "Did this persona add something meaningfully new?" (independent of correctness - can be correct but shallow)
                3.Engagement & Communication - "Was this pleasant and effective to interact with?"
                4.Redundancy & Complexity - "Did this persona avoid repeating others?"
                5.Self-Regulation & Scope Awareness - "Did the persona stay in its lane?"
        2.Logging (mostly debug)
            1.Personas loaded
            2.Why they were selected
            3.Who spoke when
            4.who graded whom
            5.what inputs the grader saw
"""

POWER_PERMISSIONS = {
    "filesystem_access",
    "filesystem_write",
    "execute_code",
    "network_access",
    "delegate_agents",
    "modify_personas"
}

COORDINATOR_ROLES = { #for when new coordinator variants are introduced
    "passive_coordinator"
}

from collections import defaultdict
import json
from datetime import date
from uuid import uuid4
from copy import deepcopy
from pathlib import Path
import os
from pathlib import Path


#TODO: Move to client
def manage_personas(paths):
    """
    Entry point for persona management.
    Called when user enters 'personas'.
    """
    personas_path = paths["personas"]
    master_path = paths["master_list"]
    # ---------------------------
    # Load or initialize personas
    # ---------------------------
    personas = []
    master = []
    if Path(personas_path).exists():
        with open(personas_path, "r", encoding="utf-8") as f:
            personas = json.load(f)

    if not Path(master_path).exists():
        master = generate_persona_master_list(personas)
        with open(master_path, "w", encoding="utf-8") as f:
            json.dump(master, f, indent=2)
    else:
        with open(master_path, "r", encoding="utf-8") as f:
            master = json.load(f)

    # ---------------------------
    # Main loop
    # ---------------------------
    while True:
        print_master(master)

        choice = input("\nSelect an option: ").strip()

        # ---------------------------
        # Exit
        # ---------------------------
        if choice == "0":
            print("Exiting persona manager.")
            return

        # ---------------------------
        # View personas (summary)
        # ---------------------------
        elif choice == "1":
            print("\nRegistered personas:")
            for p in personas.values():
                status = p.get("status", "active")
                default = " (default)" if p.get("default") else ""
                print(f" - {p['persona_name']} [{status}]{default}")

        # ---------------------------
        # Register persona
        # ---------------------------
        elif choice == "2":

            path = input("Path to persona JSON file: ").strip() # client concern
            try:
                with open(path, "r", encoding="utf-8") as f: #client concern
                    persona_input = json.load(f)
                #Add to registry?
                register_persona(persona_input, personas_path, master_path) # mixed

                # Reload personas & master
                with open(personas_path, "r", encoding="utf-8") as f: # persistence - service
                    personas = json.load(f)

                master = generate_persona_master_list(personas) # registry logic
                with open(master_path, "w", encoding="utf-8") as f: #persistence
                    json.dump(master, f, indent=2)

            except Exception as e:
                print("Registration failed:", e)

        # ---------------------------
        # Set default persona
        # ---------------------------
        elif choice == "3":
            name = input("Persona name to set as default: ").strip()
            found = False

            for p in personas.values():
                if p["persona_name"] == name:
                    found = True
                    p["default"] = True
                else:
                    p["default"] = False

            if not found:
                print("Persona not found.")
                continue

            with open(personas_path, "w", encoding="utf-8") as f: #persist = system
                json.dump(personas, f, indent=2)

            master = generate_persona_master_list(personas) #system change
            with open(master_path, "w", encoding="utf-8") as f:
                json.dump(master, f, indent=2)

            print(f"'{name}' set as default persona.")

        # ---------------------------
        # Activate / deactivate persona #TODO: split into cli/service/persona_registry responsibilities
        # ---------------------------
        elif choice == "4":
            name = input("Persona name: ").strip()
            for p in personas:
                if p["persona_name"] == name:
                    new_status = "inactive" if p.get("status") == "active" else "active"
                    p["status"] = new_status
                    print(f"{name} is now {new_status}")
                    break
            else:
                print("Persona not found.")
                continue

            with open(personas_path, "w", encoding="utf-8") as f:
                json.dump(personas, f, indent=2)

            master = generate_persona_master_list(personas)
            with open(personas_path, "w", encoding="utf-8") as f:
                json.dump(master, f, indent=2)

        # ---------------------------
        # View persona details #TODO: client-side input request (registry returns list)
        # ---------------------------
        elif choice == "5":
            name = input("Persona name: ").strip()
            for p in personas.values():
                if p["persona_name"] == name:
                    print(json.dumps(p, indent=2))
                    break
            else:
                print("Persona not found.")

        # ---------------------------
        # Refresh master list
        # ---------------------------
        elif choice == "6":
            master = generate_persona_master_list(personas)
            with open(master_path, "w", encoding="utf-8") as f:
                json.dump(master, f, indent=2)
            print("Master list refreshed.")

        # ---------------------------
        # Future features
        # ---------------------------
        else:
            print("Invalid option.")

        # ---- FUTURE EXTENSIONS ----
        # - Edit persona metadata
        # - Delete persona (with safety checks)
        # - Launch conversation with selected personas
        # - Show performance history
        # - Permission audit view
        # - Persona conflict detection

#TODO: Session
def load_persona(paths, return_to_menu_fn=None, startup = False,):
    """
    Load and validate a persona for conversation use.

    Args:
        return_to_menu_fn: callable to return control to manage_personas()
                           (optional but recommended)

    Returns:
        dict representing the loaded persona, or None if loading fails
    """
    personas_path = paths["personas"]
    roles_registry = load_roles(paths["roles"])
    # ---------------------------
    # Load persona registry
    # ---------------------------
    if not Path(personas_path).exists():
        print("Persona registry not found.")
        if return_to_menu_fn:
            return_to_menu_fn()
        return None

    with open(personas_path, "r", encoding="utf-8") as f:
        personas = json.load(f)

    if not personas:
        print("No personas registered.")
        if return_to_menu_fn:
            return_to_menu_fn()
        return None

    # ---------------------------
    # Persona selection
    # ---------------------------
    print("\nAvailable personas:")
    persona_list = list(personas.values())
    for idx, p in enumerate(persona_list, start=1):
        status = p.get("status", "active")
        default = " (default)" if p.get("default") else ""
        print(f" {idx}) {p['persona_name']} [{status}]{default}")

    persona = None

    # --- STARTUP MODE: auto-load default ---
    if startup: #first load automatically loads default
        persona = next((p for p in persona_list if p.get("default")), None)
        if not persona:
            print("No default persona set.")
            if return_to_menu_fn:
                return_to_menu_fn()
            return None

    # --- INTERACTIVE MODE ---
    else:
        selection = input("\nSelect persona by number (or press ENTER for default): ").strip()

        if not selection:
            persona = next((p for p in persona_list if p.get("default")), None)
            if not persona:
                print("No default persona set.")
                if return_to_menu_fn:
                    return_to_menu_fn()
                return None
        else:
            try:
                idx = int(selection) - 1
                persona = persona_list[idx]
            except (ValueError, IndexError):
                print("Invalid selection.")
                if return_to_menu_fn:
                    return_to_menu_fn()
                return None

    # ---------------------------
    # Second-pass verification
    # ---------------------------

    # 1️⃣ Status check
    if persona.get("status", "active") != "active":
        print(f"Persona '{persona['persona_name']}' is inactive.")
        if return_to_menu_fn:
            return_to_menu_fn()
        return None

    # 2️⃣ System prompt existence
    system_prompt_path = persona.get("system_prompt_path")
    if not system_prompt_path:
        print("Persona is missing system_prompt_path.")
        if return_to_menu_fn:
            return_to_menu_fn()
        return None

    if not Path(system_prompt_path).exists():
        print(f"System prompt file not found: {system_prompt_path}")
        if return_to_menu_fn:
            return_to_menu_fn()
        return None

    # 3️⃣ Load system prompt
    try:
        with open(system_prompt_path, "r", encoding="utf-8") as f:
            system_prompt_text = f.read().strip()
    except Exception as e:
        print("Failed to load system prompt:", e)
        if return_to_menu_fn:
            return_to_menu_fn()
        return None

    # 4️⃣ Background file (optional)
    background_path = persona.get("background_file_path")
    background_text = None

    if background_path:
        if Path(background_path).exists():
            with open(background_path, "r", encoding="utf-8") as f:
                background_text = f.read().strip()
        else:
            print(
                f"Warning: background file not found ({background_path}). "
                "Some persona functions may be limited."
            )

    # ---------------------------
    # System prompt assembly (PLACEHOLDER)
    # ---------------------------
    # TODO:
    # - Combine system_prompt_text
    # - Inject persona name, version, roles
    # - Inject safety / permissions summary
    # - Append background_text (if present)
    # - Append prior feedback report (future)
    # - Append performance score & rank (future)
    #
    # assembled_prompt = ...
    sys_prompt = system_prompt_text

    #roles_registry = load_roles("Personas/roles.json")  # however you're loading it

    sys_prompt = assemble_sys_prompt(
        base_prompt=sys_prompt,
        persona=persona,
        roles_registry=roles_registry
    )

    # sys_prompt now ready to pass to model

    # ---------------------------
    # Role injection (PLACEHOLDER)
    # ---------------------------
    # TODO:
    # - Translate persona["roles"] into behavioral constraints
    # - Limit tool usage based on persona["permissions"]
    # - Prepare role routing metadata for multi-persona conversations

    # ---------------------------
    # Final loaded persona object
    # ---------------------------
    loaded_persona = {
        "persona_id": persona.get("persona_id"),
        "persona_name": persona["persona_name"],
        "system_prompt": sys_prompt,
        "background": background_text,
        "roles": persona.get("roles", []),
        "permissions": persona.get("permissions", []),
        "version": persona.get("version"),
    }

    print(f"\nLoaded persona: {persona['persona_name']}")
    return loaded_persona

#variation of load_persona to allow for multi-persona interactions - being updated to create individual system prompts and include coordinator handling
def load_personas_legacy(return_to_menu_fn=None, startup = False):
    """
    Load and validate a persona for conversation use.

    Args:
        return_to_menu_fn: callable to return control to manage_personas()
                           (optional but recommended)

    Returns:
        list of dicts representing the loaded personas, or None if loading fails
    layout:
        [GLOBAL CORE CONSTRAINTS]

        ────────────────────────────────
        [PERSONA: bot1 | v0.0 | ACTIVE]
        ────────────────────────────────
        <contents of Personas/bot1 Persona.txt>

        <roles + metadata injected>

        ────────────────────────────────
        [PERSONA: analyst1 | v0.1 | ACTIVE]
        ────────────────────────────────
        <contents of Personas/analyst Persona.txt>

        <roles + metadata injected>

    """
    PERSONAS_PATH = "Personas/personas.json"


    # ---------------------------
    # Load persona registry
    # ---------------------------
    if not Path(PERSONAS_PATH).exists():
        print("Persona registry not found.")
        if return_to_menu_fn:
            return_to_menu_fn()
        return None

    with open(PERSONAS_PATH, "r", encoding="utf-8") as f:
        personas = json.load(f)

    if not personas:
        print("No personas registered.")
        if return_to_menu_fn:
            return_to_menu_fn()
        return None

    # ---------------------------
    # Persona selection
    # ---------------------------
    print("\nAvailable personas:")
    persona_list = list(personas.values())
    for idx, p in enumerate(persona_list, start=1):
        status = p.get("status", "active")
        default = " (default)" if p.get("default") else ""
        print(f" {idx}) {p['persona_name']} [{status}]{default}")

    selected_personas = []

    # --- STARTUP MODE: auto-load default ---
    if startup: #first load automatically loads default
        persona = next((p for p in persona_list if p.get("default")), None)
        if not persona:
            print("No default persona set.")
            if return_to_menu_fn:
                return_to_menu_fn()
            return None
        selected_personas = [persona]

    # --- INTERACTIVE MODE --- #changed to allow multiple personas
    else:
        selection = input("\nSelect persona numbers (comma-separated) (or press ENTER for default): ").strip()

        if startup or not selection:
            persona = next((p for p in persona_list if p.get("default")), None)
            if not persona:
                print("No default persona set.")
                if return_to_menu_fn:
                    return_to_menu_fn()
                return None
            selected_personas = [persona]
        else:
            try:
                indices = [int(i.strip()) - 1 for i in selection.split(",")]
                for idx in indices:
                    selected_personas.append(persona_list[idx])
            except (ValueError, IndexError):
                print("Invalid selection.")
                if return_to_menu_fn:
                    return_to_menu_fn()
                return None
    #TODO: Check if multiple personas were added and automatically add coordinator

    # ---------------------------
    # Second-pass verification
    # ---------------------------

    # 1️⃣ Status check
    for persona in selected_personas:
        if persona.get("status", "active") != "active":
            print(f"Persona '{persona['persona_name']}' is inactive.")
            if return_to_menu_fn:
                return_to_menu_fn()
            return None

        # 2️⃣ System prompt existence
        system_prompt_path = persona.get("system_prompt_path")
        if not system_prompt_path:
            print("Persona is missing system_prompt_path.")
            if return_to_menu_fn:
                return_to_menu_fn()
            return None

        if not Path(system_prompt_path).exists():
            print(f"System prompt file not found: {system_prompt_path}")
            if return_to_menu_fn:
                return_to_menu_fn()
            return None

        # 3️⃣ Load system prompt
        try:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                system_prompt_text = f.read().strip()
        except Exception as e:
            print("Failed to load system prompt:", e)
            if return_to_menu_fn:
                return_to_menu_fn()
            return None

        # 4️⃣ Background file (optional)
        background_path = persona.get("background_file_path")
        background_text = None

        if background_path:
            if Path(background_path).exists():
                with open(background_path, "r", encoding="utf-8") as f:
                    background_text = f.read().strip()
            else:
                print(
                    f"Warning: background file not found ({background_path}). "
                    "Some persona functions may be limited."
                )

    # ---------------------------
    # System prompt assembly (PLACEHOLDER)
    # ---------------------------
    # TODO:
    # - Combine system_prompt_text
    # - Inject persona name, version, roles
    # - Inject safety / permissions summary
    # - Append background_text (if present)
    # - Append prior feedback report (future)
    # - Append performance score & rank (future)
    ## ---------------------------
    # Role injection (PLACEHOLDER)
    # ---------------------------
    # TODO:
    # - Translate persona["roles"] into behavioral constraints
    # - Limit tool usage based on persona["permissions"]
    # - Prepare role routing metadata for multi-persona conversations
    # assembled_prompt = ...
    roles_registry = load_roles("Personas/roles.json")

    #load constraints
    with open("Personas/constraints.txt", "r", encoding="utf-8") as f:
        sys_constraints = f.read().strip()
    sys_prompt = sys_constraints  # core constraints only
    """
        #One shared sys_prompt - personas are metadata only - no persona speaks unless controller allows it
        loaded_personas.append({
            "persona_id": persona.get("persona_id"),
            "persona_name": persona["persona_name"],
            "roles": persona.get("roles", []),
            "permissions": persona.get("permissions", []),
            "version": persona.get("version"),
        })
        print(f"\nLoaded persona: {persona['persona_name']}")
        
                {
          "personas": [ ... ],
          "persona_prompts": {
            "bot1_id": "<assembled system prompt>",
            "bot2_id": "<assembled system prompt>"
          },
          "coordinator_prompt": "<coordinator system prompt or None>"
        }
    """
    #Assemble system prompts for individual personas
    loaded_personas = []
    for persona in selected_personas:
        # 1️⃣ Load persona system prompt file
        persona_prompt_path = persona["system_prompt_path"]
        with open(persona_prompt_path, "r", encoding="utf-8") as f:
            persona_prompt_text = f.read().strip()

        # 2️⃣ Wrap persona prompt with hard boundaries
        persona_block = (
            f"\n\n"
            f"{'=' * 60}\n"
            f"PERSONA: {persona['persona_name']} "
            f"(version {persona.get('version', 'unknown')})\n"
            f"{'=' * 60}\n"
            f"{persona_prompt_text}\n"
        )

        # 3️⃣ Append to system prompt
        sys_prompt += persona_block

        # 4️⃣ Inject structured metadata + roles
        sys_prompt = assemble_sys_prompt(
            base_prompt=sys_prompt,
            persona=persona,
            roles_registry=roles_registry
        )
        loaded_personas.append({
            "persona_id": persona.get("persona_id"),
            "persona_name": persona["persona_name"],
            "roles": persona.get("roles", []),
            "permissions": persona.get("permissions", []),
            "version": persona.get("version"),
        })
        print(f"\nLoaded persona: {persona['persona_name']}")
    return {
    "system_prompt": sys_prompt,
    "personas": loaded_personas
    }

#TODO: Session
def load_personas(paths, return_to_menu_fn=None, startup = False):
    """
    Load and validate a persona for conversation use.

    Args:
        return_to_menu_fn: callable to return control to manage_personas()
                           (optional but recommended)

    Returns:
        list of dicts representing the loaded personas, or None if loading fails
    layout:
        [GLOBAL CORE CONSTRAINTS]

        ────────────────────────────────
        [PERSONA: bot1 | v0.0 | ACTIVE]
        ────────────────────────────────
        <contents of Personas/bot1 Persona.txt>

        <roles + metadata injected>

        ────────────────────────────────
        [PERSONA: analyst1 | v0.1 | ACTIVE]
        ────────────────────────────────
        <contents of Personas/analyst Persona.txt>

        <roles + metadata injected>

    """
    personas_path = paths["personas"]
    roles_path = paths["roles"]
    constraints_path = paths["constraints"]
    # ---------------------------
    # Load persona registry
    # ---------------------------
    if not Path(personas_path).exists():
        print("Persona registry not found.")
        if return_to_menu_fn:
            return_to_menu_fn()
        return None

    with open(personas_path, "r", encoding="utf-8") as f:
        personas = json.load(f)

    if not personas:
        print("No personas registered.")
        if return_to_menu_fn:
            return_to_menu_fn()
        return None

    # ---------------------------
    # Persona selection
    # ---------------------------
    print("\nAvailable personas:")
    persona_list = list(personas.values())
    for idx, p in enumerate(persona_list, start=1):
        status = p.get("status", "active")
        default = " (default)" if p.get("default") else ""
        print(f" {idx}) {p['persona_name']} [{status}]{default}")

    selected_personas = []

    # --- STARTUP MODE: auto-load default ---
    if startup: #first load automatically loads default
        persona = next((p for p in persona_list if p.get("default")), None)
        if not persona:
            print("No default persona set.")
            if return_to_menu_fn:
                return_to_menu_fn()
            return None
        selected_personas = [persona]

    # --- INTERACTIVE MODE --- #changed to allow multiple personas
    else:
        selection = input("\nSelect persona numbers (comma-separated) (or press ENTER for default): ").strip()

        if startup or not selection:
            persona = next((p for p in persona_list if p.get("default")), None)
            if not persona:
                print("No default persona set.")
                if return_to_menu_fn:
                    return_to_menu_fn()
                return None
            selected_personas = [persona]
        else:
            try:
                indices = [int(i.strip()) - 1 for i in selection.split(",")]
                for idx in indices:
                    selected_personas.append(persona_list[idx])
            except (ValueError, IndexError):
                print("Invalid selection.")
                if return_to_menu_fn:
                    return_to_menu_fn()
                return None

    # ---------------------------
    # Second-pass verification
    # ---------------------------

    # 1️⃣ Status check
    for persona in selected_personas:
        if persona.get("status", "active") != "active":
            print(f"Persona '{persona['persona_name']}' is inactive.")
            if return_to_menu_fn:
                return_to_menu_fn()
            return None

        # 2️⃣ System prompt existence
        system_prompt_path = persona.get("system_prompt_path")
        if not system_prompt_path:
            print("Persona is missing system_prompt_path.")
            if return_to_menu_fn:
                return_to_menu_fn()
            return None

        if not Path(system_prompt_path).exists():
            print(f"System prompt file not found: {system_prompt_path}")
            if return_to_menu_fn:
                return_to_menu_fn()
            return None

        # 3️⃣ Load system prompt
        try:
            with open(system_prompt_path, "r", encoding="utf-8") as f:
                system_prompt_text = f.read().strip()
        except Exception as e:
            print("Failed to load system prompt:", e)
            if return_to_menu_fn:
                return_to_menu_fn()
            return None

        # 4️⃣ Background file (optional)
        background_path = persona.get("background_file_path")
        background_text = None

        if background_path:
            if Path(background_path).exists():
                with open(background_path, "r", encoding="utf-8") as f:
                    background_text = f.read().strip()
            else:
                print(
                    f"Warning: background file not found ({background_path}). "
                    "Some persona functions may be limited."
                )

    #Finished verification, start assembly
    #temporary demo loading difference

        #load roles early in case of coordinator
    roles_registry = load_roles(roles_path)
       # load constraints early in case of coordinator
    with open(constraints_path, "r", encoding="utf-8") as f:
        sys_constraints = f.read().strip()  # core constraints only

    #If valid personas and more than 1 loaded, request additional coordinator persona
    if len(selected_personas) > 1:
        print("Would you like to add a conversation coordinator? [y/n]: ", end="")
        response = input().strip().lower()
        if response == "y":
            #TODO: list coordinator objects searched by role in COORDINATOR_ROLES, give user a number for each, then assign to coordinator_persona based on choice
            coordinator_persona = None
            if response == "y":
                # ---------------------------
                # Find coordinator-capable personas
                # ---------------------------
                coordinator_candidates = []

                for p in persona_list:
                    if p.get("status", "active") != "active":
                        continue

                    roles = set(p.get("roles", []))
                    if roles & COORDINATOR_ROLES:
                        coordinator_candidates.append(p)

                if not coordinator_candidates:
                    print("No coordinator personas available. Continuing without coordinator.")
                    coordinator_persona = None
                else:
                    print("\nAvailable coordinators:")
                    for idx, p in enumerate(coordinator_candidates, start=1):
                        print(f" {idx}) {p['persona_name']}")

                    try:
                        selection = input("\nSelect coordinator number: ").strip()
                        choice = int(selection) - 1
                        coordinator = coordinator_candidates[choice]
                    except (ValueError, IndexError):
                        print("Invalid selection. Continuing without coordinator.")
                        coordinator_persona = None
                    else:
                        # ---------------------------
                        # Load coordinator system prompt
                        # ---------------------------
                        coordinator_prompt_path = coordinator.get("system_prompt_path")

                        if not coordinator_prompt_path or not Path(coordinator_prompt_path).exists():
                            print("Coordinator system prompt missing or invalid. Continuing without coordinator.")
                            coordinator_persona = None
                        else:
                            with open(coordinator_prompt_path, "r", encoding="utf-8") as f:
                                coordinator_prompt_text = f.read().strip()

                            # ---------------------------
                            # Assemble coordinator system prompt
                            # ---------------------------
                            coordinator_system_prompt = assemble_sys_prompt(
                                base_prompt=sys_constraints + "\n\n" + coordinator_prompt_text,
                                persona=coordinator,
                                roles_registry=roles_registry
                            )

                            coordinator_persona = {
                                "persona_id": coordinator.get("persona_id"),
                                "persona_name": coordinator["persona_name"],
                                "roles": coordinator.get("roles", []),
                                "permissions": coordinator.get("permissions", []),
                                "version": coordinator.get("version"),
                                "system_prompt": coordinator_system_prompt
                            }

                            print(f"\nLoaded coordinator: {coordinator['persona_name']}")

        else:
            print("Continuing load without coordinator.")
            coordinator_persona = None
    else:
        coordinator_persona = None

    # ---------------------------
    # System prompt assembly (PLACEHOLDER)
    # ---------------------------
    # TODO:
    # - Individual system_prompt_text for each persona in a conversation
    # - Inject persona name, version, roles
    # - Inject safety / permissions summary
    # - Append background_text (if present)
    # - Append prior feedback report (future)
    # - Append performance score & rank (future)
    ## ---------------------------
    # Role injection (PLACEHOLDER)
    # ---------------------------
    # TODO:
    # - Translate persona["roles"] into behavioral constraints
    # - Limit tool usage based on persona["permissions"]
    # - Prepare role routing metadata for multi-persona conversations

    """
    loaded_personas = []
    #Iterate through each selected persona
    for persona in selected_personas:
        sys_prompt = assemble_sys_prompt(
            base_prompt=sys_prompt,
            persona=persona,
            roles_registry=roles_registry
        )
        #One shared sys_prompt - personas are metadata only - no persona speaks unless controller allows it
        loaded_personas.append({
            "persona_id": persona.get("persona_id"),
            "persona_name": persona["persona_name"],
            "roles": persona.get("roles", []),
            "permissions": persona.get("permissions", []),
            "version": persona.get("version"),
        })
        print(f"\nLoaded persona: {persona['persona_name']}")
        WIP 1/30/26: TODO: Keep persona system prompts separate to prevent prompt bleed (happens when together already)
                {
          "personas": [ ... ],
          "persona_prompts": {
            "bot1_id": "<assembled system prompt>",
            "bot2_id": "<assembled system prompt>"
          },
          "coordinator_prompt": "<coordinator system prompt or None>"
        }
    """
    loaded_personas = []
    for persona in selected_personas:
        # System constraints
        cur_prompt = sys_constraints+"\n\n"
        # Load persona system prompt file
        persona_prompt_path = persona["system_prompt_path"]
        with open(persona_prompt_path, "r", encoding="utf-8") as f:
            persona_prompt_text = f.read().strip()
        cur_prompt += persona_prompt_text

        # Inject structured metadata + roles
        sys_prompt = assemble_sys_prompt(
            base_prompt= cur_prompt,
            persona=persona,
            roles_registry=roles_registry
        )
        loaded_personas.append({
            "persona_id": persona.get("persona_id"),
            "persona_name": persona["persona_name"],
            "roles": persona.get("roles", []),
            "permissions": persona.get("permissions", []),
            "version": persona.get("version"),
            "system_prompt": sys_prompt,
            "rank": persona["rank"],
            "performance_points": persona["performance_points"]
        })
        print(f"\nLoaded persona: {persona['persona_name']}")
    return { #persona list with individual system prompts, coordinator object (or none if no coordinator)
    "personas": loaded_personas,
    "coordinator": coordinator_persona
    }


def assemble_sys_prompt(
    base_prompt: str,
    persona: dict,
    roles_registry: dict
) -> str:
    """
    Assemble and append system prompt content for a persona.

    Parameters:
        base_prompt (str): Existing system prompt content (can be empty).
        persona (dict): Persona object.
        roles_registry (dict): Loaded roles.json contents.

    Returns:
        str: Updated system prompt.
    """

    sections = []
    persona_meta_data = []

    # --- Persona Identity ---
    persona_meta_data.append(
        f"You are {persona['persona_name']}.\n"
        f"Persona version: {persona.get('version', 'unknown')}."
    )

    # --- Persona Status / Visibility (optional but useful context) ---
    if persona.get("status"):
        sections.append(f"Persona status: {persona['status']}.")

    # --- Roles Injection ---
    roles = persona.get("roles", [])
    if roles:
        sections.append(
            "You are operating under the following role(s) in this conversation:"
        )

        for role_id in roles:
            role = roles_registry.get(role_id)

            if not role:
                sections.append(
                    f"[WARNING] Role '{role_id}' could not be found in roles registry."
                )
                #Send warning to user as well
                print(f"[WARNING] Role '{role_id}' could not be found in roles registry.")
                print("Available roles: ", roles_registry)
                print("Persona roles: ", roles)
                continue

            # Role header
            sections.append(
                f"\nRole: {role.get('role_name', role_id)} "
                f"(version {role.get('version', 'unknown')})"
            )

            # Description
            if role.get("description"):
                sections.append(f"Description: {role['description']}")

            # Core objectives
            if role.get("core_objectives"):
                sections.append("Core objectives:")
                for obj in role["core_objectives"]:
                    sections.append(f"- {obj}")

            # Behavioral guidelines
            if role.get("behavioral_guidelines"):
                sections.append("Behavioral guidelines:")
                for guide in role["behavioral_guidelines"]:
                    sections.append(f"- {guide}")

            # Constraints
            if role.get("constraints"):
                sections.append("Constraints:")
                for constraint in role["constraints"]:
                    sections.append(f"- {constraint}")

            # Prompt fragment (most important part)
            if role.get("prompt_fragment"):
                sections.append("Role instructions:")
                for line in role["prompt_fragment"]:
                    sections.append(f"- {line}")

    # --- Final Assembly ---
    # Separate sections clearly to reduce prompt bleed
    assembled = "\n\n".join(sections)

    if base_prompt:
        return base_prompt + "\n\n" + assembled
    print("DEBUG: Assembled system prompt: ", assembled)
    return assembled

def register_persona( #TODO: Finish client and service sections
    persona_input: dict,
    personas_path,
    master_path
) -> dict:
    """
    Register a new persona.
    Returns the registered persona object.
    """

    # ---- Validate input ----
    validate_persona_object(persona_input) # registry

    persona = deepcopy(persona_input)
    print("input validated") #Remove print systems, replace with metadata
    # ---- Confirmation checks ----
    reasons = requires_user_confirmation(persona, POWER_PERMISSIONS) #Belongs in CLI code - registry should NEVER block waiting for user input
    if reasons:
        print("\n⚠️ Persona registration requires confirmation:")
        for r in reasons:
            print(" -", r)
        input("\nPress ENTER to confirm, or Ctrl+C to abort...")

    # ---- Load existing personas ---- service loads -> registry should be given list during initialization
    if Path(personas_path).exists():
        with open(personas_path, "r", encoding="utf-8") as f:
            personas = json.load(f)
    else:
        personas = []
    print("loading personas") #start registry
    # ---- Uniqueness checks ----
    existing_names = {p["persona_name"] for p in personas.values()}
    if persona["persona_name"] in existing_names:
        raise ValueError("Persona name already exists")
    print("passed uniqueness check")
    # ---- Assign system-managed fields ----
    persona["persona_id"] = str(uuid4())
    persona["version"] = "0.0"
    persona["date_registered"] = date.today().isoformat()
    persona["rank"] = 0
    persona["performance_points"] = 0
    persona.setdefault("default", False)
    persona.setdefault("status", "active")
    persona.setdefault("visibility", "user")
    print("passed field assignment check")
    # ---- Default persona rule ----
    """
    if persona["default"]:
        for p in personas:
            p["default"] = False
    """
    if persona["default"]:
        for p in personas.values():
            p["default"] = False
    print("passed default persona rule")
    #personas.append(persona) #generates error
    persona_id = persona["persona_id"]#attempted fix1

    personas[persona_id] = persona#end registry

    # ---- Write personas ---- #persist (service)
    with open(personas_path, "w", encoding="utf-8") as f:
        json.dump(personas, f, indent=2, ensure_ascii=False)

    # ---- Regenerate master list ---- #service
    master = generate_persona_master_list(personas.values())
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(master, f, indent=2, ensure_ascii=False)

    print(f"Persona '{persona['persona_name']}' registered successfully.") #keep as raw data for debugging

    return persona

def generate_persona_master_list( #service
    personas: dict,
    schema_version: str = "1.0"
) -> dict:
    """
    Generate a master index of registered personas.

    This function is pure:
    - no file I/O
    - safe to regenerate at any time
    """
    # ---- Normalize personas container ---- #avoid errors with passing keys instead of objects
    if isinstance(personas, dict):
        persona_iter = personas.values()
    else:
        persona_iter = personas
    print("generating master list")
    roles_index = defaultdict(list)
    permissions_index = defaultdict(list)

    total = len(personas)
    print("total: ", total)
    active = 0
    inactive = 0
    default_persona = None

    persona_summaries = []

    for p in persona_iter:
        name = p["persona_name"]
        status = p.get("status", "active")

        if status == "active":
            active += 1
        else:
            inactive += 1

        if p.get("default", False):
            default_persona = name

        for role in p.get("roles", []):
            roles_index[role].append(name)

        for perm in p.get("permissions", []):
            permissions_index[perm].append(name)

        persona_summaries.append({
            "persona_name": name,
            "persona_id": p.get("persona_id"),
            "version": p.get("version", "0.0"),
            "status": status,
            "visibility": p.get("visibility", "user"),
            "roles": p.get("roles", []),
            "permissions": p.get("permissions", []),
            "rank": p.get("rank", 0),
            "performance_points": p.get("performance_points", 0),
            "date_registered": p.get("date_registered")
        })
        print("Iterating through personas for master list: ", p)
    return {
        "generated_at": date.today().isoformat(),
        "schema_version": schema_version,
        "summary": {
            "total_personas": total,
            "active_personas": active,
            "inactive_personas": inactive,
            "default_persona": default_persona
        },
        "roles_index": dict(roles_index),
        "permissions_index": dict(permissions_index),
        "personas": persona_summaries
    }

def requires_user_confirmation(persona: dict, power_permissions: set) -> list[str]: #TODO: CLI_client
    """
    Returns a list of reasons why confirmation is required.
    """

    reasons = []

    if persona.get("default", False):
        reasons.append("Persona is marked as default")

    risky_perms = set(persona.get("permissions", [])) & power_permissions
    if risky_perms:
        reasons.append(f"Persona requests power permissions: {sorted(risky_perms)}")

    if persona.get("visibility", "user") != "user":
        reasons.append(f"Persona visibility is '{persona['visibility']}'")

    return reasons

#Transitioned to persona_registry; continue edits there
def validate_persona_object(persona: dict) -> None:
    """
    Validate a user-submitted persona object.
    Raises ValueError on failure.
    """

    required_fields = {
        "persona_name": str,
        "system_prompt_path": str,
        "roles": list,
        "permissions": list,
    }

    optional_fields = {
        "status": str,        # active | inactive
        "visibility": str,    # user | internal | hidden
        "default": bool,
        "background_file_path": str,
    }

    # ---- Required fields ----
    for field, expected_type in required_fields.items():
        if field not in persona:
            raise ValueError(f"Missing required field: {field}")
        if not isinstance(persona[field], expected_type):
            raise ValueError(f"Field '{field}' must be {expected_type}")

    # ---- Optional fields ----
    for field, expected_type in optional_fields.items():
        if field in persona and not isinstance(persona[field], expected_type):
            raise ValueError(f"Field '{field}' must be {expected_type}")

    # ---- Enums ----
    if persona.get("status", "active") not in {"active", "inactive"}:
        raise ValueError("Invalid status value")

    if persona.get("visibility", "user") not in {"user", "internal", "hidden"}:
        raise ValueError("Invalid visibility value")

    # ---- Name sanity ----
    if len(persona["persona_name"].strip()) < 3:
        raise ValueError("persona_name too short")

    # ---- Roles / permissions sanity ----
    if not persona["roles"]:
        raise ValueError("Persona must have at least one role")

    if not all(isinstance(r, str) for r in persona["roles"]):
        raise ValueError("All roles must be strings")

    if not all(isinstance(p, str) for p in persona["permissions"]):
        raise ValueError("All permissions must be strings")

#session
def replace_system_prompt(messages: list[dict], new_sys_prompt: str):
    """
    Replace the system prompt in a runtime message list.
    Removes any existing system messages and inserts a new one at the top.
    Operates in-place
    Parameters:
        messages (dict): list of conversation messages (wrapped at end)
        new_sys_prompt (str): Fully assembled system prompt
    """

    if not isinstance(messages, list):
        raise TypeError("messages must be a list")

    for msg in messages:
        if msg.get("role") == "system":
            msg["content"] = new_sys_prompt
            return

        # No system prompt yet → insert at top
    messages.insert(0, {
        "role": "system",
        "content": new_sys_prompt
    })

def load_roles(path: str) -> dict: #TODO: service loads at startup
    """
    Load roles registry from a JSON file.

    Parameters:
        path (str): Path to roles.json

    Returns:
        dict: Roles registry keyed by role_id
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Roles file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        roles = json.load(f)

    if not isinstance(roles, dict):
        raise ValueError("roles.json must contain a JSON object keyed by role_id")

    return roles


def initialize_conversation_with_persona(persona, roles) -> list[dict]: #TODO: session
    """
    Create a fresh message list with persona-based system prompt.
    """

    messages = []  # empty runtime conversation

    sys_prompt = assemble_sys_prompt("",persona,roles)
    replace_system_prompt(messages, sys_prompt)

    return messages

#TODO: CLI
def print_master(master_list):
    #print master list (moved from main loop for clarity
    summary = master_list.get("summary")
    print("\n=== PERSONA MANAGER ===")
    print(f"Total personas: {summary['total_personas']}")
    print(f"Active personas: {summary['active_personas']}")
    print(f"Default persona: {summary.get('default_persona')}")
    print(f"Current Roles: ")
    print("\nCurrent Roles:")
    roles_index = master_list.get("roles_index", {})
    if not roles_index:
        print(" (none)")
    else:
        for role, members in roles_index.items(): #.items to get the list of roles and members from the object
            print(f" Role: {role}")
            for member in members:
                print(f"   - {member}")

    print("\nCurrent Permissions:")
    permissions_index = master_list.get("permissions_index", {})
    if not permissions_index:
        print(" (none)")
    else:
        for perm, members in permissions_index.items():
            print(f" Permission: {perm}")
            for member in members:
                print(f"   - {member}")
    print("\nActions:")
    print(" 1) View personas")
    print(" 2) Register new persona")
    print(" 3) Set default persona")
    print(" 4) Activate / deactivate persona")
    print(" 5) View persona details")
    print(" 6) Refresh master list")
    print(" 0) Exit")