"""
    Functionality for memory refinement
        -Assign a content and semantic hash to each memory
        -Archive duplicate memories
        -merge similar ideas

    Memory object structure additions after refinement
    {
        "uid": "mem_01J2F5G9ZC8KX3R8V6TQ",
        "hashes": { ... },
        "memory": { ... }, (replaced content)
        "refinement": {
            "hash_assigned_at": "2026-01-12",
            "hash_schema_version": "1.0"
        },
        "legacy": {
            "local_id": 0
            }
    }
"""


import hashlib
import re
import ulid
import json
from datetime import date
from copy import deepcopy
#load prompt
from pathlib import Path
from string import Template
import os
from datetime import datetime
from typing import List, Tuple, Dict

#from scipy.stats import yulesimon_gen
#semantic embedding using vectors
from sentence_transformers import SentenceTransformer
import math #cosine similarity manual function

model = SentenceTransformer("all-MiniLM-L6-v2")
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMS = 384
EMBEDDING_NORMALIZED = True

"""
    Add object to structure
    "refinement": {
        "hash_assigned_at": "2026-01-12",
        "hash_schema_version": "1.0"
        }

"""
def refine_memories(memFile, archiveFile, client, model_name):
    #assign hashes and validate memories (written directly to file once testing finishes - for now, write to hash_output
    #hash_output = "Memories/meta/generation reports/CMWork_hash.json" #Testing use only - write to file once certain of functions
    hash_output = "Memories/meta/generation reports/hash_test.json"
    memFile = resolve_memory_path(memFile)
    print(f"file:{memFile}")
    memories = load_memories(memFile)
    memories, hash_report = assign_hashes(memories, client, model_name, limit = 5000)
    print(hash_report)
    #load hash-assigned memories #not needed with file assignment done at end
    #memories = load_hash_assigned_memories(hash_output, False)
    #identify duplicates and return a list of valid memories and memories to archive (memories to archive will have metadata included)
    #check for issues with hashing
    #hash_ready = [
     #   m for m in memories
      #  if m.get("refinement", {}).get("hash_status") == "complete"
    #]
    hash_ready = []
    not_processed = [] #list of memories not edited to append at the end
    for m in memories:
        if m.get("refinement", {}).get("hash_status") == "complete":
            hash_ready.append(m)
        else:
            not_processed.append(m)
    if len(not_processed) + len(hash_ready) != len(memories):
        print("Memories not processed:", len(not_processed))
        print("Memories processed", len(hash_ready))
        print("Memory Length", len(memories))
        raise RuntimeError("Memory separation logic failed")

    if not hash_ready:
        raise RuntimeError("No hash-complete memories available")
    hashes = [m["hashes"]["content"] for m in hash_ready]
    unique_hashes = set(hashes)

    print("Total memories:", len(memories))
    print("Unique content hashes:", len(unique_hashes))
    print("Sample hashes:", list(unique_hashes)[:5])
    if len(hashes) > 0:
        unique_ratio = len(set(hashes)) / len(hashes)
        if unique_ratio < 0.5:
            raise RuntimeError("Content hash collapse detected")
    else:
        raise  RuntimeError("Content hash error - no hashes created")
    #filter hard duplicate memories
    kept, archived = archive_hard_duplicates(memories)
    print(f"Kept: {len(kept)}")
    print(f"Archived: {len(archived)}")
    #identify merge candidates and group
    groups = find_merge_candidates(memories)
    #print_groups(groups)
    print("Merge groups:", len(groups))
    print("Avg group size:", sum(len(g) for g in groups) / max(len(groups), 1))
    #filter merge groups
    approved_groups,approved_report = filter_merge_groups(groups)
    #print_groups(approved_groups)
    print(approved_report)
    print("Avg approved group size:", sum(len(g) for g in approved_groups) / max(len(approved_groups), 1))
    # merge similar memories
    tags = load_tags("Memories/meta/tags.json")
    merged_memories, merge_preview, merge_failures, merge_reviews = merge_memory_group(
        approved_groups,
        client,
        model_name,
        tags,
        "Memories/meta/prompts/merge_prompt.txt",
        is_trusted_date_fn=is_trusted_date,
        preview_only=False
    )

    print_merge_summary(merge_preview, merge_failures, merge_reviews)

    #Remove merged memories from kept and append to archive

    #create list of uids from memories that were merged
    merged_source_uids = set()
    for m in merged_memories:
        merged_source_uids.update(m["merged_from"])
    print("merged_source_uids:", merged_source_uids)

    #build final kept list
    """
    final_kept = []
    for mem in kept:
        if mem["uid"] not in merged_source_uids:
            final_kept.append(mem)
    """
    final_kept = [m for m in kept if m["uid"] not in merged_source_uids]
    #final_kept.extend(merged_memories)

    # 4️⃣ Optionally, archive the merged-from memories
    archived_from_merges = [m for m in kept if m["uid"] in merged_source_uids]
    archived.extend(archived_from_merges)

    #invariant check - all UIDs must be unique
    #uids = [m["uid"] for m in final_kept]
    #assert len(uids) == len(set(uids)), "UID collision in final kept"

    # 5️⃣ Sanity check: ensure all UIDs are unique
    uids = [m["uid"] for m in final_kept]
    if len(uids) != len(set(uids)):
        # log duplicates for debugging
        dupes = [uid for uid in uids if uids.count(uid) > 1]
        #print("Duplicate UIDs detected after merge:", dupes)
        raise AssertionError("UID collision in final kept memories")

    # 6️⃣ Optional: log counts
    print(f"Final kept memories: {len(final_kept)}")
    print(f"Total archived memories (including merged-from): {len(archived)}")

    #build final archive list
    final_archived = list(archived)
    arch_memories = [
        m for m in memories
        if m.get("refinement", {}).get("hash_status") == "complete"
    ]
    require_hashed(arch_memories, "final archive list")
    uid_to_memory = {m["uid"]: m for m in arch_memories} #avoid checking non-refined memories
    #append merged memories to archive
    for uid in merged_source_uids:
        mem = uid_to_memory.get(uid)
        if mem:
            mem = mem.copy()
            mem["archived"] = {
                "reason": "merged",
                "merged_into": [
                    m["uid"] for m in merged_memories if uid in m["merged_from"]
                ]
            }
            final_archived.append(mem)
    #invariant check
    archived_uids = {m["uid"] for m in final_archived}
    assert not archived_uids & {m["uid"] for m in final_kept}

    #checks all function results individually one last time (issue with appending everything to keep before check)
    summary = finalize_refinement_guard(
        processed_memories= hash_ready,
        kept_memories=final_kept,
        archived_memories=final_archived,
        merged_memories=merged_memories,
    )
    # merge kept + merged + not processed
    final_memories = final_kept + merged_memories + not_processed

    print("\nFINAL REFINEMENT SUMMARY")
    for k, v in summary.items():
        print(f"{k:>12}: {v}")

    input("\nPress ENTER to confirm and write files, or Ctrl+C to abort...")
    if len(final_memories) + len(final_archived) < len(memories):
        print(len(final_memories), len(final_archived), len(memories))
        raise RuntimeError("Original memories should not be smaller than final memories and archived")
    # backup
    safe_backup_and_write(memFile, final_memories)
    safe_backup_and_write(archiveFile, archived)
    finalize_memory_files(memFile, archiveFile, final_memories, final_archived)

    return

def finalize_refinement_guard(
    processed_memories: list[dict],
    kept_memories: list[dict],
    archived_memories: list[dict],
    merged_memories: list[dict],
    strict: bool = True,
):
    """
    Hard gate before file writes.
    Raises RuntimeError if any invariant fails.
    """
    print("processed_memories type:", type(processed_memories))
    print("first element:", processed_memories[0])
    print("first element type:", type(processed_memories[0]))

    # ---------------------------
    # UID integrity
    # ---------------------------
    processed_uids = {m["uid"] for m in processed_memories}
    kept_uids = {m["uid"] for m in kept_memories}
    archived_uids = {m["uid"] for m in archived_memories}
    merged_uids = {m["uid"] for m in merged_memories}

    if len(kept_uids) != len(kept_memories):
        raise RuntimeError("Duplicate UIDs in kept memories")

    if kept_uids & archived_uids:
        raise RuntimeError("Overlap between kept and archived memories")

    if merged_uids & kept_uids:
        raise RuntimeError("Merged memories appear in kept list")

    # ---------------------------
    # merged_from correctness
    # ---------------------------
    merged_from_uids = set()
    for m in merged_memories:
        mf = m.get("merged_from")
        if not mf:
            raise RuntimeError(f"Merged memory {m['uid']} missing merged_from")
        merged_from_uids.update(mf)

    # 🔑 Only require merged_from to exist within the processed scope
    if not merged_from_uids.issubset(processed_uids):
        raise RuntimeError(
            "merged_from references memories outside the processed scope"
        )

    if merged_from_uids & kept_uids:
        raise RuntimeError("Merged-from memories still present in kept list")

    # ---------------------------
    # Schema validation
    # ---------------------------
    #check kept memories and merged memories separately (merged memories are not assigned a hash until used in a refinement run)
    for m in kept_memories:
        if not validate_memory_object_v1_1(m):
            print("Invalid object detected in kept memories: ", m)
            raise RuntimeError(f"Invalid memory detected: {m['uid']}")
    # Validate merged memories structurally only
    for m in merged_memories:
        if not all(k in m for k in ("uid", "content", "merged_from", "schema_version")):
            print("Invalid merged memory (structural):", m)
            raise RuntimeError(f"Invalid merged memory: {m.get('uid')}")
    # ---------------------------
    # Hash sanity
    # ---------------------------
    content_hashes = [m["hashes"]["content"] for m in kept_memories]
    unique_ratio = len(set(content_hashes)) / max(len(content_hashes), 1)

    if unique_ratio < 0.3:
        raise RuntimeError("Hash collapse detected in kept memories")

    # ---------------------------
    # Size sanity
    # ---------------------------
    delta = len(processed_memories) - (len(kept_memories) + len(archived_memories))
    if abs(delta) > len(processed_memories) * 0.5:
        raise RuntimeError("Unexpected memory count delta")

    # ---------------------------
    # Optional strict mode
    # ---------------------------
    if strict and not merged_memories and len(processed_memories) > 100:
        print("Warning: no merges detected in large dataset")

    if strict:
        untouched = processed_uids - (kept_uids | archived_uids | merged_from_uids)
        if untouched:
            print(
                f"Warning: {len(untouched)} processed memories "
                "were not classified (kept/archived/merged)"
            )
    return {
        "processed": len(processed_memories),
        "kept": len(kept_memories),
        "archived": len(archived_memories),
        "merged": len(merged_memories),
        "merged_from": len(merged_from_uids),
    }

def load_memories(memFile):
    with open(memFile, "r", encoding="utf-8") as f:
        memories = json.load(f)
    return memories
def finalize_memory_files(
    mem_file: str,
    archive_file: str,
    final_kept_memories: list,
    final_archived_memories: list,
    backup_dir: str = "Memories/meta/backups"
):
    """
    Safely writes refined memories and archive with backup + atomic replace.
    """

    print("\n🔒 Finalizing memory refinement")

    # ---- Backup originals ----
    mem_backup = backup_file(mem_file, backup_dir)
    print(f"Backup created: {mem_backup}")

    if os.path.exists(archive_file):
        archive_backup = backup_file(archive_file, backup_dir)
        print(f"Archive backup created: {archive_backup}")

    # ---- Write archive first (append-only logic assumed upstream) ----
    atomic_write_json(archive_file, final_archived_memories)
    print(f"Archive written: {archive_file}")

    # ---- Write refined memory file ----
    atomic_write_json(mem_file, final_kept_memories)
    print(f"Memory file written: {mem_file}")

    print("✅ Refinement finalized successfully")

"""pseudocode
def assign_hashes(memFile, semantic_hash):
    #memFile - memory file to assign hashes to
    #semantic_hash - whether the program should assign semantic hashes (call AI)
    #for each memory in memFile
        #check for UID and assign if not present (TODO:implement UID assignment during ingestion phase)
        #retrieve the content
        #assign content hash
        #assign semantic hash if semantic_hash

    #post processing
    #Testing - write out new memories to memFileTest.json
    #after testing - replace updated entries in memFile with new memory objects

    return
    Implementation below:
"""
def backup_and_write(path, memories):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{path}.bak_{ts}"

    shutil.copy2(path, backup_path)
    print(f"Backup created: {backup_path}")

    with open(path, "w", encoding="utf-8") as f:
        json.dump(memories, f, indent=2, ensure_ascii=False)
import os
import shutil
import json
from datetime import datetime

def safe_backup_and_write(path: str, memories: list):
    """
    Safely write memories to a JSON file with timestamped backup.
    - Creates a backup if the file exists.
    - Writes to a temporary file first to avoid partial writes.
    - Moves the temp file to the final path only after successful write.
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # Create a timestamped backup if the original file exists
    if os.path.exists(path):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{path}.bak_{ts}"
        shutil.copy2(path, backup_path)
        print(f"Backup created: {backup_path}")
    else:
        print(f"No existing file found at {path}, skipping backup.")

    # Write to a temporary file first
    temp_path = f"{path}.tmp"
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(memories, f, indent=2, ensure_ascii=False)
        # Move temp file to the final path
        shutil.move(temp_path, path)
        print(f"Memories successfully written to {path}")
    except Exception as e:
        # If something goes wrong, remove temp file and raise
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise RuntimeError(f"Failed to write memories to {path}: {e}")

def print_merge_summary(merge_preview, merge_failures, merge_reviews):
    print(f"\nMERGE PREVIEW SUMMARY")
    print(f"Groups merged: {len(merge_preview)}")
    for p in merge_preview[:5]:
        print("-" * 60)
        print("Merged:", p["source_uids"])
        print("Before:", p["source_contents"])
        print("After :", p["merged_content"])

    for p in merge_failures:
        print("-" * 60)
        print("Merge Failure - Reason: ", p["reason"])
        print("UIDs: ", p["group_uids"])


    for p in merge_reviews:
        print("-" * 60)
        mem = p.get("merged_memory")
        print("Memory created: ", mem["content"])
        print("Confidence: ", p["confidence"])

#debug for groups of memories
def print_groups(groups):
    g = 0
    for group in groups:
        print(f"Group: ", g)
        for mem in group:
            print(mem["content"])
        g += 1


from copy import deepcopy
from datetime import date
from typing import List, Dict, Tuple

def assign_hashes(
    memories: List[Dict],
    client,
    model_name: str,
    *,
    assign_semantic: bool = True,
    refiner_name: str = "memory_refiner_v1",
    limit: int | None = None,
) -> Tuple[List[Dict], Dict]:
    """
    Assign UIDs, hashes, and embeddings to memory objects.
    Pure in-memory transformation — no file writes.

    Returns:
        updated_memories, report
    """

    updated_memories: List[Dict] = []
    today = date.today().isoformat()

    processed = 0
    skipped = 0
    legacy_upgraded = 0

    for memory in memories:
        if limit is not None and processed >= limit:
            skipped += 1
            memory.setdefault("refinement", {})
            memory["refinement"]["hash_status"] = "pending"
            updated_memories.append(memory)  # preserve untouched
            continue

        mem = deepcopy(memory)
        assert "content" in mem, f"Missing content in memory {mem.get('uid')}"

        # ---------------------------
        # Schema version upgrade
        # ---------------------------
        schema_version = mem.get("schema_version", "1.0")
        if schema_version < "1.1":
            mem["schema_version"] = "1.1"
            if "id" in mem:
                mem.setdefault("legacy", {})
                mem["legacy"]["ingestion_id"] = mem.pop("id")
            legacy_upgraded += 1

        # ---------------------------
        # UID assignment
        # ---------------------------
        if not mem.get("uid"):
            mem["uid"] = generate_memory_uid()

        # ---------------------------
        # Hash container
        # ---------------------------
        mem.setdefault("hashes", {})

        # ---------------------------
        # Content hash (deterministic)
        # ---------------------------
        if not mem["hashes"].get("content"):
            mem["hashes"]["content"] = content_hash(mem["content"])

        # ---------------------------
        # Semantic hashing + embedding
        # ---------------------------
        if assign_semantic:
            if "semantic_fingerprint" not in mem["hashes"]:
                mem["hashes"]["semantic_fingerprint"] = generate_semantic_fingerprint(
                    mem["content"], client, model_name
                )

            attach_embedding_if_missing(mem, generate_embedding)

        # ---------------------------
        # Refinement metadata
        # ---------------------------
        mem.setdefault("refinement", {})
        mem["refinement"].update({
            "hash_assigned_at": today,
            "hash_schema_version": "1.1",
            "refined_by": refiner_name,
            "date_trust": (
                "trusted"
                if is_trusted_date(mem.get("memory_date", ""))
                else "untrusted"
            ),
        })
        mem["refinement"]["hash_status"] = "complete"
        # ---------------------------
        # Validation
        # ---------------------------
        validate_memory_object_v1_1(mem)

        updated_memories.append(mem)
        processed += 1

    report = {
        "total_input": len(memories),
        "processed": processed,
        "skipped_due_to_limit": skipped,
        "legacy_upgraded": legacy_upgraded,
        "semantic_hashing": assign_semantic,
        "limit": limit,
    }

    return updated_memories, report

def require_hashed(memories, context: str = ""):
    bad = [
        m["uid"]
        for m in memories
        if m.get("refinement", {}).get("hash_status") != "complete"
    ]
    if bad:
        raise RuntimeError(
            f"{context} received {len(bad)} un-hashed memories "
            f"(example: {bad[:3]})"
        )

from copy import deepcopy
from datetime import datetime
from collections import defaultdict
from typing import TypedDict, List, Dict, Any

class MergeGroup(TypedDict):
    memories: List[Dict[str, Any]]
    similarities: List[float]
    avg_similarity: float
    max_similarity: float

def find_merge_candidates(
    all_memories: List[Dict],
    min_similarity: float = 0.80,
    max_similarity: float = 0.95,
) -> List[MergeGroup]:
    """
    Identify soft-duplicate memory groups suitable for AI merging.

    Returns:
        List of lists of memory dicts (each list size >= 2)
    """
    memories = [
        m for m in all_memories
        if m.get("refinement", {}).get("hash_status") == "complete"
    ]
    require_hashed(memories)
    # ---- Step 1: group by semantic fingerprint ----
    fingerprint_groups = defaultdict(list)

    for mem in memories:
        # Skip already-archived memories defensively
        if "archived" in mem:
            print("archived memory detected, continuing")
            continue

        fingerprint = mem["hashes"].get("semantic_fingerprint")
        if not fingerprint:
            print("no fingerprint found")
            continue
        fingerprint_groups[fingerprint].append(mem)

    merge_groups: List[MergeGroup] = []
    print(f"groups size: {len(fingerprint_groups)}")
    # ---- Step 2: within each fingerprint group, find similarity clusters ----
    for group in fingerprint_groups.values():
        if len(group) < 2:
            continue

        used = set()

        for i, base in enumerate(group):
            if base["uid"] in used:
                continue

            cluster = [base]
            used.add(base["uid"])

            for other in group[i + 1:]:
                if other["uid"] in used:
                    continue

                # Mechanical vetoes
                if base["memory_type"] != other["memory_type"]:
                    continue
                if base["memory_domain"] != other["memory_domain"]:
                    continue
                if not (base["shared"] and other["shared"]):
                    continue
                if base["owner"] != other["owner"]:
                    continue

                sim = hash_cosine_similarity(
                    base["hashes"]["semantic_embedding"],
                    other["hashes"]["semantic_embedding"]
                )

                if min_similarity <= sim < max_similarity:
                    cluster.append(other)
                    used.add(other["uid"])

            if len(cluster) < 2:
                continue

            # ✅ compute pairwise similarities for the final cluster
            similarities: List[float] = []
            for a in range(len(cluster)):
                for b in range(a + 1, len(cluster)):
                    similarities.append(
                        hash_cosine_similarity(
                            cluster[a]["hashes"]["semantic_embedding"],
                            cluster[b]["hashes"]["semantic_embedding"]
                        )
                    )

            merge_group: MergeGroup = {
                "memories": cluster,
                "similarities": similarities,
                "avg_similarity": sum(similarities) / len(similarities) if similarities else 0,
                "max_similarity": max(similarities) if similarities else 0,
            }

            merge_groups.append(merge_group)

    return merge_groups

from typing import List, Tuple
#revision for MergeGroup object
def filter_merge_groups(
    merge_groups: list[MergeGroup],
    require_same_domain: bool = True,
    require_same_type: bool = True,
    require_same_owner: bool = True,
    require_same_shared: bool = True,
    allow_cross_origin: bool = True,
    report: dict | None = None,
) -> tuple[List[MergeGroup], dict]:
    """
    Filters merge candidate groups based on strict compatibility rules.
    """
    scope_groups = []
    for group in merge_groups:
        scope_groups.append(group.get("memories"))

    approved_groups = []
    report = report if report is not None else defaultdict(int)

    for group in merge_groups:
    #for group in scope_groups:
        mem_group = group.get("memories")
        if len(mem_group) < 2:
            report["too_small"] += 1
            continue
        domains = {m["memory_domain"] for m in mem_group}
        types = {m["memory_type"] for m in mem_group}
        owners = {m.get("owner") for m in mem_group}
        shared_vals = {m.get("shared") for m in mem_group}
        origins = {m.get("origin") for m in mem_group}

        # --- Hard blockers ---
        if require_same_domain and len(domains) > 1:
            report["domain_mismatch"] += 1
            continue

        if require_same_type and len(types) > 1:
            report["type_mismatch"] += 1
            continue

        if require_same_owner and len(owners) > 1:
            report["owner_mismatch"] += 1
            continue

        if require_same_shared and len(shared_vals) > 1:
            report["shared_mismatch"] += 1
            continue

        # --- Origin handling --- TODO: allow for now, reevaluate as needed
        if not allow_cross_origin and len(origins) > 1:
            report["origin_mismatch"] += 1
            continue

        # avoid recursive merging
        if any(m.get("merged_from") for m in mem_group):
            report["already_merged"] += 1
            continue

        if all("memory_date" in m and m["memory_date"] for m in mem_group):
            if dates_block_merge(mem_group):
                print("Date mismatch:")
                for m in mem_group:
                    print("Date:", m["memory_date"])
                report["date_drift"] += 1
                continue

        approved_groups.append(group)
        report["approved"] += 1

    return approved_groups, dict(report)

def merge_memory_group(
    approved_groups: List[MergeGroup],
    client,
    model: str,
    allowed_tags: list[str],
    prompt_template_path: str,
    is_trusted_date_fn,
    preview_only: bool = False,
):
    merged_memories = []
    merge_previews = []
    merge_failures = []
    merge_reviews = []
    #for group in approved_groups:
    for idx, m_group in enumerate(approved_groups):
        group = m_group.get("memories")
        require_hashed(group, "merge_memory_group") # check to ensure all memories are hashed
        # ---------- Phase 1: Build AI prompt ----------
        prompt = build_merge_prompt(group, allowed_tags, prompt_template_path)

        # ---------- Phase 2: AI language merge ----------
        try:
            ai_result = call_ai_merge(client, model, prompt)
        except Exception as e:
            print(f"[MERGE] AI call failed for group {[m['uid'] for m in group]}: {e}")
            merge_failures.append({
                "group_uids": [m["uid"] for m in group],
                "reason": "ai_call_failed",
            })
            continue

        # ---------- Phase 3: Validate AI output ----------
        if not validate_ai_merge_output(ai_result, group):
            print(f"[MERGE] AI output invalid for group {[m['uid'] for m in group]}")
            merge_failures.append({
                "group_uids": [m["uid"] for m in group],
                "reason": "ai_invalid_output",
            })
            continue

        # ---------- Phase 4: Deterministic resolution ----------
        resolved = resolve_deterministic_fields(group)

        # ---------- Phase 5: Date resolution ----------
        for d in group:
            print("Memory date: ", d.get("memory_date"))
        memory_date = resolve_memory_date(group, is_trusted_date_fn)

        # ---------- Phase 6: Assemble merged memory ----------
        merged_memory = {
            "uid": generate_memory_uid(),
            "content": ai_result["content"].strip(),
            "keywords": ai_result.get("keywords", []),
            "tags": list(set(ai_result.get("tags", []))),
            **resolved,
            "memory_date": memory_date,
            "merged_from": [m["uid"] for m in group],
            "schema_version": "1.1",
            "hashes": {},      # assigned later"content": content_hash(ai_result["content"].strip())
            "refinement": {
                "merged_at": datetime.now().strftime("%Y-%m-%d"),
                "merge_policy": "ai_content_only_v1"
                }  # assigned later
            }
        #print(f"[MEMORY] {merged_memory}")
        confidence = compute_merge_confidence(m_group.get("similarities"), group) #TODO: implement confidence checking and manual confirmation check
        print(f"Merge confidence: ",confidence)
        #integrate confidence check WIP
        merged_memory["merge_confidence"] = confidence
        print("test: ",max(m_group.get("similarities", [])))
        print(m_group.get("max_similarity"))
        stats = {
                "min": min(m_group.get("similarities", [])),
                "max": max(m_group.get("similarities", [])),
                "avg": m_group.get("avg_similarity", []),
                "group_size": len(group)
            }
        review = memory_review(
            merged_memory=merged_memory,
            source_memories=group,
            merge_confidence=confidence,
            similarity_stats= stats,
            interactive=True  # flip to True for manual runs
        )

        if review["decision"] == "accept":
            merged_memories.append(merged_memory)

        elif review["decision"] == "review":
            merge_previews.append({
                "merged_memory": merged_memory,
                "confidence": confidence,
                "status": "needs_review",
            })

        else:  # reject
            merge_failures.append({
                "group_uids": [m["uid"] for m in group],
                "reason": review["reason"],
            })

        #end
        preview = build_merge_preview(group, merged_memory, idx)
        #print(f"debug: preview: {preview}")
        merge_previews.append(preview)

        merge_reviews.append({
            "merged_memory": merged_memory,
            "source_memories": [m["uid"] for m in group],
            "confidence": confidence,
            "similarity_stats": stats,
            "decision": review["decision"],
        })

    return merged_memories, merge_previews, merge_failures, merge_reviews

def compute_merge_confidence(
    semantic_similarities: list[float],
    group: list[dict],
) -> float:
    if not semantic_similarities:
        return 0.0

    avg_similarity = sum(semantic_similarities) / len(semantic_similarities)
    max_similarity = max(semantic_similarities)

    group_size_factor = min(len(group) / 3.0, 1.0)  # caps at 3 memories

    stability_values = [m["stability"] for m in group]
    stability_penalty = (max(stability_values) - min(stability_values)) / 100.0

    date_trust_bonus = 0.1 if any(m.get("memory_date") for m in group) else 0.0

    confidence = (
        (0.6 * avg_similarity) +
        (0.2 * max_similarity) +
        (0.15 * group_size_factor) +
        date_trust_bonus -
        stability_penalty
    )

    return round(max(0.0, min(confidence, 1.0)), 3)


def resolve_deterministic_fields(group: List[dict]) -> dict:
    return {
        "memory_type": group[0]["memory_type"],
        "memory_domain": group[0]["memory_domain"],
        "owner": group[0].get("owner"),
        "origin": resolve_origin(group),
        "shared": group[0].get("shared"),
        "salience": max(m["salience"] for m in group),
        "stability": min(m["stability"] for m in group),
        "certainty": min(m["certainty"] for m in group)
    }

ORIGIN_PRIORITY = ["user", "assistant", "persona"]
def resolve_origin(group: list[dict]) -> str | None:
    origins = {m.get("origin") for m in group if m.get("origin")}

    for o in ORIGIN_PRIORITY:
        if o in origins:
            return o

    return None
def memory_review(
    *,
    merged_memory: dict,
    source_memories: list,
    merge_confidence: float,
    similarity_stats: dict,
    auto_accept_threshold: float = 0.85,
    reject_threshold: float = 0.55,
    interactive: bool = False,
) -> dict:
    """
    Human-aware merge review helper.
    """

    decision = {
        "decision": None,
        "reason": None,
        "confidence": merge_confidence,
        "merged_from": [m["uid"] for m in source_memories],
        "uid": merged_memory.get("uid"),
    }

    # ---------- Auto reject ----------
    if merge_confidence < reject_threshold:
        decision["decision"] = "reject"
        decision["reason"] = "confidence below reject threshold"
        return decision

    # ---------- Auto accept ----------
    if merge_confidence >= auto_accept_threshold:
        decision["decision"] = "accept"
        decision["reason"] = "confidence above auto-accept threshold"
        return decision

    # ---------- Human review ----------
    decision["decision"] = "review"
    decision["reason"] = "manual review required"

    if not interactive:
        return decision

    # ---------- Interactive display ----------
    print("\n================ MERGE REVIEW =================")
    print(f"Merge confidence: {merge_confidence:.3f}")
    print(f"Similarity stats: {similarity_stats}")

    print("\n--- SOURCE MEMORIES ---")
    for i, m in enumerate(source_memories, 1):
        print(f"\n[{i}] UID: {m['uid']}")
        print(f"Date: {m.get('memory_date')}")
        print(f"Origin: {m['origin']} | Owner: {m.get('owner')}")
        print(f"Type / Domain: {m['memory_type']} / {m['memory_domain']}")
        print(f"Salience/Stability/Certainty: "
              f"{m['salience']}/{m['stability']}/{m['certainty']}")
        print(f"Tags: {m['tags']}")
        print("Content:")
        print(m['content'])

    print("\n--- PROPOSED MERGED MEMORY ---")
    print(f"Resolved origin: {merged_memory['origin']}")
    print(f"Resolved date: {merged_memory.get('memory_date')}")
    print(f"Tags: {merged_memory['tags']}")
    print("Merged content:")
    print(merged_memory['content'])

    print("\nAccept merge? [y/n]: ", end="")
    response = input().strip().lower()

    if response == "y":
        decision["decision"] = "accept"
        decision["reason"] = "accepted by human reviewer"
    else:
        decision["decision"] = "reject"
        decision["reason"] = "rejected by human reviewer"

    return decision


def validate_ai_merge_output(ai_out: dict, source_group: List[dict]) -> bool:
    if not isinstance(ai_out, dict):
        return False

    if "content" not in ai_out or len(ai_out["content"].strip()) < 20:
        return False

    if len(ai_out["content"]) > 200:
        return False

    if not (1 <= len(ai_out.get("keywords", [])) <= 5):
        return False

    # Tag safety: must be subset of union of source tags
    source_tags = set()
    for m in source_group:
        source_tags.update(m.get("tags", []))

    for t in ai_out.get("tags", []):
        if t not in source_tags:
            return False

    return True

def call_ai_merge(
    client,
    model: str,
    prompt: str,
    temperature: float = 0.2,
) -> dict:
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=[
            {"role": "system", "content": "You merge memory language conservatively."},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content.strip()
    return json.loads(content)

PROJECT_START_DATE = datetime(2025, 10, 1)
#decide whether an individual date is hallucinated by referring to the project start date (old model used 2023/2024)
def is_trusted_date(date_str: str) -> bool:
    try:
        #d = datetime.strptime(date_str, "%m/%d/%Y") TODO: Decide on memory date format
        d = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return False

    # Reject hallucinated past dates
    if d < PROJECT_START_DATE:
        return False

    # Reject implausible future dates (optional but recommended)
    if d > datetime.now():
        return False

    return True

#select safe date from group
def resolve_memory_date(group: List[dict], is_trusted_date_fn) -> str | None:
    trusted_dates = []
    print("Trusted dates test:")
    for m in group:
        d = m.get("memory_date")
        print(d)
        if d and is_trusted_date_fn(d):
            trusted_dates.append(datetime.strptime(d, "%Y-%m-%d"))

    if not trusted_dates:
        print("No trusted dates found")
        return None
    print("Trusted date found")
    return min(trusted_dates).strftime("%Y-%m-%d")
from datetime import datetime, timedelta
MAX_ALLOWED_DATE_DRIFT = timedelta(days=180)
def dates_block_merge(group: list[dict]) -> bool:
    dates = []
    for m in group:
        date_str = m.get("memory_date")
        if date_str and is_trusted_date(date_str):
            dates.append(datetime.strptime(date_str, "%Y-%m-%d"))

    # If fewer than 2 trusted dates, do NOT block merge
    if len(dates) < 2:
        return False

    return max(dates) - min(dates) > MAX_ALLOWED_DATE_DRIFT

#helper for filter_merge_groups
def dates_too_far_apart(group, max_days=180):
    dates = []
    for m in group:
        if "memory_date" in m:
            try:
                dates.append(datetime.fromisoformat(m["memory_date"]))
            except ValueError:
                pass

    if len(dates) < 2:
        return False

    return (max(dates) - min(dates)).days > max_days

def archive_hard_duplicates(
    memories: List[Dict],
    similarity_threshold: float = 0.95,
) -> Tuple[List[Dict], List[Dict]]:
    """
    Identify hard-duplicate memories and return kept vs archived.

    Hard duplicates are identified by identical content hashes.
    (If you later add fuzzy content-hash similarity, it plugs in here.)

    Returns:
        kept_memories: list of memory dicts to keep active
        archived_memories: list of memory dicts enriched with archive metadata
    """

    # Defensive copy: never mutate caller data
    all_memories = [deepcopy(m) for m in memories]

    memories = [
        m for m in all_memories
        if m.get("refinement", {}).get("hash_status") == "complete"
    ]
    require_hashed(memories, "archive_hard_duplicates")#additional check due to maintaining full list in memory

    # content_hash -> survivor memory
    survivors: Dict[str, Dict] = {}
    archived_memories: List[Dict] = []

    for memory in memories:
        content_hash = memory["hashes"]["content"]

        if content_hash not in survivors:
            survivors[content_hash] = memory
            continue

        existing = survivors[content_hash]

        survivor, loser = choose_survivor(existing, memory)

        # Update survivor reference if needed
        survivors[content_hash] = survivor

        archived_memories.append(
            mark_as_archived(
                loser=loser,
                survivor_uid=survivor["uid"],
                reason="hard_duplicate"
            )
        )

    kept_memories = list(survivors.values())

    return kept_memories, archived_memories

def choose_survivor(a: Dict, b: Dict) -> Tuple[Dict, Dict]:
    """
    Determine which memory survives based on deterministic rules.

    Priority order:
    1. Higher certainty
    2. Higher stability
    3. Higher salience
    4. Newer memory_date
    5. Lexicographically smaller uid (final tie-breaker)
    """

    fields = ["certainty", "stability", "salience"]

    for field in fields:
        if a[field] != b[field]:
            return (a, b) if a[field] > b[field] else (b, a)

    # Memory date comparison (optional field)
    date_a = parse_date_safe(a.get("memory_date"))
    date_b = parse_date_safe(b.get("memory_date"))

    if date_a != date_b:
        return (a, b) if date_a > date_b else (b, a)

    # Final deterministic tie-breaker
    return (a, b) if a["uid"] < b["uid"] else (b, a)

def mark_as_archived(
    loser: Dict,
    survivor_uid: str,
    reason: str
) -> Dict:
    """
    Attach archive metadata to a memory.
    """

    archived = deepcopy(loser)

    archived["archived"] = {
        "archived_at": datetime.now().strftime("%Y-%m-%d"),
        "reason": reason,
        "superseded_by": survivor_uid
    }

    return archived

def parse_date_safe(date_str):
    """
    Parse date string safely.
    Missing or invalid dates are treated as the oldest possible date.
    """

    if not date_str:
        return datetime.min

    try:
        return datetime.strptime(date_str, "%m/%d/%Y")
    except ValueError:
        return datetime.min


def validate_memory_object_v1_1(memory: dict, tags_list: list = None, strict: bool = True) -> bool:
    """
    Validate a single memory object against the refined schema v1.1

    Args:
        memory: dict representing the memory object
        tags_list: optional list of allowed tags
        strict: enforce tag validity if True

    Returns:
        True if valid, False otherwise
    """

    # ---------------------------
    # Required fields for v1.1
    # ---------------------------
    required_fields = {
        "uid": str,
        "content": str,
        "memory_type": str,
        "memory_domain": str,
        "tags": list,
        "owner": (str, type(None)),
        "origin": str,
        "shared": bool,
        "salience": int,
        "stability": int,
        "certainty": float,
        "keywords": list,
        "schema_version": str,
        "hashes": dict,
        "refinement": dict,
    }

    optional_fields = {
        "legacy": dict,
        "memory_date": str,
        "merged_from": list
    }

    # ---------------------------
    # Check required fields exist and types
    # ---------------------------
    for field, expected_type in required_fields.items():
        if field not in memory:
            print(f"Invalid Memory: missing required field '{field}'")
            return False
        if not isinstance(memory[field], expected_type):
            print(f"Invalid Memory: field '{field}' expected type {expected_type}, got {type(memory[field])}")
            return False

    # ---------------------------
    # Optional fields type check
    # ---------------------------
    for field, expected_type in optional_fields.items():
        if field in memory and not isinstance(memory[field], expected_type):
            print(f"Invalid Memory: optional field '{field}' wrong type")
            return False

    # ---------------------------
    # Enum checks
    # ---------------------------
    if memory["memory_type"] not in {"emotional", "work", "factual", "preference"}:
        print("Invalid Memory - memory_type enum failed")
        return False

    if memory["memory_domain"] not in {"personal", "project", "system", "meta"}:
        print("Invalid Memory - memory_domain enum failed")
        return False

    if memory["origin"] not in {"user", "assistant", "persona"}:
        print("Invalid Memory - origin enum failed")
        return False

    """ TODO: decide about ownership coherence (do I want to have personas remember interactions with each other?)
    if memory["origin"] == "persona" and not memory["owner"]:
        print("Invalid Memory - persona-origin requires owner")
        return False
    """
    # ---------------------------
    # Range checks
    # ---------------------------
    if not (1 <= memory["salience"] <= 100):
        print("Invalid Memory - salience out of range")
        return False

    if not (1 <= memory["stability"] <= 100):
        print("Invalid Memory - stability out of range")
        return False

    if not (0.0 <= memory["certainty"] <= 1.0):
        print("Invalid Memory - certainty out of range")
        return False

    # ---------------------------
    # Content checks
    # ---------------------------
    if len(memory["content"].strip()) < 20:
        print("Invalid Content - memory too short")
        return False

    if not (1 <= len(memory["keywords"]) <= 5):
        print("Invalid Memory - keywords length invalid")
        return False
    if not all(isinstance(k, str) for k in memory["keywords"]):
        print("Invalid Memory - keywords must be strings")
        return False

    """ TODO: decide if tags can be empty []
    if not memory["tags"]:
        print("Invalid Memory - tags cannot be empty")
        return False
    """
    # ---------------------------
    # Hash checks
    # ---------------------------
    hashes = memory["hashes"]
    if "content" not in hashes or not hashes["content"]:
        print("Invalid Memory - content hash missing")
        return False

    embedding = memory["hashes"].get("semantic_embedding")
    if embedding is not None:
        if not isinstance(embedding, list) or not all(isinstance(x, float) for x in embedding):
            return False

    fingerprint = hashes.get("semantic_fingerprint")
    if fingerprint is not None and not isinstance(fingerprint, str):
        print("Invalid Memory - semantic_fingerprint must be string")
        return False
    # ---------------------------
    # Refinement metadata check
    # ---------------------------
    if not memory["uid"].startswith("mem_"):
        print("Invalid Memory - uid format invalid")
        return False
    refinement = memory["refinement"]
    if not isinstance(refinement.get("hash_assigned_at"), str):
        print("Invalid Memory - refinement.hash_assigned_at missing or invalid")
        return False
    if not isinstance(refinement.get("hash_schema_version"), str):
        print("Invalid Memory - refinement.hash_schema_version missing or invalid")
        return False
    meta = hashes.get("embedding_metadata")
    if meta is not None:
        required_meta = {"model": str, "dims": int, "normalized": bool, "generated_at": str}
        for k, t in required_meta.items():
            if k not in meta or not isinstance(meta[k], t):
                print(f"Invalid Memory - embedding_metadata.{k} invalid")
                return False
    if embedding is not None and meta is not None:
        if len(embedding) != meta["dims"]:
            print("Invalid Memory - embedding dims mismatch")
            return False
    # ---------------------------
    # Tags checks
    # ---------------------------
    if strict and tags_list:
        for t in memory["tags"]:
            if t not in tags_list:
                print(f"Invalid Memory - tag '{t}' not in allowed list")
                return False

    # ---------------------------
    # Schema version check
    # ---------------------------
    if parse_version(memory["schema_version"]) < (1, 1):
        print("Invalid Memory - schema_version < 1.1")
        return False

    return True

def parse_version(v: str):
    return tuple(map(int, v.split(".")))

def generate_semantic_fingerprint(mem_text, client, model_name):
    prompt_file = "Memories/meta/prompts/gen_semantic_finger_prompt.txt"

    prompt = load_prompt(prompt_file, mem_text)
    #print(f"prompt:\n {prompt}\n")
    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200
    )
    fingerprint = response.choices[0].message.content
    return fingerprint

#UID should be: top-level, never change, never reused
def generate_memory_uid():
    return f"mem_{ulid.new()}"

#ensure consistent memory content formatting
def normalize_content(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

#calculate the content hash for a given memory
def content_hash(text: str) -> str:
    assert isinstance(text, str), "content must be a string"
    assert text.strip(), "content must not be empty"

    normalized = normalize_content(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

#calculate the semantic hash for a given memory
def semantic_hash(fingerprint: str) -> str:
    normalized = normalize_content(fingerprint)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

from datetime import date

#See top for model name, dims, normalized
def attach_embedding(memory: dict, embedding: list[float]) -> None:
    """
    Attach a semantic embedding and metadata to a memory object in-place.
    """

    # Ensure hashes container exists
    memory.setdefault("hashes", {})

    # Attach embedding vector
    memory["hashes"]["semantic_embedding"] = embedding

    # Attach embedding metadata (separate namespace for clarity)
    memory["hashes"]["embedding_metadata"] = {
        "model": EMBEDDING_MODEL_NAME,
        "dims": EMBEDDING_DIMS,
        "normalized": EMBEDDING_NORMALIZED,
        "generated_at": date.today().isoformat()
    }

def attach_embedding_if_missing(memory: dict, embedding_fn) -> bool:
    """
    Attach embedding only if missing or incompatible.
    Returns True if embedding was generated.
    """

    hashes = memory.setdefault("hashes", {})

    meta = hashes.get("embedding_metadata")
    if meta:
        if (
            meta.get("model") == EMBEDDING_MODEL_NAME and
            meta.get("dims") == EMBEDDING_DIMS and
            meta.get("normalized") == EMBEDDING_NORMALIZED
        ):
            return False  # already valid

    embedding = embedding_fn(memory["content"])
    attach_embedding(memory, embedding)
    return True


def generate_embedding(text: str) -> list[float]:
    return model.encode(text, normalize_embeddings=True).tolist()

def semantic_similarity(hash_a, hash_b) -> float:
    """
    Compute similarity between two semantic hashes.
    Replace implementation as needed (simhash, cosine, etc.).
    """
    # Placeholder example for simhash-style similarity
    return 1.0 - (bin(hash_a ^ hash_b).count("1") / 64)


def load_hash_assigned_memories(
    hash_output_file: str,
    include_archived: bool = False
) -> List[Dict]:
    """
    Load hash-assigned memories from a JSON file into a list
    suitable for archive_hard_duplicates().

    Args:
        hash_output_file: path to JSON file produced by assign_hashes
        include_archived: whether to include memories already archived

    Returns:
        List of memory dicts
    """

    with open(hash_output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support both list and wrapped formats
    if isinstance(data, list):
        memories = data
    elif isinstance(data, dict):
        # Common patterns: {"memories": [...]} or similar
        if "memories" in data and isinstance(data["memories"], list):
            memories = data["memories"]
        else:
            raise ValueError(
                f"Unsupported memory file structure in {hash_output_file}"
            )
    else:
        raise ValueError(
            f"Invalid JSON root type in {hash_output_file}"
        )

    if not include_archived:
        memories = [
            m for m in memories
            if "archived" not in m
        ]

    return memories
#for generate memories' prompt
def load_prompt(prompt_path: str, memory: str) -> str:
    """
    Load the extraction prompt from a text file and replace placeholders.
    """
    #print("Debug: Loading prompt...")
    #print(f"Debug: Prompt parameters: path: {prompt_path} , persona: {persona} , tags_list: {tags_list}, chunk_text: {chunk_text}")
    path = Path(prompt_path)
    #print(f"Debug: prompt to load: {path}")
    if not path.exists():
        print("Prompt not found")
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    #print("Debug: opening prompt file...")
    with open(path, "r", encoding="utf-8") as f:
        prompt_text = f.read()
    #print(f"Debug: prompt loaded: {prompt_text}")
    #print("Debug: replacing placeholders...")

    prompt_template = Template(prompt_text)
    prompt_filled = prompt_template.substitute(
        MEMORY_TEXT = memory
    )
    #print(f"Debug: Prompt loaded:{prompt_filled}")
    return prompt_filled

#copied from memories.py into helper function
def load_tags(file_path):
    # Load tags
    tags_file_path = Path(file_path)
    if not tags_file_path.exists():
        raise FileNotFoundError(f"Tags file not found at {tags_file_path}")

    with open(tags_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        tags_list = data.get("tags", [])
    #print(f"DEBUG: Valid tags: {tags_list}")
    if not tags_list:
        raise ValueError("Tags list is empty in tags.json")
    return tags_list

def build_merge_prompt(
    group: list[dict],
    allowed_tags: list[str],
    prompt_template_path: str,
) -> str:
    with open(prompt_template_path, "r", encoding="utf-8") as f:
        template = f.read()

    memories_payload = [
        {
            "content": m["content"],
            "keywords": m.get("keywords", []),
            "tags": m.get("tags", [])
        }
        for m in group
    ]

    return (
        template
        .replace("{{ALLOWED_TAGS}}", json.dumps(allowed_tags))
        .replace("{{MEMORIES_JSON}}", json.dumps(memories_payload, indent=2))
    )
#merge reporting helper
def build_merge_preview(group: list[dict], merged_memory: dict, index: int) -> dict:
    return {
        "group_index": index,
        "source_uids": [m["uid"] for m in group],
        "source_contents": [m["content"] for m in group],
        "merged_content": merged_memory["content"],
        "keywords_before": [m.get("keywords", []) for m in group],
        "keywords_after": merged_memory.get("keywords", []),
        "tags_before": [m.get("tags", []) for m in group],
        "tags_after": merged_memory.get("tags", []),
        "salience_before": [m["salience"] for m in group],
        "salience_after": merged_memory["salience"],
        "stability_before": [m["stability"] for m in group],
        "stability_after": merged_memory["stability"],
    }

def resolve_memory_path(filename, base_dir="Memories"):
    """
    Resolve a filename safely within the Memories directory.
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



def hash_cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not isinstance(vec_a, list) or not isinstance(vec_b, list):
        raise TypeError("cosine_similarity_list expects list[float] inputs")
    if len(vec_a) != len(vec_b):
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)

import os
import shutil
from datetime import datetime

def backup_file(path: str, backup_dir: str = None) -> str:
    """
    Create a timestamped backup of a file.
    Returns the backup path.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot back up missing file: {path}")

    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    base = os.path.basename(path)
    backup_name = f"{base}.bak.{ts}"

    if backup_dir:
        os.makedirs(backup_dir, exist_ok=True)
        backup_path = os.path.join(backup_dir, backup_name)
    else:
        backup_path = os.path.join(os.path.dirname(path), backup_name)

    shutil.copy2(path, backup_path)
    return backup_path

import json
import tempfile

def atomic_write_json(path: str, data, *, indent=2):
    """
    Atomically write JSON data to disk.
    """
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=dir_name,
        delete=False
    ) as tmp:
        json.dump(data, tmp, indent=indent, ensure_ascii=False)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name

    os.replace(tmp_path, path)
