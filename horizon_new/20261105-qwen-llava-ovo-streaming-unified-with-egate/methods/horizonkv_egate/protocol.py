"""HorizonKV-EGate allocation and answer routing for OVO integration."""
import csv
import hashlib
from pathlib import Path

DEFAULT_ALLOCATION = Path(__file__).parent / "assets/per_head_allocation.csv"

def load_allocation(path=DEFAULT_ALLOCATION):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 112:
        raise ValueError("Expected 112 KV head allocations")
    output = {}
    for idx, row in enumerate(rows):
        layer, head = divmod(idx, 4)
        if (int(row["layer"]), int(row["kv_head"])) != (layer, head):
            raise ValueError("Unexpected KV head order")
        recent, history = int(row["recent_tokens"]), int(row["history_tokens"])
        branch = row["current_branch"]
        if not (32 <= recent <= 500 and 0 <= history <= 468 and recent + history == 500):
            raise ValueError("Invalid per-head visual allocation")
        if branch not in ("one_frame", "four_frames"):
            raise ValueError("Invalid recent branch")
        output[(layer, head)] = {"recent_group": branch, "recent_quota": recent, "history_quota": history, "total_quota": 500}
    if sum(v["recent_group"] == "one_frame" for v in output.values()) != 56:
        raise ValueError("Expected 56 one-frame heads")
    return output, hashlib.sha256(path.read_bytes()).hexdigest()

def format_mc_question(question, choices, *, recent_gate=False):
    if len(choices) not in range(1, 9):
        raise ValueError("EGate expects one to eight original options")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Empty question")
    options = list(choices)
    gate = chr(65 + len(options)) if recent_gate else None
    if recent_gate:
        options.append("Insufficient visual evidence in the shown recent frames to choose an original answer;")
    text = "Question: " + question + "\nOptions:\n" + "\n".join(f"{chr(65+i)}. {value}" for i, value in enumerate(options))
    text += "\n\nRespond only with the letter corresponding to your chosen option."
    if gate:
        text += f" If the needed event is not visible in the shown frames, choose {gate} instead of guessing."
    text += " Do not include any additional text or explanation in your response."
    return text, gate

def route_answer(recent_choice, history_choice, original_count):
    if original_count not in range(1, 9):
        raise ValueError("EGate expects one to eight original options")
    valid = "ABCDEFGH"[:original_count]
    abstention = chr(65 + original_count)
    if recent_choice not in valid + abstention or history_choice not in valid:
        raise ValueError("Invalid model choice for gate")
    return (history_choice, "history") if recent_choice == abstention else (recent_choice, "recent")
