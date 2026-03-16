import json

with open("/tmp/fork_test_0tlhrcnk/problem_0003.json") as f:
    data = json.load(f)

history = data["full_history"]

for h in history:
    if "output" not in h:
        continue
    turn = h["turn"]
    att = h.get("attempt", 0)
    acc = h.get("accepted", "?")
    ppl = h.get("ppl", 0)
    text = h["output"]
    has_boxed = "\\boxed" in text
    print(f"--- turn {turn} att={att} accepted={acc} ppl={ppl:.2f} has_boxed={has_boxed} ---")
    if turn >= 6:
        print(text[-400:])
        print()

# Also check: what does extract_answer see?
# Reconstruct the prompt history at each turn to see what extract_answer would get
print("=" * 60)
print("Reconstructing accepted history (what goes into prompt[1]):")
print("=" * 60)
accepted_history = []
for h in history:
    if h.get("event") == "fork":
        print(f"\n[turn {h['turn']}] FORK -> history replaced")
        # After fork, we don't know the exact history from here
        continue
    if "output" not in h:
        continue
    if h.get("accepted") is True:
        accepted_history.append(h["output"])
        has_boxed = "\\boxed" in h["output"]
        print(f"  turn {h['turn']} att={h.get('attempt',0)}: ACCEPTED (has_boxed={has_boxed})")
    elif h.get("accepted") is False and h["turn"] == 7:
        # turn 7 was rejected but kept as best
        has_boxed = "\\boxed" in h["output"]
        print(f"  turn {h['turn']} att={h.get('attempt',0)}: REJECTED (has_boxed={has_boxed})")
