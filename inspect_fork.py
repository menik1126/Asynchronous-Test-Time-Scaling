import json, os, sys

output_dir = sys.argv[1] if len(sys.argv) > 1 else "/tmp/fork_test_0tlhrcnk"

for i in range(4):
    filepath = os.path.join(output_dir, f"problem_{i:04d}.json")
    with open(filepath) as f:
        data = json.load(f)

    history = data["full_history"]
    print(f"====== Sample {i} ======")
    print(f"Final answer: {data['final_answer']}")
    print()

    for h in history:
        if h.get("event") == "fork":
            print(f"  [turn {h['turn']}] === FORK from idx={h['forked_from']} ===")
            print(f"    peer_score={h['peer_score']:.3f}, my_score={h['my_score']:.3f}, peer_turn={h['peer_turn']}")
        elif "output" in h:
            out = h["output"][:200].replace("\n", " ")
            acc = h.get("accepted", "?")
            ppl_val = h.get("ppl", 0)
            turn = h["turn"]
            att = h.get("attempt", 0)
            print(f"  [turn {turn} att={att}] accepted={acc} ppl={ppl_val:.2f}")
            print(f"    {out}")
    print()
