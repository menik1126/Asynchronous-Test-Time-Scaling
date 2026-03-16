import json, os, sys

result_dir = sys.argv[1]

fork_heavy = []
total = 0
total_forks = 0
total_ans_found = 0
invalids = 0
turn0_forks = 0

for f in sorted(os.listdir(result_dir)):
    if not f.endswith('.json'):
        continue
    total += 1
    with open(os.path.join(result_dir, f)) as fh:
        d = json.load(fh)
    forks = [h for h in d['full_history'] if h.get('event') == 'fork']
    ans_found = [h for h in d['full_history'] if h.get('accepted_reason') == 'answer_found']
    total_forks += len(forks)
    total_ans_found += len(ans_found)
    if d['final_answer'] == 'invalid':
        invalids += 1
    for fk in forks:
        if fk['turn'] == 0:
            turn0_forks += 1
    if len(forks) >= 2:
        idx = int(f.replace('problem_', '').replace('.json', ''))
        fork_heavy.append((f, len(forks), d['final_answer'], forks))

fork_heavy.sort(key=lambda x: -x[1])

print(f"=== Fork analysis ({total} samples) ===")
print(f"  Total forks: {total_forks} ({total_forks/total:.2f}/sample)")
print(f"  Turn-0 forks (herding): {turn0_forks}/{total_forks}")
print(f"  Answer-found shortcuts: {total_ans_found}")
print(f"  Invalid answers: {invalids}")
print(f"  Samples with 2+ forks: {len(fork_heavy)}")

# Analyze a fork-heavy sample's CoT
print(f"\n=== Top fork-heavy samples ===")
for f, nf, ans, forks in fork_heavy[:5]:
    print(f"\n  {f}: forks={nf}, answer={ans!r}")
    for fk in forks:
        print(f"    turn {fk['turn']}: from idx={fk['forked_from']}, "
              f"my_ppl={fk['my_avg_ppl']:.3f}, peer_ppl={fk['peer_avg_ppl']:.3f}, "
              f"ppl_gap={fk['ppl_gap']:.3f}, score_gap={fk['score_gap']:.3f}")

# Pick one fork-heavy sample and show the actual CoT content
if fork_heavy:
    pick = fork_heavy[0]
    fp = os.path.join(result_dir, pick[0])
    with open(fp) as fh:
        d = json.load(fh)
    print(f"\n=== CoT for {pick[0]} (most forks) ===")
    for h in d['full_history']:
        if h.get('event') == 'fork':
            print(f"\n--- [FORK at turn {h['turn']}] from idx={h['forked_from']} "
                  f"(my_ppl={h['my_avg_ppl']:.3f} → peer_ppl={h['peer_avg_ppl']:.3f}) ---")
        elif 'output' in h:
            accepted = h.get('accepted', False)
            reason = h.get('accepted_reason', '')
            status = 'ACCEPTED' if accepted else 'REJECTED'
            if reason:
                status += f' ({reason})'
            out_preview = h['output'][:150].replace('\n', ' ')
            print(f"\n  [Turn {h['turn']} Att {h['attempt']}] {status} ppl={h.get('ppl',0):.3f}")
            print(f"    {out_preview}...")
