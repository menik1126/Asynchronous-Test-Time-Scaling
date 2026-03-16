import json, os, sys

result_dir = sys.argv[1]
repeats = 16
n_questions = 100

group_stats = []
for q in range(n_questions):
    durs = []
    invalids = 0
    max_dur_sample = None
    for r in range(repeats):
        idx = q * repeats + r
        fp = os.path.join(result_dir, f"problem_{idx:04d}.json")
        if not os.path.exists(fp):
            continue
        with open(fp) as fh:
            d = json.load(fh)
        dur = d["duration_seconds"]
        durs.append(dur)
        if d["final_answer"] == "invalid":
            invalids += 1
        if max_dur_sample is None or dur > max_dur_sample[1]:
            max_dur_sample = (idx, dur, d["final_answer"])
    if not durs:
        continue
    group_stats.append({
        "q": q,
        "max_dur": max(durs),
        "median_dur": sorted(durs)[len(durs) // 2],
        "sum_dur": sum(durs),
        "invalids": invalids,
        "bottleneck": max_dur_sample,
        "n": len(durs),
    })

group_stats.sort(key=lambda x: -x["max_dur"])

total_wall = sum(g["max_dur"] for g in group_stats)
total_work = sum(g["sum_dur"] for g in group_stats)

print(f"=== Group sync barrier analysis ({len(group_stats)} groups) ===")
print(f"Wall time (sum of per-group max): {total_wall:.0f}s ({total_wall/60:.1f} min)")
print(f"Total GPU work:                   {total_work:.0f}s ({total_work/60:.1f} min)")
print(f"Idle time (sync waiting):         {total_wall * repeats - total_work:.0f}s ({(total_wall * repeats - total_work)/60:.1f} min)")
print(f"GPU utilization:                  {total_work / (total_wall * repeats) * 100:.1f}%")
print()

groups_with_inv = [g for g in group_stats if g["invalids"] > 0]
groups_no_inv = [g for g in group_stats if g["invalids"] == 0]

avg_inv = sum(g["max_dur"] for g in groups_with_inv) / max(len(groups_with_inv), 1)
avg_no = sum(g["max_dur"] for g in groups_no_inv) / max(len(groups_no_inv), 1)

print(f"Groups WITH invalid samples: {len(groups_with_inv)}/100")
print(f"  Avg group wall time: {avg_inv:.1f}s")
print(f"  Total wall time:     {sum(g['max_dur'] for g in groups_with_inv):.0f}s "
      f"({sum(g['max_dur'] for g in groups_with_inv)/total_wall*100:.1f}% of total)")
print()
print(f"Groups WITHOUT invalid: {len(groups_no_inv)}/100")
print(f"  Avg group wall time: {avg_no:.1f}s")
print(f"  Total wall time:     {sum(g['max_dur'] for g in groups_no_inv):.0f}s "
      f"({sum(g['max_dur'] for g in groups_no_inv)/total_wall*100:.1f}% of total)")
print()

print(f"=== Top 10 slowest groups ===")
for g in group_stats[:10]:
    bn = g["bottleneck"]
    ratio = g["max_dur"] / max(g["median_dur"], 0.1)
    print(f"  Q{g['q']:3d}: wall={g['max_dur']:6.1f}s, median={g['median_dur']:5.1f}s, "
          f"ratio={ratio:.1f}x, invalids={g['invalids']}, "
          f"bottleneck=idx{bn[0]}({bn[1]:.1f}s, {bn[2]!r})")

print()
print(f"=== Summary ===")
print(f"  {len(groups_with_inv)} groups have invalid samples")
print(f"  They consume {sum(g['max_dur'] for g in groups_with_inv)/total_wall*100:.1f}% of wall time")
print(f"  Bottleneck ratio (max/median): avg {sum(g['max_dur']/max(g['median_dur'],0.1) for g in group_stats)/len(group_stats):.1f}x")
print(f"  → Each group waits for its slowest sample (usually an invalid one grinding through 15 turns)")
