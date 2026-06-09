"""Rebalance the action-magnitude distribution of annotations_vln_llamafactory.jsonl.

Atomic actions are taken DIRECTLY from r2r.jsonl (the GT, action codes
0=stop,1=forward,2=left,3=right; each = 1 atomic = 25cm / 15deg). For each vln
record we read its current frame index k (from the last image) and the horizon n
(how many atomic actions the original answer covered), then take r2r actions[k:k+n]
and RE-CHUNK them with a balancing greedy:
  forward -> 25/50/75 cm (1/2/3 atoms)
  turn    -> 15/30 deg   (1/2 atoms; a 45deg run becomes 30+15)
Greedy: among feasible chunk sizes (<= remaining run), emit the one whose running
count is currently lowest -> pushes toward equal proportions.
Trajectory (total movement) is UNCHANGED; only the step-size labels change.

Out: annotations_vln_balanced_llamafactory.jsonl
"""
import json, re

DATA = ""
R2R = f"{DATA}/r2r.jsonl"
SRC = f"{DATA}/annotations_vln_llamafactory.jsonl"
OUT = f"{DATA}/annotations_vln_balanced_llamafactory.jsonl"

CODE = {0: "stop", 1: "f", 2: "l", 3: "r"}     # r2r action codes
FRAME = re.compile(r"frame_(\d+)\.jpg")


def load_r2r():
    acts = {}
    for l in open(R2R):
        r = json.loads(l)
        acts[str(r["episode_id"])] = r["actions"]
    return acts


def answer_horizon(ans):
    """how many ATOMIC actions the original answer covers (to slice r2r)."""
    n = 0
    for seg in ans.split(","):
        s = seg.lower().strip()
        if not s:
            continue
        if "stop" in s:
            n += 1; continue
        m = re.search(r"(\d+)", s); v = int(m.group(1)) if m else 0
        n += v // 25 if "forward" in s else v // 15
    return n


# running counts -> balance toward uniform
cnt_f = {1: 0, 2: 0, 3: 0}      # 25/50/75
cnt_t = {1: 0, 2: 0}            # 15/30


def chunk(run_len, cnt, maxsize):
    out = []
    while run_len > 0:
        feas = [s for s in range(1, maxsize + 1) if s <= run_len]
        s = min(feas, key=lambda z: cnt[z])   # least-used size first -> even
        cnt[s] += 1; out.append(s); run_len -= s
    return out


def reencode(atoms):
    """atoms: list of 'f'/'l'/'r'/'stop' (from r2r) -> rebalanced answer string."""
    parts, i = [], 0
    while i < len(atoms):
        d = atoms[i]
        if d == "stop":
            parts.append("stop"); i += 1; continue
        j = i
        while j < len(atoms) and atoms[j] == d:
            j += 1
        run = j - i
        if d == "f":
            for s in chunk(run, cnt_f, 3):
                parts.append(f"forward {25*s} cm")
        else:
            side = "left" if d == "l" else "right"
            for s in chunk(run, cnt_t, 2):
                parts.append(f"turn {side} {15*s} degree")
        i = j
    return ", ".join(parts)


def main():
    acts = load_r2r()
    n = miss = 0
    with open(OUT, "w") as w:
        for l in open(SRC):
            l = l.strip()
            if not l:
                continue
            r = json.loads(l)
            ep = str(r["episode_id"])
            k = int(FRAME.search(r["images"][-1]).group(1))
            horizon = answer_horizon(r["messages"][1]["value"])
            ep_acts = acts.get(ep, [])
            atoms = [CODE[a] for a in ep_acts[k:k + horizon]]   # GT atomic actions from r2r
            if len(atoms) != horizon:
                miss += 1                                       # near episode end / oob
            r["messages"][1]["value"] = reencode(atoms)
            w.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    f = sum(cnt_f.values()); t = sum(cnt_t.values())
    print(f"wrote {n} records (r2r-sourced; {miss} short-slices at episode ends) -> {OUT}")
    print(f"forward 25/50/75 cm -> { {k: round(100*cnt_f[k]/f,1) for k in cnt_f} }")
    print(f"turn 15/30 deg      -> { {k: round(100*cnt_t[k]/t,1) for k in cnt_t} }")


if __name__ == "__main__":
    main()
