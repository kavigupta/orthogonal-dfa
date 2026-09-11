"""Classify each logged ce_pass split as spurious or legitimate.

A split separates two prefixes (witness, sprime) that reach one leaf but were
judged to behave differently.  In the TRUE 8-state DFA for "reject iff frames 0
AND 1 both closed", a prefix's Myhill-Nerode state is its signature
(f0_closed, f1_closed, phase), with (1,1,*) collapsing to the reject sink.
If witness and sprime share that signature they are MN-equivalent, so splitting
them is SPURIOUS (over-splitting on oracle noise); different signatures => a
legitimate refinement."""
import re, sys, ast

def sig(pref):
    wc = 0
    f0 = f1 = False
    for c in pref:
        if c >= 3:            # wildcard advances phase
            wc += 1
        else:                 # kmer closes the frame at the current phase
            ph = wc % 3
            if ph == 0: f0 = True
            elif ph == 1: f1 = True
    if f0 and f1:
        return "REJECT_SINK"
    return (f0, f1, wc % 3)

for path, label in [(sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])]:
    spur = leg = 0
    rows = []
    for ln in open(path):
        m = re.search(r"n=(\d+) s1=(\d+) witness=(\[[^\]]*\]) sprime=(\[[^\]]*\])", ln)
        if not m:
            continue
        n, s1 = int(m.group(1)), int(m.group(2))
        w = ast.literal_eval(m.group(3)); sp = ast.literal_eval(m.group(4))
        sw, ss = sig(w), sig(sp)
        kind = "SPURIOUS" if sw == ss else "legit"
        if sw == ss: spur += 1
        else: leg += 1
        rows.append((n, s1, kind, sw, ss))
    tot = spur + leg
    print(f"\n### {label}: {tot} splits | legit={leg} SPURIOUS={spur} "
          f"({(spur/tot*100 if tot else 0):.0f}% spurious)")
    for n, s1, kind, sw, ss in rows:
        print(f"   at n={n:2d} split state {s1:2d}: {kind:8s}  witness_sig={sw}  sprime_sig={ss}")
