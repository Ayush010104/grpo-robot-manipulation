"""Quantify a specific claim: once a run goes (near-)fully degenerate, does it ever
produce nonzero success again, or is it permanently dead? And does grad_norm blow up
during the dead phase? This is the mechanistic root-cause check for why GRPO-family
methods collapse and whether SM-GRPO / MM-GRPO actually behave differently, not just
"collapse later".
"""
import re
import sys

LINE_RE = re.compile(
    r"\[iter (\d+)\].*?success_rate=([\d.]+).*?degenerate_frac=([\d.]+).*?"
    r"adv_abs_mean=([\d.]+).*?grad_norm=([\d.]+)"
)


def parse_log(path):
    rows = []
    with open(path) as f:
        for line in f:
            m = LINE_RE.search(line)
            if m:
                it, succ, deg, adv, gn = m.groups()
                rows.append((int(it), float(succ), float(deg), float(adv), float(gn)))
    return rows


def analyze(name, path):
    rows = parse_log(path)
    if not rows:
        print(f"{name}: no data")
        return
    n = len(rows)
    first_full_deg = next((it for it, s, d, a, g in rows if d >= 0.999), None)
    # "permanently dead" = after some iteration, success_rate is 0 for every remaining iter
    last_nonzero = max((it for it, s, d, a, g in rows if s > 0), default=-1)
    permanently_dead_from = last_nonzero + 1 if last_nonzero < rows[-1][0] else None
    max_grad_norm = max(g for *_, g in rows)
    max_grad_norm_iter = next(it for it, s, d, a, g in rows if g == max_grad_norm)
    final5_succ = sum(s for _, s, *_ in rows[-5:]) / min(5, n)
    final5_gn = sum(g for *_, g in rows[-5:]) / min(5, n)
    peak_succ = max(s for _, s, *_ in rows)
    peak_succ_iter = next(it for it, s, *_ in rows if s == peak_succ)
    # bounce-back: any iter with deg==1.0 followed later by a nonzero-success iter
    bounces = 0
    seen_full_deg = False
    for it, s, d, a, g in rows:
        if seen_full_deg and s > 0:
            bounces += 1
            seen_full_deg = False
        if d >= 0.999:
            seen_full_deg = True
    print(f"\n=== {name} ({n} iters) ===")
    print(f"  peak success_rate={peak_succ:.3f} @ iter {peak_succ_iter}")
    print(f"  first fully-degenerate (frac>=1.0) iter: {first_full_deg}")
    print(f"  last iter with nonzero success: {last_nonzero}")
    print(f"  permanently dead (0 success to end) from iter: {permanently_dead_from}")
    print(f"  bounce-backs (nonzero success AFTER a fully-degenerate iter): {bounces}")
    print(f"  max grad_norm={max_grad_norm:.2f} @ iter {max_grad_norm_iter}")
    print(f"  final-5-iter mean success={final5_succ:.4f}  mean grad_norm={final5_gn:.2f}")


if __name__ == "__main__":
    runs = [
        ("GRPO (G=8, net256, scale-matched)", "logs/month3_grpo_scalematched.log"),
        ("SM-GRPO (G=8, net256, scale-matched)", "logs/month3_sm_grpo_scalematched.log"),
        ("MM-GRPO (G=8, net256, scale-matched)", "logs/month3_sm_grpo_memory.log"),
        ("GRPO (G=64, net512)", "logs/month3_grpo_g64_net512.log"),
        ("GRPO (G=128, net512, round1)", "logs/month3_grpo_g128_net512.log"),
        ("GRPO (G=128, net512, round2/full-60)", "logs/month3_grpo_g128_net512_round2.log"),
    ]
    for name, path in runs:
        try:
            analyze(name, path)
        except FileNotFoundError:
            print(f"{name}: file not found ({path})")
