#!/usr/bin/env python3
"""Replay published rollout traces with the unchanged official offline scorer.

Only world dispatch is parallelized. Official score_world, report and self_test
execute unchanged; no policy, simulator, visual encoder or gated weights run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import time

import numpy as np
from scipy.stats import ttest_rel

ROOT = Path(__file__).resolve().parent
SCORER = ROOT / "sources/score_bottles.py"
SCORER_URL = "https://raw.githubusercontent.com/uynitsuj/abc-rabc/9a9fbb5b/score_bottles.py"
SUITES = {
    "n128": (128, "warp", "b5ac676cf5dc23ea7d423d3eee89c9caa82252f9"),
    "n512": (512, "paperwarp512", "d3b63a144070f4c24ffe83b66e5cc1bef8dbefa8"),
}


def load_scorer():
    spec = importlib.util.spec_from_file_location("official_bottle_scorer", SCORER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def score_one(job):
    arm, seed, filename, rules = job
    scorer = load_scorer()
    with np.load(filename, allow_pickle=False) as z:
        if len(z.files) != 1:
            raise ValueError(f"Unexpected arrays: {filename}")
        trace = z[z.files[0]].astype(np.float64)
    if trace.shape != (1800, 65):
        raise ValueError(f"Unexpected trace: {filename} {trace.shape}")
    # Preserve the release's handling of divergent simulations. No filling or
    # filtering is applied to the primary replay; anomalies are audited below.
    with np.errstate(invalid='ignore', over='ignore'):
        return arm, seed, {r: scorer.score_world(trace, *scorer.RULES[r]) for r in rules}


def write_json(path, data):
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temp.replace(path)


def replay(suite, workers):
    started = time.monotonic()
    n, warp_arm, revision = SUITES[suite]
    trace_dir = ROOT / "traces" / suite
    output = ROOT / "trace_replay" / suite
    output.mkdir(parents=True, exist_ok=True)
    scorer = load_scorer()
    rules = list(scorer.RULES) if suite == "n128" else [scorer.PAPER_RULE]
    audit = json.loads((trace_dir / "local_download_audit.json").read_text())
    if audit["revision"] != revision or audit["files"] != 2 * n:
        raise ValueError("Download audit does not match pinned suite")
    if audit.get("manifest_tree_matches") is False:
        raise ValueError("Upstream tree checksum mismatch")
    jobs, files, nonfinite = [], [], []
    arms = ("vanilla", warp_arm)
    arm_seeds = []
    for arm in arms:
        paths = scorer.collect(str(trace_dir), arm, "fullhz_{arm}_sh*")
        actual_files = list(trace_dir.glob(f"fullhz_{arm}_sh*/qpos_trace_*.npz"))
        if len(paths) != n or len(actual_files) != n:
            raise ValueError("Incorrect world count or duplicate seeds")
        arm_seeds.append(set(paths))
        for seed, path in sorted(paths.items()):
            files.append((str(Path(path).relative_to(trace_dir)), sha(path)))
            with np.load(path, allow_pickle=False) as z:
                trace = z[z.files[0]]
                if trace.shape != (1800, 65):
                    raise ValueError('Unexpected trace dimensions')
                bad = np.argwhere(~np.isfinite(trace))
            if len(bad):
                nonfinite.append(dict(arm=arm, seed=seed, file=str(Path(path).relative_to(trace_dir)),
                    nonfinite_values=len(bad), first_frame=int(bad[:,0].min()), last_frame=int(bad[:,0].max()),
                    coordinates=np.unique(bad[:,1]).tolist()))
            jobs.append((arm, seed, path, rules))
    if arm_seeds[0] != arm_seeds[1]:
        raise ValueError("Arms do not share exactly the same world seeds")
    tree = hashlib.sha256("".join(f"{digest}  {path}\n" for path, digest in sorted(files)).encode()).hexdigest()
    if tree != audit["tree_sha256"]:
        raise ValueError("Local traces changed since download audit")
    provenance = dict(suite=suite, revision=revision, tree_sha256=tree,
                      scorer_url=SCORER_URL, scorer_sha256=sha(SCORER), rules=rules)
    cache_file = output / "world_scores.json"
    if cache_file.is_file():
        saved = json.loads(cache_file.read_text())
        if saved["provenance"] != provenance:
            raise ValueError("Cached scores have different provenance; choose a fresh output directory")
        results = saved["worlds"]
    else:
        results = []
    done = {(arm, int(seed)) for arm, seed, _ in results}
    remaining = [job for job in jobs if (job[0], job[1]) not in done]
    if remaining:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(score_one, remaining, chunksize=2):
                results.append(result)
                i = len(results)
                if i % 32 == 0 or i == len(jobs):
                    print(f"{suite}: {i}/{len(jobs)} worlds scored", flush=True)
                    write_json(cache_file, dict(provenance=provenance, worlds=results))
        write_json(cache_file, dict(provenance=provenance, worlds=results))
    if len(results) != len(jobs):
        raise ValueError('Incomplete or duplicated world-score cache')

    cache = {(arm, rule): {} for arm in arms for rule in rules}
    for arm, seed, values in results:
        for rule, value in values.items():
            cache[arm, rule][int(seed)] = value

    # Execute the upstream aggregation and all its locked self-test assertions.
    # Only the I/O loop score_arm is substituted with already computed results.
    def cached_arm(directory, arm, pattern, rule):
        if Path(directory).resolve() != trace_dir.resolve() or pattern != "fullhz_{arm}_sh*":
            raise ValueError("Unexpected scorer request")
        return cache[arm, rule]

    scorer.score_arm = cached_arm
    reports = {r: scorer.report(str(trace_dir), "fullhz_{arm}_sh*", "vanilla", warp_arm, r)
               for r in rules}
    log = io.StringIO()
    with redirect_stdout(log):
        if suite == "n128":
            self_test_exit = scorer.self_test(str(trace_dir), "fullhz_{arm}_sh*")
        else:
            self_test_exit = None
            scorer.print_res(reports[scorer.PAPER_RULE], "published n=512 traces")
    (output / "official_output.txt").write_text(log.getvalue())
    print(log.getvalue(), flush=True)
    counts = [[cache[a, scorer.PAPER_RULE][s]["count"] for s in sorted(arm_seeds[0])] for a in arms]
    exact_t = ttest_rel(counts[1], counts[0])
    bad_seeds = {row['seed'] for row in nonfinite}
    valid_seeds = sorted(arm_seeds[0]-bad_seeds)
    valid_counts = [[cache[a, scorer.PAPER_RULE][s]['count'] for s in valid_seeds] for a in arms]
    finite_t = ttest_rel(valid_counts[1], valid_counts[0])
    quality = dict(total_traces=2*n, nonfinite_traces=nonfinite, affected_paired_scenes=len(bad_seeds),
        affected_scene_fraction=len(bad_seeds)/n, primary_replay='All released traces retained, unchanged official scorer',
        finite_pair_sensitivity=dict(n=len(valid_seeds), vanilla_mean=float(np.mean(valid_counts[0])),
            warp_mean=float(np.mean(valid_counts[1])), diff=float(np.mean(np.array(valid_counts[1])-valid_counts[0])),
            t=float(finite_t.statistic), p=float(finite_t.pvalue)),
        severity='medium' if nonfinite else 'none', confidence='high; inspected pinned NPZ values',
        cause='Consistent with simulated state divergence; the original higher-precision state was not provided, so the exact cause is unverified',
        remediation='Retain original replay for table comparability; separately report finite-pair sensitivity. Reject nonfinite states in local training/export validity masks.')
    published_checks = []
    if suite == 'n512':
        headline = reports[scorer.PAPER_RULE]
        for name, got, expected, tolerance in [
            ('van_count', headline['van_count'], 1989, 0), ('warp_count', headline['warp_count'], 2321, 0),
            ('van_per', headline['van_per'], 3.885, .0005), ('warp_per', headline['warp_per'], 4.533, .0005),
            ('thru_van', headline['thru_van'], 237, .5), ('thru_warp', headline['thru_warp'], 290, .5),
            ('all6_van', headline['geq'][6][0], 9.4, .05), ('all6_warp', headline['geq'][6][1], 25., .05),
        ]:
            published_checks.append(dict(name=name, got=got, expected=expected, tolerance=tolerance,
                                         passed=bool(abs(got-expected)<=tolerance)))
    result = dict(
        provenance=provenance, download_audit=audit, metrics=reports,
        official_self_test_exit=self_test_exit,
        official_self_test_pass_count=log.getvalue().count("[PASS]"),
        paired_count_t_scipy=dict(statistic=float(exact_t.statistic), pvalue=float(exact_t.pvalue), df=n-1),
        data_quality=quality,
        published_headline_checks=published_checks,
        elapsed_seconds=round(time.monotonic() - started, 3), workers=workers,
        protocol={"trace_steps": 1800, "official_scorer_hz": 30, "official_horizon_seconds": 60,
                  "published_rollout_dt_seconds": .034,
                  "note": "Uses release scoring clock, not 1800 * rollout dt. Upstream p uses a normal approximation; exact Student t p is separately recorded."},
        scope="Offline rescoring of author-generated traces; no training, policy rollout, DINO inference or local robot evaluation performed.",
    )
    write_json(output / "report.json", result)
    if self_test_exit not in (None, 0):
        raise SystemExit("Published n128 self-test failed")
    if any(not row['passed'] for row in published_checks):
        raise SystemExit('Published n512 headline check failed')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    for suite in SUITES if args.suite == "all" else [args.suite]:
        replay(suite, args.workers)


if __name__ == "__main__":
    main()
