# Reports

The JSON a study wrote, committed so that a table in `docs/` can be traced back
to the run that produced it.

Each file carries its own provenance: the commit, whether the working tree was
clean, the library versions, the full configuration and its fingerprint, which
recordings were read, how many frames each contributed and why the rest were
excluded. `docs/RESULTS.md` is generated from these files and from nothing else.

```bash
python3 tools/render_results.py reports --out docs/RESULTS.md
```

No frame data is here. The recordings themselves are photographs of the
operator's room and stay out of the repository; `docs/RESULTS.md` describes them
statistically instead.

## Every file here was measured on the target hardware

All eleven reports were written on the Raspberry Pi 5 (`aarch64`), which is why
their `ms`, `ms_p95` and `ms_p99` columns mean anything. Re-running a study on
a desktop reproduces the *correctness* columns and replaces the timings with
numbers from different hardware, so a report must not be regenerated anywhere
else and committed.

The correctness columns do move slightly between architectures - `aarch64` and
`x86-64` order floating-point reductions differently. Measured across 351
numeric fields of `fair_comparison.json`, the largest difference was 5e-6, well
below any published digit.

## `fair_comparison.json` predates the cache check

The raw-signal half of that report was read out of `measures_cache.npz` without
verifying the processing fingerprint, so nothing guaranteed the raw arrays had
been written under the same configuration as the pipelines they were being
compared against. `tools/fair_comparison.py` now requires a verified cache and
refuses to run against a stale one.

The committed file was checked against a re-run with the verification active:
every published digit is identical, so the numbers were right and what was
missing was the guarantee. It is left as measured on the Pi rather than
replaced by a desktop run. To refresh it on the Pi:

```bash
python3 tools/fair_comparison.py data --json reports/fair_comparison.json
```

Its `cache` field then reads `raw metrics verified; reference verified` rather
than `reference verified` alone - that string is the evidence the check ran.

## The Д07-Д20 studies were measured on a desktop

`study_d*.json` were computed on an x86-64 workstation, not the Pi, because
they replay the corpus hundreds of times and none of their conclusions is about
speed.  Their `ms` columns are therefore **not** comparable with the eleven
reports above, and not with each other across machines.  For anything about
throughput, use the Pi reports.

All fourteen were recomputed from a clean tree so that `source_modified` is
false in every one: an earlier set was written while the study code was still
being edited, which meant no result could be tied to an exact state of it.

```bash
python3 tools/programme.py all --data <recordings> --workers 32
```
