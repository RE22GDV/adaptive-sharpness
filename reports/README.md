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
