python - <<'EOF'
import io
p = "README.md"
s = io.open(p, encoding="utf-8").read()
s = s.replace(
"""- Resume support: a `_organize_index.json` file remembers already-processed
files, and an `_analysis_checkpoint.json` file caches AI results so an
interrupted run only infers the remaining images.""",
"""- Resume support: a `_organize_index.json` file remembers already-processed
files, an `_analysis_checkpoint.json` file caches per-image AI results, and
an `_event_names.json` file caches event names by cluster membership, so an
interrupted run only infers what is missing.""")
s = s.replace(
"""- **Resume:** delete `_organize_index.json` in the output folder to force
re-processing of everything. Checkpointed AI results are keyed by file
size and modification time, so edited files are re-analyzed.""",
"""- **Resume:** delete `_organize_index.json` in the output folder to force
re-processing of everything. Checkpointed AI results are keyed by file
size and modification time, so edited files are re-analyzed. Cached event
names are keyed by cluster membership (path, size, and mtime of every
member), so adding, removing, or editing an image re-names its event.""")
io.open(p, "w", encoding="utf-8", newline="").write(s)
      EOF
      python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in
      ['config.py','state.py','clustering.py','organizer.py']]; print('syntax ok')")