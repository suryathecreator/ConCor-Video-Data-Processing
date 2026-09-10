# Contributing

Issues and focused pull requests are welcome. Please add a small fixture or unit test when changing span extraction, dataset parsing, filtering, checkpoint recovery, or export schemas. These rules are intentionally conservative and a locally useful language heuristic can regress another expression pattern.

Do not commit dataset media, annotations governed by upstream licenses, model weights, tokens, private paths, generated campaigns, or Slurm logs. Run `pytest -q` before opening a pull request.
