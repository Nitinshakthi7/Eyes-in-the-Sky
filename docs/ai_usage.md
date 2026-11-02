# AI usage log

We disclose every use of AI in this project, as the brief requires. Add one row each time an AI tool helps with code, text or planning.

| date | member | tool | what AI did | what we checked |
|---|---|---|---|---|
| 2026-11-02 | Nitin | Claude Code | Generated repo skeleton and check_setup.py from a prompt | Ran check_setup.py on my laptop and read the code |
| 2026-11-02 | Nitin | Claude Code | Wrote the team plan (docs/plan.md), the five member logs and the relay rules in the README, with research sub-agents | Claude ran link and character checks. Nitin's own review: TODO |
| 2026-11-02 | Nitin | Claude Code | Wrote src/utils.py (set_seed, get_device, paths) and drafted the line-by-line notes in docs/log/nitin.md | Claude ran the self-check (python -m src.utils) and it passed. Nitin reads every line before merging: TODO |
| 2026-11-02 | Rachana | ChatGPT | Provided commands for laptop/environment setup, installation, configuration, and verification | I personally copied and pasted the commands into the terminal, executed them myself, and shared the outputs with ChatGPT for validation and troubleshooting |
| 2026-11-02 | Rachana | Antigravity (Gemini 3.8 Flash) | Assisted with Git global identity configuration, created `setup/rachana` branch, executed `python check_setup.py --name Rachana`, staged `docs/setup/rachana.md`, and inspected work-log and AI disclosure requirements | I provided my GitHub username and email, directed re-running check_setup.py without --download, and reviewed the branch, Git identity results, setup report, git status, and diffs |
