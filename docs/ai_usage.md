# AI usage log

We disclose every use of AI in this project, as the brief requires. Add one row each time an AI tool helps with code, text or planning.

| date | member | tool | what AI did | what we checked |
|---|---|---|---|---|
| 2026-11-02 | Nitin | Claude Code | Generated repo skeleton and check_setup.py from a prompt | Ran check_setup.py on my laptop and read the code |
| 2026-11-02 | Nitin | Claude Code | Wrote the team plan (docs/plan.md), the five member logs and the relay rules in the README, with research sub-agents | Claude ran link and character checks. Nitin's own review: TODO |
| 2026-11-02 | Nitin | Claude Code | Wrote src/utils.py (set_seed, get_device, paths) and drafted the line-by-line notes in docs/log/nitin.md | Claude ran the self-check (python -m src.utils) and it passed. Nitin reads every line before merging: TODO |
| 2026-11-03 | Srivalli | Claude Code (Claude Opus 5.5) | [written by the AI] Installed the venv, downloaded the data, wrote src/dataset.py, generated data/splits.csv, built and ran notebooks/01_eda.ipynb, drafted the report sections Dataset and Limitations, ethics and future work, and wrote these log notes. Seed 3 runs NOT RUN (train() and models.py do not exist yet) | i ran the notebook (reset the kernels and ran it) there were no leakages and content check was ok. the dataset was clearly analysed from its's classes (with smaple images of each class), data augmentation is also done, and many parameters were analysed too include color and other things. |
| 2026-11-03 | Srivalli | Copilot (model not known) | [written by Claude Code] The six Part A viva questions and answers came from this tool (Srivalli confirmed). Claude Code only fixed their grammar and format | NOT CHECKED YET |
