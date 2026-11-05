# Backend: the CNN project

This folder holds the code, notebooks, results, report drafts, documents and the offline website server of **Eyes in the Sky**, our land-use classifier for EuroSAT satellite tiles.

**The full README (what the project is, how to set up your laptop, how to run everything, the results, who did what) is at the repository root: [../README.md](../README.md).**

Quick commands, run inside this folder with the virtual environment on:

| What | Command |
|---|---|
| Set up and download data (first time, see README steps 3 to 6) | `python check_setup.py --name YourName --download` |
| Check the code runs | `python -m src.utils` then `python -m src.train` |
| Train one model | `python -c "from src.train import train; train('mobilenet_v2_full', 1.0, 0)"` |
| Open the notebooks | `jupyter notebook` |
| Start the website | `python -m app.server` then open <http://127.0.0.1:8000/> |
| Check the website | `python -m app.test_server` |

Where things are: `src/` code, `notebooks/` 01 to 05, `app/` the website server (the pages are in `../frontend/`), `data/` the fixed split (images are downloaded here), `models/` weights (not in git, see [models/README.md](models/README.md)), `results/` `metrics.csv` and figures, `report/` report sections, `docs/` the team plan, one log per member, viva questions and the AI usage table.
