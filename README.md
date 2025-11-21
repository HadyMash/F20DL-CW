# F20DL-CW

Data Mining and Machine Learning coursework repository

# How to get started?

Install the required dependencies in a virtual env

```
pip install -r requirements.txt
```

This will install of the dependencies needed.

## How to use the notebooks

Because it can be hard to resolve conflicts in jupyter notebooks, we version control the `py:percent` format .py version and use [`jupytext`](https://jupytext.readthedocs.io/en/latest/) to sync a local notebook and the .py file. `jupytext` is already included in the requirements, so if you use the jupyter notebook server (by running `jupyter notebook`) it will be automatically synced. You can open the `.py` file and it will open as a notebook.

You can also use the cli to sync notebooks by running `jupytext --sync <file>`.

If you use VS Code, you can use [Jupytext Sync](https://marketplace.visualstudio.com/items?itemName=caenrigen.jupytext-sync) to sync the notebook and python files (make sure you configure the extension's settings to sync properly). If you're using neovim, you can use [Molten](https://github.com/benlubas/molten-nvim) with [Jupytext.nvim](https://github.com/GCBallesteros/jupytext.nvim) (check [Molten](https://github.com/benlubas/molten-nvim) for configuration instructions). If you're using another editor, you may find a similar Jupytext extension.

You can also run the notebook py files as normal python scripts if you don't want to use notebooks or sync them. Just keep in mind that some cells which show plots may not work as expected.

### Git hooks

If you're contributing to this repository, it's recommended to use git hooks to keep the notebooks and .py files synced automatically.

You should also configure the git hooks by running `setup-hooks.sh` (or `setup-hooks.ps1` on windows). This file will configure the hooks so whenever you commit, pull, or merge, the notebooks will be synced automatically and files will be formatted. This way your notebooks always stay synced. On Linux/MacOS, the setup script creates symbolic links, keeping them updated. On Windows, however, it copies the hooks, so if the hooks are updated, you may need to re-run the setup script.

# How to make changes

Make a feature branch, make the changes, then create a PR into main

# Overview of files

- eda.py/ipynb: Exploratory Data Analysis of the dataset. This script contains the EDA and also some preprocessing steps and it saves the output, you must run this first if you wish to use any of the other notebooks.
- stat-models.py/ipynb: Implements the baseline statistical models.
- nn-\*.py/ipynb: Implements the different neural network models. The latest one is nn-state-batchnorm. state refers to using one-hot encoded states as part of the input features. the nn-mse uses mse instead of huber, but it performs worse.
- we.py is a python script that implements the novel qr code approach described in the report.
