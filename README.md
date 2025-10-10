# F20DL-CW

Data Mining and Machine Learning coursework repository

# How to get started?

Install the required dependencies in a virtual env

```
pip install -r requirements.txt
```

This will install of the dependencies needed.

## How to use the notebooks

Because it can be hard to resolve conflicts in jupyter notebooks, we version control the `py:percent` format .py version and use jupytext to sync a local notebook and the .py file. `jupytext` is already included in the requirements, so if you use the jupyter notebook server (by running `jupyter notebook`) it will be automatically synced. You can open the `.py` file and it will open as a notebook.

You can also use the cli to sync notebooks by running `jupytext --sync <file-name>`.

If you use VS Code, you can use [Jupytext Sync](https://marketplace.visualstudio.com/items?itemName=caenrigen.jupytext-sync) to sync the notebook and python files. If you're using neovim, you can use [Molten](https://github.com/benlubas/molten-nvim) with [Jupytext.nvim](https://github.com/GCBallesteros/jupytext.nvim) (check [Molten](https://github.com/benlubas/molten-nvim) for configuration instructions). If you're using another editor, you may find a similar Jupytext extension

### Git hooks

You should also configure the git hooks by running `setup-hooks.sh`. This file will configure the hooks so whenever you commit, pull, or merge, the notebooks will be synced automatically and files will be formatted. This way your notebooks always stay synced.

# How to make changes

Make a feature branch, make the changes, then create a PR into main
