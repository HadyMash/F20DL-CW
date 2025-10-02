# F20DL-CW

Data Mining and Machine Learning coursework repository

# How to get started?

Install the required dependencies in a virtual env

```
pip install -r requirements.txt
```

## How to use the notebooks

Because it can be hard to resolve conflicts in jupyter notebooks, we version control the `py:percent` format .py version and use jupytext to sync a local notebook and the .py file. `jupytext` is already included in the requirements, so if you use the jupyter notebook server (by running `jupyter notebook`) it will be automatically synced. You can open the `.py` file and it will open as a notebook.

If you use VS Code, you can use [Jupytext Sync](https://marketplace.visualstudio.com/items?itemName=caenrigen.jupytext-sync) to sync the notebook and python files. If you're using neovim, you can use [Molten](https://github.com/benlubas/molten-nvim) with [Jupytext.nvim](https://github.com/GCBallesteros/jupytext.nvim) (check [Molten](https://github.com/benlubas/molten-nvim) for configuration instructions). If you're using another editor, you may find a similar Jupytext extension

# How to make changes

Make a feature branch, make the changes, then create a PR into main
