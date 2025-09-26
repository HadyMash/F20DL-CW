# F20DL-CW

Data Mining and Machine Learning coursework repository

# How to get started?

Install the required dependencies in a virtual env

```
pip install -r requirements.txt
```

## How to use the notebooks

Because it can be hard to resolve conflicts in jupyter notebooks, we version control the `py:percent` format .py version and use jupytext to sync a local notebook and the .py file. `jupytext` is already included in the requirements, so if you use the jupyter notebook server (by running `jupyter notebook`) it will be automatically synced.

If you use VS Code, you can use [Jupytext Sync](https://marketplace.visualstudio.com/items?itemName=caenrigen.jupytext-sync) to sync the notebook and python files.

# How to make changes

Make a feature branch, make the changes, then create a PR into main
