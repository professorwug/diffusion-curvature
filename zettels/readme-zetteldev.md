*Author: Kincaid MacDonald @ WherewithAI, as a companion to the **Zetteldev** framework.*
# Zetteldev!

Welcome, fellow traveler, to [Wherewithal](http://wherewith.ai)’s *Zetteldev* framework: combining the literate programming of [nbdev](https://nbdev.fast.ai), the publishing capabilities of [quarto](https://quarto.org), and the intellectual exuberance of [zettelkasten](https://obsidian.md) – all within an environment hand-crafted to make experimenting, implementing, and writing up your machine learning research frictionless and joyful.

Here you’ll find a brief overview of how to use this framework. Besides being superb bedtime reading material, this document is also good to share with your collaborators, lest the uninitiated glimpse these folders and suspect you have lost your sanity.
# Huh? What am I supposed to do here?

This is a highly idiosyncratic tool made to scratch the itch of a perhaps rather idiosyncratic person. Nonetheless, that person is pleased to report that *zetteldev* scratches his itch marvelously well – and perhaps it may also scratch yours.

Below, you'll find more detailed instructions (and some philosophical wax). Here's the quickstart version:

1. Install pixi from [https://pixi.sh](https://pixi.sh)
2. `pixi install` to install the environment
3. `pixi run postinstall` to set up the project: installing it as a python package, creating a jupyter kernel, and setting up git hooks to keep notebooks clean & conflict free
4. `pixi run nbsync` to sync cells with `#|export` from a notebook to the python file specified in `#|default_exp` , and pixi run pysync to sync edits in a python file back to a notebook. Since it’s a bidirectional sync, you can edit wherever you prefer.
5. If you use VSCode, consider installing the fastai/nbdev-vscode extension!

## Declarative Package Management

Zetteldev uses [pixi](https://pixi.sh), a cargo-like package manager for Python, developed by the team behind Mamba. Think of pixi as mamba/conda, but with a poetry-like project dependencies file and accompanying lock file that supports both conda and pip packages. 

### Environment Setup

To set up an environment with pixi, you need only run two commands within the project folder

```sh
pixi install -e cuda # Omit the -e cuda if you don't have a gpu.
pixi run postinstall
```
The first command downloads all the packages specified in `pixi.toml` and creates a conda like environment stored in the .pixi folder. The second installs the python repo as a locally editable, importable package -- and creates a jupyter kernel so you can run notebooks in your new pixi environment.

### Using your new Pixi Env

Pixi activates its environments more like `poetry` than conda. To run an arbitrary command within the local environment, use this syntax

```sh
pixi run python main.py
```

To get a shell with the environment activated, conda style, run 

```sh
pixi shell
```

To use the pixi env from within a notebook, either run

```sh
pixi notebook # -e cuda , if you have one
```
to launch a jupyter server, or (e.g. within VSCode) connect to the jupyter kernel called `your-project-name-pixi` (this was installed during the 'postinstall' script).


### Installing Packages

To install new conda packages (and add them to the dependencies file),

```sh
pixi add package-name
```

If you need to install a *pip*-only package, run

```sh
pixi add --pip package-name
```

More options can be found [here](https://pixi.sh/latest/cli/#install).


## Literate Programming with NBDev

### Syncing From Notebooks to Python Files

Zetteldev embraces [nbdev](https://nbdev.fast.ai)'s ethos of notebook-driven development -- but extends it in ways we find more congenial to good research practices.

The core idea is a two-way sync between notebooks and python files. This allows you to define, document, and test core pieces of your codebase -- and then import them into other notebooks and scripts.

At the top of every notebook (in the Library folder; more on that later) you'll notice this snippet:

```
#|default_exp module_name
```

And in strategic, reuse-worthy cells in the notebook, you'll find
```
#|export
```

The first tells Zetteldev to sync that notebook with `module_name.py`.  The second indicates which cells should be written to the python file.

Because it's a *two-way* sync, once this structure has been defined, you can edit the code in whichever format you prefer. 

To sync code from notebooks to python files, run 
```
pixi run nbsync
```

To sync from the python to the notebooks, run 

```
pixi run pysync
```

### Cleaning Notebooks to prevent Git Conflicts.

NBDev also solves the bane of notebook authors: merge conflicts. It does this by cleansing notebooks of unnecessary metadata, so that merely opening someone else's notebook doesn't change the file.

The easiest way to use this is with NBDev's [VSCode Extension](https://github.com/fastai/nbdev-vscode). We recommend you install this, if you use VSCode. If you use jupyter lab, fear not; nbdev will automatically clean notebooks on save via already-installed hooks.

To manually clean all the notebooks in the project, run 
```
pixi run nbclean
```

*There's much more to say, and many more tricks to cover! We refer you to [nbdev's End-To-End Walkthrough](https://nbdev.fast.ai/tutorials/tutorial.html) for the full details.