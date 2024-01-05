*Author: Kincaid MacDonald Riddle Press, as a companion to the **Zetteldev** framework.*
# Zetteldev!

Welcome, fellow traveler, to [Riddle](https://riddle.press)’s *Zetteldev* framework: combining the literate programming of [nbdev](https://nbdev.fast.ai), the publishing capabilities of [quarto](https://quarto.org), and the intellectual exuberance of [zettelkasten](https://obsidian.md) – all within an environment hand-crafted to make experimenting, implementing, and writing up your machine learning research frictionless and joyful.

Here you’ll find a brief overview of how to use this framework. Besides being superb bedtime reading material, this document is also good to share with your collaborators, lest the uninitiated glimpse these folders and suspect you have lost your sanity.
# Huh? What am I supposed to do here?

*insert comic*

This is a highly idiosyncratic tool made to scratch the itch of a perhaps rather idiosyncratic person. Nonetheless, that person is pleased to report that *zetteldev* scratches his itch marvelously well – and perhaps it may also scratch yours.
## Package Management

Zetteldev uses [pixi](https://pixi.sh), a cargo-like package manager for Python, developed by the team behind Mamba. Think of pixi as mamba/conda, but with a poetry-like project dependencies file and accompanying lock file that supports both conda and pip packages. 

To install new conda packages (and add them to the dependencies file),

```
pixi add package-name
```

If you need to install a *pip*-only package, run

```
pixi add --pip package-name
```

