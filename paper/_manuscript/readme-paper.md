# Publishing from Zetteldev with Quarto Manuscripts

All of Zetteldev is constructed from the foundation materials provided by Quarto: it powers our sync from notebooks to markdown (Zettels) and documentation (in NBDev). Here we employ Quarto's fullest capabilities to directly author manuscripts in markdown, while dynamically pulling in results from our experimental notebooks.

It is an obdurate feature of science that the projects are formed, scoped, shackled, and developed by the paper they shall someday become. The standard scientific practice is to experiment for (many) months before suddenly, in a matter of weeks, throwing together a paper. Of course in the course of writing many weaknesses in the experimentation are revealed, along with new directions. A project only fully conceives of itself in writing -- and it is thus widely-acknowledged academic best practice to begin writing early.

But traditionally the writing is done entirely separately from the experimentation. Porting results from your code to your paper is tedious, not to mention that authoring anything in overleaf is cumbersome. 

Zetteldev helps with this on three counts:

1. It allows authoring in Markdown instead of latex. Quarto Markdown, specifically, gives you all of the power over tables, figures, references and cross-references one needs, without the cumbersome syntax. Plus, Quarto can automatically convert this into PDFs/Latex of a specified style.
2. It makes bringing results from your experiments easy, and updating them even easier. Now you needn't wait until the week of the deadline to make your various plots presentable and try to piece together some narrative from them. You can do this, piecemeal, as the results come in.
3. And, perhaps most significantly, Zetteldev encourages the use of the zettelkasten method to develop ideas in piecemeal outside of the paper. By exporting experimental summaries to markdown files that can be linked from within Obsidian, Zetteldev let's you explore a narrative, develop ideas, and do huge chunks of the drafting in a freer format than the paper. 

Our hope is that Zetteldev empowers these three facets of the creative to work together concurrently, leading to better ideation, experimentation, and higher quality papers.

## Embedding Experimental Figures

In the experiment notebook, use the `#|label fig-name` quarto flag.

In the manuscript, use the following syntax to embed the figure:

```
{{< embed ../nbs/experiments/2c3-are-kernels-zeitgeibers.ipynb#fig-spread-of-diffusion-2d >}}
```


