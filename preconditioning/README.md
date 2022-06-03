The files in this repository are a collection of colabs used to create results from the convex pre-conditioning experiments.

Due to colab magic (i.e. colabtools to upload files), these notebooks will not work out of the box in Jupyter.

Structure:
- Notebook files contain the two notebooks required to run the experiments. The first is 'ConvexNN vs RegularNN - Amil.ipynb' which is used to run the dense traning after preconditioning. This was set up on colab to be able to freely access a GPU. The other file is entitled 'convex_nn.demo.'

Installation:
- Though out of scope for the final project, the convex preconditioning experiments make use of Aaron Mishkins implementations of convex neural networks. The library should be installed via (https://github.com/aaronpmishkin/convex_nn). We will work to make this more convenient if the results are fully open-sourced.
