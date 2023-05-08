# CognitiveModel

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://Song921012.github.io/CognitiveModel.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://Song921012.github.io/CognitiveModel.jl/dev/)
[![Build Status](https://github.com/Song921012/CognitiveModel.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/Song921012/CognitiveModel.jl/actions/workflows/CI.yml?query=branch%3Amain)

# Set Up Environment

- Step 1: install `Julia`(necessary) and `VScode`(Optional but recommended), and configure your environment

- Step 2: git clone this repo and `cd` to this repo

- Step 3: Start `Julia Repl` and type `] instantiate`, and then it will download packages and set up the environment

# Code files description

- `src` folder: don't touch it
- `test` folder: only for test
- `docs` folder: for generating documents (will be done at the end of the projects if we want to make this project a package)
- `data` folder: data source files
- `output` folder: results output folder
- `dataprocess.ipynb`: data preprocessing
- `dataexploration.ipynb`: data exploration to find the time span
- `*epineural.jl`: train $\beta(t)$ $\nu(t)$
- `*neuralvac.jl`: train M C impact socre of vaccine
- `*neuralinter.jl`: train M C impact socre of intervention
