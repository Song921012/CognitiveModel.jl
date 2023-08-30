# CognitiveModel

# Set Up Environment

- Step 1: install `Julia`(necessary) and `VScode`(Optional but recommended), and configure your environment

- Step 2: git clone this repo and `cd` to this repo

- Step 3: Start `Julia Repl` and type `] instantiate`, and then it will download packages and set up the environment

- Step 4: (Optional) Julia's compilation time is very long, if you want to shorten project start time, you can run `precompile.jl`

# Code files description

- `archive` folder: archive files, no need to touch it
- `output` folder: results output folder
- `dataprocess.ipynb`: data preprocessing
- `dataexploration.ipynb`: data exploration
- `*neuralvac.jl`: train M C impact socre of vaccine
- `*neuralodeinter.jl`: train M C impact socre of intervention
- `*neuralodeinter.jl`: train $\beta(t)$ $\nu(t)$
