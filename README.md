# Maximum Independent Set: Self-Training through Dynamic Programming

[![ArXiv](https://img.shields.io/badge/Preprint-ArXiv-blue.svg)](https://arxiv.org/abs/2310.18672)

<!-- [![Blogpost](https://img.shields.io/badge/BlogPost-site-red.svg)](https://grigorisg9gr.github.io/polynomial-nets/) -->

Official implementation of the paper ["**Maximum Independent Set: Self-Training through Dynamic Programming**"](https://arxiv.org/pdf/2310.18672.pdf) (NeurIPS'23).

Both folders contain their respective experiments. Please follow the instructions below on how to run the experiments and reproduce the results. This repository contains implementations in [PyTorch](https://pytorch.org/) and [NetworkX](https://networkx.org/).

## Browsing the experiments

The folder structure is the following:

- **Maximum Independent Set (MIS folder)**

  Run the following command under the folder 'MIS' for training the model for the MIS problem: `python main.py --mode [mode] --dataset_name [dataset_name]`.

  For MIS, the dataset names that are allowed are:
  - TWITTER_SNAP
  - COLLAB
  - RB
  - SPECIAL

  The options are:
  - `--mode`: either 'train' or 'test'
  - `--model_path`: path where the model parameters are stored
  - `--D`: number of neurons of the output layers of the model
  - `--gnn_depth`: number of iterations of GEM
  - `--dense_depth`: number of layers of the fully connected network
  - `--batch_size`: it is the size of the batch
  - `--dim_datasamples`: number of labels each graph can produce
  - `--dim_dataset`: total number of labels employed in each training epoch
  - `--root_graphs_per_iter`: total number of graphs employed, at each iteration, for extracting labels
  - `--idx0 --idx1  --idx0_validation  --idx1_validation`: indices used to load the correct graphs from the dataset in `--dataset_name`
  - `--model_name`: name of the model used for testing
  - `--mixed_rollout`: if set to 'True', the model is trained using mixed rollouts

- **Minimum Vertex Cover (MVC folder)**  

  Run the following command under the folder 'MVC' for training the model for the MVC problem: `python main.py --mode [mode] --dataset_name [dataset_name]`

  For MVC, the dataset names that are allowed are:
  - RB200
  - RB500

  The options are:
  - `--mode`: either 'train' or 'test'
  - `--model_path`: path where the model parameters are stored
  - `--D`: number of neurons of the output layers of the model
  - `--gnn_depth`: number of iterations of GEM
  - `--dense_depth`: number of layers of the fully connected network
  - `--batch_size`: it is the size of the batch
  - `--dim_datasamples`: number of labels each graph can produce
  - `--dim_dataset`: total number of labels employed in each training epoch
  - `--root_graphs_per_iter`: total number of graphs employed, at each iteration, for extracting labels
  - `--idx0 --idx1  --idx0_validation  --idx1_validation`: indices used to load the correct graphs from the dataset in `--dataset_name`
  - `--model_name`: name of the model used for testing
  - `--flag_density`: flag for using a better implementation of the code

## More information on Π-nets

More information about the paper such as a pitch and poster, as presented at NeurIPS'23, can be found on the [conference website](https://neurips.cc/virtual/2023/poster/70728).

Our approach utilizes a self-supervised learning algorithm that iteratively refines a graph-comparing function, which guides the construction of an MIS. By leveraging the recursive nature of dynamic programming and the pattern recognition capabilities of GNNs, the method generates its own training data, progressively enhancing its ability to predict larger independent sets without the need for pre-labeled examples.

## Results

We assess our models, which utilize "normal roll-outs" or "mixed roll-outs" (incorporating a greedy heuristic), against various baselines including traditional algorithms and neural approaches. The results demonstrate that our method outperforms other neural models and exhibits competitive performance against classical methods, particularly excelling in the Twitter and SPECIAL datasets.

Additionally, our approach shows promising generalization capabilities across different graph distributions, as evidenced by out-of-distribution testing. The experiments highlight the effectiveness of our self-supervised learning scheme in learning diverse data distributions and achieving fast and near-optimal solutions for the MIS problem.

## Citing

If you use this code or wish to reference the paper, please cite the following:

**BibTeX**:

```bibtex
@inproceedings{brusca2023maximum,
  title={Maximum Independent Set: Self-Training through Dynamic Programming},
  author={Brusca, Lorenzo and Quaedvlieg, Lars CPM and Skoulakis, Stratis and Chrysos, Grigorios G and Cevher, Volkan},
  booktitle={Advances in neural information processing systems (NeurIPS)},
  year={2023}
}
