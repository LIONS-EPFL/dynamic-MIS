# Maximum Independent Set: Self-Training through Dynamic Programming

[![ArXiv](https://img.shields.io/badge/Preprint-ArXiv-blue.svg)](https://arxiv.org/abs/2310.18672)

<!-- [![Blogpost](https://img.shields.io/badge/BlogPost-site-red.svg)](https://grigorisg9gr.github.io/polynomial-nets/) -->

Website implementation of the paper ["**Maximum Independent Set: Self-Training through Dynamic Programming**"](https://arxiv.org/pdf/2310.18672.pdf) (NeurIPS'23). If you would like to view the code implementation results, please navigate to the "main" branch.

## More information on the paper

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
