# HPOMax 

Easy-to-configure hyper-parameters (HPs) optimization for Deep Learning simulations on top of [Scikit-Optimize](https://scikit-optimize.github.io/stable/index.html). 

This library allows to parse easily the parameters for your model. Then, you can configure the HPs that will be optimized, and run the `skopt.gp_minimize` function. The `template` folder allows to define new HPO processes.

This library was used for HPO applied to physics-informed neural networks in:

```
@misc{https://doi.org/10.48550/arxiv.2205.06704,
  doi = {10.48550/ARXIV.2205.06704},
  url = {https://arxiv.org/abs/2205.06704},
  author = {Escapil-Inchauspé, Paul and Ruz, Gonzalo A.},
  title = {Hyper-parameter tuning of physics-informed neural networks: Application to Helmholtz problems},
  publisher = {arXiv},
  year = {2022},
}
```
