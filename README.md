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

## How it works? 

First, go to (or copy) the template folder:

```
cd template
```


- In ``config.yaml``, define all the parameters for your model. It will allow you to run different models with one line of code.
- Define your model in ``model.py``. The model is with two functions:
  - create_model: define and compile your model
  - train_model: train the model and return its accuracy as an output
- Set the HPO in ``model_gp``.
