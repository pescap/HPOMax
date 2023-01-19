This folder allows to reproduce the results in:

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

# 2D Dirichlet

To run the 2D Dirichlet experiments:

```
cd Dirichlet;
```
and run
```
DDEBACKEND='tensorflow' GPU_VISIBLE_DEVICES='1' python dirichlet_gp.py --name ze0_11 --hard_constraint --seed 11
```

To train the best model, run:
```
DDEBACKEND='tensorflow' GPU_VISIBLE_DEVICES='2' python main.py --name best43 --hard_constraint --seed 11 --learning_rate 0.0001 --num_dense_layers 2 --num_dense_nodes 275 --activation 'sin'
```


# 3D Neumann

To run the 3D Neumann experiments:


```
cd Neumann;
```
and run
```
DDEBACKEND='tensorflow' GPU_VISIBLE_DEVICES='1' python neumann_gp.py --name finaln0_11 --seed 11 --d 3
```


