HPO-PINNs
=======

This document allow to reproduce the results in HPO-PINNs. 

Dirichlet 2D
------
Go to the ``Dirichlet`` repository. Then, simply run:
::
	DDEBACKEND='tensorflow' GPU_VISIBLE_DEVICES='1' python dirichlet_gp.py --name ze0_11 --hard_constraint --seed 11


To run the best model:
::
	DDEBACKEND='tensorflow' GPU_VISIBLE_DEVICES='2' python main.py --name best43 --hard_constraint --seed 11 --learning_rate 0.0001 --num_dense_layers 2 --num_dense_nodes 275 --activation 'sin' --return_model

The results will be exported to the ``results`` folder. Then, you can analyze this output throughout the ``HPO-2D.ipynb`` notebook, which allows to reproduce the figures in the paper.

Neumann 3D
------
Go to the ``Neumann`` repository and run:
::
	DDEBACKEND='tensorflow' GPU_VISIBLE_DEVICES='1' python neumann_gp.py --name finaln0_11 --seed 11 --d 3

The results will be exported to the ``results`` folder. Then, you can analyze this output throughout the ``HPO-3D.ipynb`` notebook, which allows to reproduce the figures in the paper.