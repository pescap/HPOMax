HPO-PINNs
=======

This document allow to reproduce the results in HPO-PINNs.


HPO 2D
------

Run: ::
	DDEBACKEND='tensorflow' GPU_VISIBLE_DEVICES='1' python dirichlet_gp.py --name ze0_11 --hard_constraint --seed 11


To run the best model:

Run: ::
	DDEBACKEND='tensorflow' GPU_VISIBLE_DEVICES='2' python main.py --name best43 --hard_constraint --seed 11 --learning_rate 0.0001 --num_dense_layers 2 --num_dense_nodes 275 --activation 'sin' --return_model

HPO 3D
------

Run
::
	DDEBACKEND='tensorflow' GPU_VISIBLE_DEVICES='1' python neumann_gp.py --name finaln0_11 --seed 11 --d 3

