from config import Parser
import skopt 
from neumann_model import create_model, train_model 

try:
    backend = os.environ['DDEBACKEND']
except:
    backend = 'tensorflow.compat.v1'

import numpy as np

import deepxde as dde 
import numpy as np

from deepxde.backend import tensorflow as tf
#dde.config.set_default_float(config.float_precision)


from tensorflow.keras import backend as K
from skopt import gp_minimize
from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args

ITERATION = 0

#dim_learning_rate = Real(low=1e-4, high=5e-2, name = 'learning_rate', prior='log-uniform')
dim_learning_rate = Real(low=1e-5, high=5e-2, name = 'learning_rate', prior='log-uniform')
#dim_num_dense_layers = Integer(low=1, high=10, name='num_dense_layers')
dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')
dim_num_dense_nodes = Integer(low=5, high=500, name='num_dense_nodes')
dim_activation = Categorical(categories=['sin', 'sigmoid', 'tanh'],
                             name='activation')
dim_weights = Real(low=1, high=1e7, name = 'weights', prior='log-uniform')

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_dense_nodes,
              dim_activation,
             dim_weights]

#default_parameters = [1e-4, 3, 150, 'sin']
default_parameters = [1e-3, 3, 275, 'sin', 400]

@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers,
            num_dense_nodes, activation, weights):

    test = Parser()
    config = test.config
    config.learning_rate = learning_rate
    config.num_dense_layers = num_dense_layers
    config.num_dense_nodes = num_dense_nodes
    config.activation = activation
    config.weights = weights
    
    global ITERATION
    

    config.name = config.name + 'gp-' + str(ITERATION)
    print(config.name, 'config.name')
    print(ITERATION, 'it number')

    # Print the hyper-parameters.
    print('learning rate: {0:.1e}'.format(config.learning_rate))
    print('num_dense_layers:', config.num_dense_layers)
    print('num_dense_nodes:', config.num_dense_nodes)
    print('activation:', config.activation)
    print('weights: {0:.1e}'.format(config.weights))
    print()
    
    # Create the neural network with these hyper-parameters.
    model = create_model(config)
    # possibility to change where we save
    accuracy = train_model(model, config)
    #print(accuracy, 'accuracy is')
        
    if np.isnan(accuracy):
        accuracy = 10 ** 5
    
    ITERATION += 1
    return accuracy
    

#fitness(default_parameters)


if __name__ == "__main__":
    n_calls = 100
    
    test = Parser()
    config = test.config
    if config.seed != -1:
        print('set seed')
        dde.config.set_random_seed(config.seed)    

    search_result = gp_minimize(func=fitness,
                            dimensions=dimensions,
                            acq_func='EI', # Expected Improvement.
                                n_calls = n_calls,
                            x0=default_parameters,
                            random_state = config.seed)



    name = 'results/' + config.name + 'gp' 
    search_result.x

    skopt.dump(search_result, name + '.pkl')


