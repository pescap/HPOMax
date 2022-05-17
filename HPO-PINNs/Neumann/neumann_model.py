import numpy as np
import skopt
import deepxde as dde
import numpy as np
from functools import reduce

if dde.backend.backend_name == 'pytorch':
    cos = dde.backend.pytorch.cos
else:
    from deepxde.backend import tf
    cos = tf.cos
    
def create_model(config):
    def pde(x, y):
        list_dy = [dde.grad.hessian(y, x, i=dim, j=dim) for dim in range(d)]
        list_f = [cos(k0 * x[:, dim:dim+1]) for dim in range(d)]
        f =  (d-1) * k0 ** 2 * reduce(lambda x1, x2: x1*x2, list_f)        
        dy =  reduce(lambda x1, x2: x1 + x2, list_dy)
        return - dy - k0 ** 2 * y - f

    def func(x):
        list_u = [np.cos(k0 * x[:, dim:dim+1]) for dim in range(d)]
        u =  reduce(lambda x1, x2: x1*x2, list_u)
        return u
    
    def boundary(_, on_boundary):
        return on_boundary
       
        
    #geom = dde.geometry.Rectangle([0,0], [1,1])
    d = config.d
    geom = dde.geometry.Hypercube(d * [0], d * [1])
    k0 = 2 * np.pi * config.n
    wave_len = 1 / config.n

    hx_train = wave_len / config.precision_train
    nx_train = int(1 / hx_train)

    hx_test = wave_len / config.precision_test
    nx_test = int(1 / hx_test)
    
    if config.hard_constraint == True:
        raise ValueError('Only weak constraints')
    else:
        bc = dde.NeumannBC(geom, lambda x: 0, boundary)

    
    data = dde.data.PDE(geom, pde, bc, num_domain= nx_train ** d, num_boundary= 2 ** (d-1) * d * nx_train ** (d-1), solution = func, num_test = nx_test ** d)
    
    print(nx_train ** d, 'Ndof')

    net = dde.maps.FNN([d] + [config.num_dense_nodes] * config.num_dense_layers + [1], config.activation, "Glorot uniform")
    
    model = dde.Model(data, net)
    
    loss_weights = [1, config.weights]
    model.compile("adam", lr=config.learning_rate, metrics=["l2 relative error"], loss_weights = loss_weights)
    return model

def train_model(model, config):
    if config.early == True:
        callbacks = [dde.callbacks.EarlyStopping(min_delta=config.min_delta, patience=config.patience, monitor = config.monitor)]
    else:
        callbacks = []

    import time
    ta = time.time()

    losshistory, train_state = model.train(epochs=config.epochs, callbacks = callbacks)
    texec = time.time()-ta
    
    train = np.array(losshistory.loss_train).sum(axis=1).ravel()
    test = np.array(losshistory.loss_test).sum(axis=1).ravel()
    metric = np.array(losshistory.metrics_test).sum(axis=1).ravel()
    
    accuracy = test.min()
    
    skopt.dump(train, 'results/' + config.name + 'train.pkl')
    skopt.dump(test, 'results/' + config.name + 'test.pkl')
    skopt.dump(metric, 'results/' + config.name + 'metric.pkl')
    skopt.dump(texec, 'results/' + config.name + 'texec.pkl')
    

    if config.lbfgs == True:
        model.compile("L-BFGS")
        dde.optimizers.set_LBFGS_options(maxiter = config.maxiter)
        losshistory, train_state = model.train()
        train_bfgs = np.array(losshistory.loss_train).sum(axis=1).ravel()
        skopt.dump(train_bfgs, 'results/' + config.name + 'train_bfgs.pkl')
        
    return accuracy

