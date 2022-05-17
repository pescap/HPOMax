import numpy as np
import skopt
import deepxde as dde

if dde.backend.backend_name == "pytorch":
    sin = dde.backend.pytorch.sin
else:
    from deepxde.backend import tf

    sin = tf.sin


def create_model(config):
    def pde(x, y):
        dy_xx = dde.grad.hessian(y, x, i=0, j=0)
        dy_yy = dde.grad.hessian(y, x, i=1, j=1)

        d = config.d

        f = (d - 1) * k0**2 * sin(k0 * x[:, 0:1]) * sin(k0 * x[:, 1:2])
        return -dy_xx - dy_yy - k0**2 * y - f

    def func(x):
        return np.sin(k0 * x[:, 0:1]) * np.sin(k0 * x[:, 1:2])

    def transform(x, y):
        res = x[:, 0:1] * (1 - x[:, 0:1]) * x[:, 1:2] * (1 - x[:, 1:2])
        return res * y

    def boundary(_, on_boundary):
        return on_boundary

    d = config.d
    # geom = dde.geometry.Hypercube(d * [0], d * [1])
    geom = dde.geometry.Rectangle([0, 0], [1, 1])
    k0 = 2 * np.pi * config.n
    wave_len = 1 / config.n

    hx_train = wave_len / config.precision_train
    nx_train = int(1 / hx_train)

    hx_test = wave_len / config.precision_test
    nx_test = int(1 / hx_test)

    if config.level != 0:
        nx_train = 10 * 2 ** (config.level - 1)
        nx_test = nx_test

    if config.hard_constraint == True:
        bc = []
    else:
        bc = dde.DirichletBC(geom, lambda x: 0, boundary)

    data = dde.data.PDE(
        geom,
        pde,
        bc,
        num_domain=nx_train**d,
        num_boundary=2 * d * nx_train,
        solution=func,
        num_test=nx_test**d,
    )

    net = dde.maps.FNN(
        [d] + [config.num_dense_nodes] * config.num_dense_layers + [1],
        config.activation,
        "Glorot uniform",
    )

    if config.hard_constraint == True:
        net.apply_output_transform(transform)

    model = dde.Model(data, net)
    print(nx_train, "nx_train")
    print(nx_train**d, "Ndof")
    print("here is the learning rate")
    print("learning_rate", config.learning_rate)
    if config.hard_constraint == True:
        model.compile("adam", lr=config.learning_rate, metrics=["l2 relative error"])
    else:
        loss_weights = [1, config.weights]
        model.compile(
            "adam",
            lr=config.learning_rate,
            metrics=["l2 relative error"],
            loss_weights=loss_weights,
        )
    return model


def train_model(model, config):
    callbacks = []
    if config.early == True:
        callbacks.append(
            dde.callbacks.EarlyStopping(
                min_delta=config.min_delta,
                patience=config.patience,
                monitor=config.monitor,
            )
        )
    import time

    ta = time.time()
    losshistory, train_state = model.train(epochs=config.epochs, callbacks=callbacks)
    texec = time.time() - ta

    train = np.array(losshistory.loss_train).sum(axis=1).ravel()
    test = np.array(losshistory.loss_test).sum(axis=1).ravel()
    metric = np.array(losshistory.metrics_test).sum(axis=1).ravel()

    accuracy = test.min()
    print(accuracy, "accuracy")

    skopt.dump(train, "results/" + config.name + "train.pkl")
    skopt.dump(test, "results/" + config.name + "test.pkl")
    skopt.dump(metric, "results/" + config.name + "metric.pkl")
    skopt.dump(texec, "results/" + config.name + "texec.pkl")

    if config.lbfgs == True:
        model.compile("L-BFGS")
        dde.optimizers.set_LBFGS_options(maxiter=config.maxiter)
        losshistory, train_state = model.train()
        train_bfgs = np.array(losshistory.loss_train).sum(axis=1).ravel()
        skopt.dump(train_bfgs, "results/" + config.name + "train_bfgs.pkl")

    return accuracy
