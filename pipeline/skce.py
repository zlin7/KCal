from julia.api import Julia
jl = Julia(compiled_modules=False)
from pycalibration import ca
import numpy as np
import random
#random.seed(1234)
#predictions = [ca.Normal(random.gauss(0, 1), random.random()) for _ in range(100)]
#outcomes = [random.gauss(0, 1) for _ in range(100)]
_skce =  ca.SKCE(ca.tensor(ca.ExponentialKernel(), ca.WhiteKernel()), unbiased=True, blocksize=2)
#skce(predictions, outcomes)


def skce_eval(preds, y):
    if isinstance(preds, np.ndarray):
        preds = [p for p in preds]
    return _skce(preds, y)
if __name__ == '__main__':


    rng = np.random.default_rng(1234)
    predictions = [rng.dirichlet((3, 2, 5)) for _ in range(100)]
    outcomes = rng.integers(low=1, high=4, size=100)
    print(skce_eval(predictions, outcomes))
