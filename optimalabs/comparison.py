from numpy import ndarray, asarray, arange, abs, sign, zeros, sum
from optimalabs.nlp import ralgb5a, emshor


def main():
    print('ralgb5a')
    print('-' * 50, end='\n\n')
    print(ralgb5a(mse, asarray([[12], [-3]])))
    print(ralgb5a(rosenbrock, asarray([[0], [3]])))
    print(ralgb5a(rosenbrock, asarray([[0], [3]]), h0=2.25))
    print(ralgb5a(piecewise_linear, zeros((100, 1), dtype=float), h0=10.0, q1=1.0, max_iter=5000))
    print(ralgb5a(piecewise_linear2, zeros((20, 1), dtype=float), h0=10.0, q1=1.0))
    print('\n\nemshor')
    print('-' * 50, end='\n\n')
    print(emshor(mse, asarray([[12], [-3]]), 64))
    print(emshor(rosenbrock, asarray([[0], [3]]), 64))
    print(emshor(piecewise_linear, zeros((100, 1), dtype=float), 64, 1e-6, max_iter=500000))
    print(emshor(piecewise_linear2, zeros((20, 1), dtype=float), 64, max_iter=20000))


def mse(x: ndarray) -> tuple[float, ndarray]:
    n = x.shape[0]
    return sum(x ** 2) / n, x * 2 / n


def rosenbrock(u: ndarray) -> tuple[float, ndarray]:
    a, b = 1, 100
    x, y = u[0, 0], u[1, 0]
    f = (a - x) ** 2 + b * (y - x ** 2) ** 2
    g = asarray([[2 * (x - a) - 4 * b * x * (y - x ** 2)], [2 * b * (y - x ** 2)]])
    return f, g


def piecewise_linear(x: ndarray) -> tuple[float, ndarray]:
    q = 1.2 ** arange(x.shape[0]).reshape(-1, 1)
    x1 = x - 1
    return (q.T @ abs(x1)).item(), q * sign(x1)


def piecewise_linear2(x: ndarray) -> tuple[float, ndarray]:
    q = 2 ** arange(x.shape[0]).reshape(-1, 1)
    x1 = x - 1
    return (q.T @ abs(x1)).item(), q * sign(x1)


if __name__ == '__main__':
    main()
