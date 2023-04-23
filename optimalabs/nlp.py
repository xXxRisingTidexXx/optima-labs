from enum import Enum, unique
from typing import NamedTuple, Callable, Optional
from numpy import ndarray, eye, asarray, sqrt
from numpy.linalg import norm


def ralgb5a(
    func: 'Func',
    x0: ndarray,
    alpha: float = 4.0,
    h0: float = 1.0,
    q1: float = 0.9,
    eps_x: float = 1e-10,
    eps_g: float = 1e-12,
    max_iter: int = 500
) -> 'Result':
    if len(x0.shape) != 2 or x0.shape[1] != 1:
        raise ValueError(f'Vector x0 must have a shape of (n, 1), got {x0.shape}')
    if alpha < 1:
        raise ValueError(f'Expected alpha > 1, got {alpha}')
    if eps_x <= 0:
        raise ValueError(f'Eps for x must be positive, got {eps_x}')
    if eps_g <= 0:
        raise ValueError(f'Eps for g must be positive, got {eps_g}')
    if max_iter < 1:
        raise ValueError(f'Must be at least 1 iteration, got {max_iter}')
    x_r, n_iter, n_calls = x0.astype(float), 0, 1
    f_r, g = func(x_r)
    if x_r.shape != g.shape:
        raise ValueError(f'Gradient must have a shape of {x0.shape}, got {g.shape}')
    if norm(g) < eps_g:
        return Result(x_r, n_iter, n_calls, ExitCode.eps_g)
    x, h_s = x_r, h0
    b = eye(x.shape[0])
    m = (1 / alpha - 1)
    n_iter = 1
    while n_iter <= max_iter:
        dx = b @ normalize(b.T @ g)
        regulation = regulate_h(func, x, h_s, dx, x_r, f_r, q1, eps_x, eps_g)
        x, h_s, x_r, f_r = regulation.x, regulation.h_s, regulation.x_r, regulation.f_r
        n_calls += regulation.n_calls
        if regulation.exit_code is not None:
            return Result(x_r, n_iter, n_calls, regulation.exit_code)
        eta = normalize(b.T @ (regulation.g - g))
        b += m * b @ eta @ eta.T
        g = regulation.g
        n_iter += 1
    return Result(x_r, n_iter, n_calls, ExitCode.max_iter)


Func = Callable[[ndarray], tuple[float, ndarray]]


class Result(NamedTuple):
    x_r: ndarray
    n_iter: int
    n_calls: int
    exit_code: 'ExitCode'


@unique
class ExitCode(Enum):
    eps_f = 1
    eps_g = 2
    eps_x = 3
    max_iter = 4
    error = 5


def normalize(x: ndarray) -> ndarray:
    return x / norm(x)


def regulate_h(
    func: Func,
    x0: ndarray,
    h_s0: float,
    dx: ndarray,
    x_r0: ndarray,
    f_r0: float,
    q1: float,
    eps_x: float,
    eps_g: float
) -> 'Regulation':
    x, h_s = x0, h_s0
    g = asarray([], dtype=float)
    x_r, f_r = x_r0, f_r0
    d, ddx = 1, 0
    l2 = norm(dx)
    n_steps, n_calls = 0, 0
    while d > 0:
        x -= h_s * dx
        ddx += h_s * l2
        f, g = func(x)
        n_calls += 1
        if f < f_r:
            x_r, f_r = x, f
        if norm(g) < eps_g:
            return Regulation(x, h_s, g, x_r, f_r, n_calls, ExitCode.eps_g)
        n_steps += 1
        if n_steps % 3 == 0:
            h_s *= 1.1
        if n_steps > 500:
            return Regulation(x, h_s, g, x_r, f_r, n_calls, ExitCode.error)
        d = dx.T @ g
    return Regulation(
        x,
        h_s * q1 if n_steps == 1 else h_s,
        g,
        x_r,
        f_r,
        n_calls,
        ExitCode.eps_x if ddx < eps_x else None
    )


class Regulation(NamedTuple):
    x: ndarray
    h_s: float
    g: ndarray
    x_r: ndarray
    f_r: float
    n_calls: int
    exit_code: Optional[ExitCode]


def emshor(
    func: Func,
    x0: ndarray,
    r0: float,
    eps_f: float = 1e-9,
    max_iter: int = 500
) -> Result:
    if len(x0.shape) != 2 or x0.shape[1] != 1:
        raise ValueError(f'Vector x0 must have a shape of (n, 1), got {x0.shape}')
    if r0 <= 0:
        raise ValueError(f'Radius r0 must be positive, got {r0}')
    if eps_f <= 0:
        raise ValueError(f'Eps for f must be positive, got {eps_f}')
    if max_iter < 1:
        raise ValueError(f'Must be at least 1 iteration, got {max_iter}')
    x, n_iter, n_calls = x0.astype(float), 0, 0
    n, r_n = x.shape[0], r0
    f, g = func(x)
    if x.shape != g.shape:
        raise ValueError(f'Gradient must have a shape of {x0.shape}, got {g.shape}')
    beta = sqrt((n - 1) / (n + 1))
    b = eye(n)
    while n_iter < max_iter:
        f, g = func(x)
        n_calls += 1
        g = b.T @ g
        d_g = norm(g)
        if r_n * d_g < eps_f:
            return Result(x, n_iter, n_calls, ExitCode.eps_f)
        x_i = (1 / d_g) * g
        d_x = b @ x_i
        h_s = r_n / (n + 1)
        x -= h_s * d_x
        b += (beta - 1) * b @ x_i @ x_i.T
        r_n *= n / sqrt(n * n - 1)
        n_iter += 1
    return Result(x, n_iter, n_calls, ExitCode.max_iter)
