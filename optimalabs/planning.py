from numpy import (
    ndarray,
    diag,
    asarray,
    zeros,
    inf,
    repeat,
    ones,
    vstack,
    insert,
    maximum,
    sum,
    reshape,
    where,
    full
)
from scipy.optimize import Bounds, LinearConstraint, OptimizeResult, minimize
from optimalabs.nlp import Func, ExitCode, ralgb5a, emshor

SUCCESS_CODES = {ExitCode.eps_f, ExitCode.eps_g, ExitCode.eps_x}


def minimize_planning_constrained(
    c: ndarray,
    d: ndarray,
    s: float,
    a: ndarray,
    b: ndarray
) -> OptimizeResult:
    m, n = len(b), len(c)
    x0 = full(n, s // n)
    x0[: s % n] += 1
    bounds = Bounds(zeros(n), full(n, inf))
    constraint = LinearConstraint(
        vstack([ones((1, n)), a]),
        insert(zeros(m), 0, s, 0),
        insert(b, 0, s, 0)
    )
    return minimize(
        planning_objective,
        x0,
        (c, d),
        'trust-constr',
        planning_jacobian,
        planning_hessian,
        bounds=bounds,
        constraints=constraint
    )


def planning_objective(x: ndarray, c: ndarray, d: ndarray) -> float:
    return c @ x + d @ x ** 2


def planning_jacobian(x: ndarray, c: ndarray, d: ndarray) -> ndarray:
    return c + 2 * d * x


def planning_hessian(_x: ndarray, _c: ndarray, d: ndarray) -> ndarray:
    return diag(2 * d)


def minimize_planning_with_penalty(
    c: ndarray,
    d: ndarray,
    s: float,
    a: ndarray,
    b: ndarray
) -> OptimizeResult:
    n = len(c)
    x0 = full(n, s // n)
    x0[: s % n] += 1
    return minimize(
        planning_with_penalty_objective,
        x0,
        (c, d, s, a, b),
        'BFGS',
        planning_with_penalty_jacobian
    )


def planning_with_penalty_objective(
    x: ndarray,
    c: ndarray,
    d: ndarray,
    s: float,
    a: ndarray,
    b: ndarray,
    mu: float = 1e6
) -> float:
    alpha = sum(maximum(a @ x - b, 0) ** 2) + sum(maximum(-x, 0) ** 2) + (sum(x) - s) ** 2
    return planning_objective(x, c, d) + mu * alpha


def planning_with_penalty_jacobian(
    x: ndarray,
    c: ndarray,
    d: ndarray,
    s: float,
    a: ndarray,
    b: ndarray,
    mu: float = 1e6
) -> ndarray:
    g = repeat(reshape(a @ x - b, (-1, 1)), len(c), 1)
    alpha = 2 * sum(maximum(g, 0) * where(g >= 0, a, 0), 0)
    alpha += 2 * maximum(-x, 0) * where(x <= 0, -1, 0) + 2 * (sum(x) - s)
    return planning_jacobian(x, c, d) + mu * alpha


def minimize_planning_ralgb5a(
    c: ndarray,
    d: ndarray,
    s: float,
    a: ndarray,
    b: ndarray
) -> OptimizeResult:
    n = len(c)
    x0 = full((n, 1), s // n)
    x0[: s % n] += 1
    result = ralgb5a(make_planning_with_penalty_joint(c, d, s, a, b), x0)
    return OptimizeResult(
        x=result.x_r.reshape(-1),
        success=(result.exit_code in SUCCESS_CODES),
        status=result.exit_code.value,
        message=result.exit_code.name,
        fun=planning_with_penalty_objective(result.x_r, c, d, s, a, b),
        nit=result.n_iter,
        nfev=result.n_calls,
        njev=result.n_calls
    )


def make_planning_with_penalty_joint(
    c: ndarray,
    d: ndarray,
    s: float,
    a: ndarray,
    b: ndarray,
    mu: float = 1e6
) -> Func:
    def func(x: ndarray) -> tuple[float, ndarray]:
        x0 = x.reshape(-1)
        f = planning_with_penalty_objective(x0, c, d, s, a, b, mu)
        g = planning_with_penalty_jacobian(x0, c, d, s, a, b, mu)
        return f, g.reshape((-1, 1))

    return func


def minimize_planning_emshor(
    c: ndarray,
    d: ndarray,
    s: float,
    a: ndarray,
    b: ndarray
) -> OptimizeResult:
    n = len(d)
    x0 = full((n, 1), s // n)
    x0[: s % n] += 1
    result = emshor(make_planning_with_penalty_joint(c, d, s, a, b), x0, 1e2, 1e-2)
    return OptimizeResult(
        x=result.x_r.reshape(-1),
        success=(result.exit_code in SUCCESS_CODES),
        status=result.exit_code.value,
        message=result.exit_code.name,
        fun=planning_with_penalty_objective(result.x_r, c, d, s, a, b),
        nit=result.n_iter,
        nfev=result.n_calls,
        njev=result.n_calls
    )


def main():
    c = asarray([0.6, 2.6, 9.15])
    d = asarray([0.19, 0.08, 1.5])
    s = 11
    a = asarray(
        [
            [2.567, 7.21, 0.12],
            [1.98, 7.3, 6.21],
            [0.31, 2.721, 8.12],
            [1.372, 4.203, 2.91]
        ]
    )
    b = asarray([95, 150.67, 23.76, 61.21])
    print('SciPy, trust-region, bounds and linear constraints\n')
    print(minimize_planning_constrained(c, d, s, a, b))
    print('-' * 100)
    print('SciPy, BFGS, penalty function\n')
    print(minimize_planning_with_penalty(c, d, s, a, b))
    print('-' * 100)
    print('Ours, ralgb5a, penalty function\n')
    print(minimize_planning_ralgb5a(c, d, s, a, b))
    print('-' * 100)
    print('Ours, emshor, penalty function\n')
    print(minimize_planning_emshor(c, d, s, a, b))


if __name__ == '__main__':
    main()
