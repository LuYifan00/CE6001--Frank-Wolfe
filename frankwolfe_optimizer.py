from __future__ import annotations

import numpy as np
from typing import Callable, Tuple, Optional, Dict, List
import warnings

warnings.filterwarnings('ignore')

class FrankWolfeOptimizer:
    """
    Frank-Wolfe (Conditional Gradient) optimizer for constrained convex optimization.

    Solves:
        minimize f(x)  s.t. x in C

    Requirements:
      - objective: Callable[[np.ndarray], float]
      - gradient: Callable[[np.ndarray], np.ndarray]
      - linear_oracle: Callable[[np.ndarray], np.ndarray]  (receives gradient and returns s ∈ C minimizing <grad, s>)
    """

    def __init__(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        line_search: str = "exact",
        verbose: bool = False,
    ):
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)
        self.line_search = line_search
        self.verbose = bool(verbose)

        # results
        self.solution: Optional[np.ndarray] = None
        self.objective_values: List[float] = []
        self.duality_gaps: List[float] = []
        self.iterations: int = 0
        self.converged: bool = False

    def solve(
        self,
        objective: Callable[[np.ndarray], float],
        gradient: Callable[[np.ndarray], np.ndarray],
        linear_oracle: Callable[[np.ndarray], np.ndarray],
        x0: np.ndarray,
        constraint_check: Optional[Callable[[np.ndarray], bool]] = None,
    ) -> Tuple[np.ndarray, float, Dict]:
        x = np.array(x0, dtype=float)
        self.objective_values = []
        self.duality_gaps = []
        self.iterations = 0
        self.converged = False

        # feasibility check
        if constraint_check and not constraint_check(x):
            raise ValueError("Initial point x0 is not feasible")

        # record initial objective value
        initial_obj = objective(x)
        self.objective_values.append(initial_obj)

        if self.verbose:
            print("=" * 70)
            print("Frank-Wolfe Algorithm")
            print("=" * 70)
            print(f"{'Iter':<8}{'f(x)':<18}{'Gap':<18}{'Step':<10}")
            print("-" * 70)
            print(f"{0:<8}{initial_obj:<18.6e}{'--':<18}{'--':<10}")

        for iteration in range(self.max_iterations):
            self.iterations = iteration + 1

            # compute gradient
            grad = gradient(x)

            # solve linear subproblem
            s = linear_oracle(grad)

            # duality gap
            gap = float(np.dot(grad, x - s))
            self.duality_gaps.append(gap)

            # convergence check
            if gap < self.tolerance:
                self.converged = True
                if self.verbose:
                    print(f"\nConverged! Duality gap = {gap:.2e} < {self.tolerance:.2e}")
                break

            # step size
            if self.line_search == "exact":
                gamma = self._exact_line_search(objective, gradient, x, s)
            elif self.line_search == "backtracking":
                gamma = self._backtracking_line_search(objective, gradient, x, s, grad)
            else:
                gamma = 2.0 / (iteration + 2.0)

            # safeguard gamma
            gamma = float(np.clip(gamma, 0.0, 1.0))

            # update
            x = x + gamma * (s - x)

            # store objective (after update)
            obj_val = objective(x)
            self.objective_values.append(obj_val)

            if self.verbose and (iteration % 10 == 0):
                print(f"{iteration+1:<8}{obj_val:<18.6e}{gap:<18.6e}{gamma:<10.4f}")

        if not self.converged and self.verbose:
            print(f"\nReached maximum iterations ({self.max_iterations})")

        self.solution = x
        final_value = float(objective(x))

        info = {
            "iterations": self.iterations,
            "converged": self.converged,
            "final_gap": self.duality_gaps[-1] if self.duality_gaps else float("inf"),
            "objective_values": self.objective_values,
            "duality_gaps": self.duality_gaps,
        }

        return x, final_value, info

    def _exact_line_search(
        self,
        objective: Callable[[np.ndarray], float],
        gradient: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        s: np.ndarray,
    ) -> float:
        """
        Try closed-form exact line search for quadratic-like objectives:
          If gradient is affine (i.e. grad(x + d) - grad(x) is linear in d),
          we can compute gamma = - (grad^T d) / (d^T A d) with A estimating curvature.

        Fallback: golden section search on [0,1].
        """
        d = s - x
        if np.linalg.norm(d) < 1e-14:
            return 0.0

        # estimate denominator using finite difference of gradient (works exactly for quadratic)
        grad_x = gradient(x)
        grad_x_plus_d = gradient(x + d)
        denom = float(np.dot(grad_x_plus_d - grad_x, d))
        numer = float(np.dot(grad_x, d))

        if abs(denom) > 1e-14:
            gamma = -numer / denom
            # clamp [0,1]
            gamma = float(np.clip(gamma, 0.0, 1.0))
            return gamma
        else:
            # fallback: golden-section search
            return self._golden_section_search(objective, x, s)

    def _golden_section_search(
        self,
        objective: Callable[[np.ndarray], float],
        x: np.ndarray,
        s: np.ndarray,
        a: float = 0.0,
        b: float = 1.0,
        tol: float = 1e-6,
    ) -> float:
        """
        Golden section search on gamma in [a,b] for minimization of
        phi(gamma) = objective(x + gamma*(s-x)).
        """
        phi = (1 + np.sqrt(5)) / 2.0

        # caching evaluations to reduce objective calls
        def f(g):
            return objective(x + g * (s - x))

        # initial interior points
        c = b - (b - a) / phi
        d = a + (b - a) / phi
        fc = f(c)
        fd = f(d)

        while abs(b - a) > tol:
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - (b - a) / phi
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + (b - a) / phi
                fd = f(d)

        return float((a + b) / 2.0)

    def _backtracking_line_search(
        self,
        objective: Callable[[np.ndarray], float],
        gradient: Callable[[np.ndarray], np.ndarray],
        x: np.ndarray,
        s: np.ndarray,
        grad: np.ndarray,
        alpha: float = 0.5,
        beta: float = 0.8,
    ) -> float:
        gamma = 1.0
        fx = objective(x)
        direction = s - x
        # Armijo condition
        while gamma > 1e-10:
            if objective(x + gamma * direction) <= fx + alpha * gamma * np.dot(grad, direction):
                break
            gamma *= beta
        return float(gamma)

    def get_convergence_history(self) -> Dict[str, List[float]]:
        return {"objective_values": self.objective_values, "duality_gaps": self.duality_gaps}


class ConvexProblem:
    """Helper to define common convex constraints + objective factories."""

    def __init__(self):
        self.objective_func: Optional[Callable[[np.ndarray], float]] = None
        self.gradient_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.oracle_func: Optional[Callable[[np.ndarray], np.ndarray]] = None
        self.x0: Optional[np.ndarray] = None
        self.constraint_check: Optional[Callable[[np.ndarray], bool]] = None
        self.problem_name: str = "Unnamed Problem"

    def set_quadratic_objective(self, A: np.ndarray, b: np.ndarray, c: float = 0.0):
        """
        f(x) = 0.5 x^T A x + b^T x + c
        A should be symmetric (preferably PSD).
        """
        A = np.array(A, dtype=float)
        b = np.array(b, dtype=float).flatten()

        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square 2D matrix")
        n = A.shape[0]
        if b.size != n:
            raise ValueError("b must have length matching A.shape[0]")

        # symmetrize A
        if not np.allclose(A, A.T):
            A = 0.5 * (A + A.T)

        def obj(x: np.ndarray) -> float:
            x = np.asarray(x, dtype=float)
            return 0.5 * float(np.dot(x, np.dot(A, x))) + float(np.dot(b, x)) + float(c)

        def grad(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            return np.dot(A, x) + b

        self.objective_func = obj
        self.gradient_func = grad
        return self

    def set_custom_objective(self, objective: Callable[[np.ndarray], float], gradient: Callable[[np.ndarray], np.ndarray]):
        self.objective_func = objective
        self.gradient_func = gradient
        return self

    def set_box_constraints(self, lower: np.ndarray, upper: np.ndarray):
        lower = np.array(lower, dtype=float)
        upper = np.array(upper, dtype=float)

        if lower.shape != upper.shape:
            raise ValueError("lower and upper must be same shape")

        def oracle(grad: np.ndarray) -> np.ndarray:
            # min <grad, x> s.t. lower <= x <= upper
            return np.where(grad > 0, lower, upper)

        def check(x: np.ndarray) -> bool:
            x = np.array(x, dtype=float)
            return np.all(x >= lower - 1e-10) and np.all(x <= upper + 1e-10)

        self.oracle_func = oracle
        self.constraint_check = check
        return self

    def set_simplex_constraint(self, n: int, radius: float = 1.0):
        def oracle(grad: np.ndarray) -> np.ndarray:
            x = np.zeros(n, dtype=float)
            x[np.argmin(grad)] = radius
            return x

        def check(x: np.ndarray) -> bool:
            return np.all(x >= -1e-10) and abs(np.sum(x) - radius) < 1e-6

        self.oracle_func = oracle
        self.constraint_check = check
        return self

    def set_l1_ball_constraint(self, n: int, radius: float = 1.0):
        """
        ||x||_1 <= radius

        Oracle: put all mass on coordinate with largest |grad_i| with sign opposite to grad_i.
        """
        def oracle(grad: np.ndarray) -> np.ndarray:
            x = np.zeros(n, dtype=float)
            i = int(np.argmax(np.abs(grad)))  # argmax (largest magnitude)
            if grad[i] == 0:
                x[i] = -radius
            else:
                x[i] = -radius * np.sign(grad[i])
            return x

        def check(x: np.ndarray) -> bool:
            return np.sum(np.abs(x)) <= radius + 1e-6

        self.oracle_func = oracle
        self.constraint_check = check
        return self

    def set_l2_ball_constraint(self, n: int, radius: float = 1.0):
        def oracle(grad: np.ndarray) -> np.ndarray:
            grad = np.asarray(grad, dtype=float)
            grad_norm = np.linalg.norm(grad)
            if grad_norm > 1e-12:
                return -radius * grad / grad_norm
            else:
                return np.zeros(n, dtype=float)

        def check(x: np.ndarray) -> bool:
            return np.linalg.norm(x) <= radius + 1e-6

        self.oracle_func = oracle
        self.constraint_check = check
        return self

    def set_initial_point(self, x0: np.ndarray):
        self.x0 = np.array(x0, dtype=float)
        return self

    def solve(
        self,
        max_iterations: int = 1000,
        tolerance: float = 1e-6,
        line_search: str = "exact",
        verbose: bool = False,
    ) -> Dict:
        if self.objective_func is None or self.gradient_func is None:
            raise ValueError("Objective and gradient must be set")
        if self.oracle_func is None:
            raise ValueError("Constraint (linear oracle) must be set")
        if self.x0 is None:
            raise ValueError("Initial point must be set")

        optimizer = FrankWolfeOptimizer(
            max_iterations=max_iterations, tolerance=tolerance, line_search=line_search, verbose=verbose
        )

        sol, val, info = optimizer.solve(self.objective_func, self.gradient_func, self.oracle_func, self.x0, self.constraint_check)

        return {
            "solution": sol,
            "optimal_value": float(val),
            "converged": bool(info["converged"]),
            "iterations": int(info["iterations"]),
            "final_gap": float(info["final_gap"]),
            "objective_values": info["objective_values"],
            "duality_gaps": info["duality_gaps"],
        }


def print_fw_solution(result: Dict, problem_name: str = ""):
    print("\n" + "=" * 70)
    if problem_name:
        print(f"SOLUTION: {problem_name}")
    else:
        print("FRANK-WOLFE SOLUTION")
    print("=" * 70)

    print(f"\nStatus: {'CONVERGED' if result['converged'] else 'MAX ITERATIONS REACHED'}")
    print(f"Iterations: {result['iterations']}")
    print(f"Final Duality Gap: {result['final_gap']:.6e}")
    print(f"Optimal Value: {result['optimal_value']:.6f}")

    solution = result["solution"]
    print("\nOptimal Solution:")
    if solution is None:
        print("  None")
    else:
        if solution.size <= 20:
            for i, val in enumerate(solution):
                print(f"  x[{i}] = {val:.6f}")
        else:
            print(f"  [Vector of length {len(solution)}]")
            print(f"  First 5 elements: {solution[:5]}")
            print(f"  Last 5 elements: {solution[-5:]}")
    print("=" * 70 + "\n")


def plot_convergence(result: Dict, title: str = "Frank-Wolfe Convergence"):
    try:
        import matplotlib.pyplot as plt

        obj_vals = result.get("objective_values", [])
        gaps = result.get("duality_gaps", [])

        if len(obj_vals) == 0 and len(gaps) == 0:
            print("No convergence history available to plot.")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        if len(obj_vals) > 0:
            ax1.plot(obj_vals, linewidth=2)
            ax1.set_xlabel("Iteration")
            ax1.set_ylabel("Objective Value")
            ax1.set_title("Objective Value History")
            ax1.grid(True, alpha=0.3)
        else:
            ax1.text(0.5, 0.5, "No objective history", ha="center", va="center")
            ax1.axis("off")

        if len(gaps) > 0:
            ax2.semilogy(gaps, linewidth=2)
            ax2.set_xlabel("Iteration")
            ax2.set_ylabel("Duality Gap")
            ax2.set_title("Duality Gap (Log Scale)")
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, "No duality gap history", ha="center", va="center")
            ax2.axis("off")

        plt.suptitle(title)
        # avoid tight_layout(); adjust spacing manually
        fig.subplots_adjust(top=0.86, wspace=0.35)
        plt.show()

    except ImportError:
        print("Matplotlib not available. Install it to visualize convergence.")