---
title: ParaNoodles
author: Johan Hidding
---

ParaNoodles is an implementation of the Parareal on top of the Noodles framework in Python. **Parareal** is an algorithm for Parallel-in-time integration of ODEs (or PDEs through method of lines). **Noodles** is a framework for parallel programming in Python.

# Building ParaNoodles

ParaNoodles is 100% Python. Requirements are placed in `requirements.txt`.

```
pip install -r requirements.txt
```

To run on a cluster environment (with the Xenon runner) you need `pyxenon` installed and a decently recent Java Runtime present.

```
pip install pyxenon
```

# Problem statement

I tried to implement the problem statement using abstract base classes (`ABC` module) and the `typing` module. However, type annotation in Python is still an immature feature (to say the least, it's next to useless). The little annotation remaining should be considered documentation.

``` {.python file=paranoodles/abstract.py}
from typing import Callable
from abc import (ABC, abstractmethod)

<<abstract-types>>
```

We have an ODE in the form

$$y' = f(y, t).$${#eq:ode}

Here $y$ can be a scalar value, a vector of values (say a `numpy` array), or any expression of *state*. A naive implementation of an ODE integrator would be

$$y_{n+1} = y_{n} + \Delta t f(y_{n}, t).$${#eq:euler-method}

+@eq:euler-method is known as the *forward Euler method*. We can capture the *state* $y$ in an abstract class `Vector`

``` {.python #abstract-types}
class Vector(ABC):
    """Abstract base class for state variable of a problem.
    This should support simple arithmatic operations."""
    @abstractmethod
    def __add__(self, other: Vector) -> Vector:
        """Summation of two result vectors."""
        pass

    @abstractmethod
    def __sub__(self, other: Vector) -> Vector:
        """Difference between two result vectors."""
        pass
    
    @abstractmethod
    def __mul__(self, other: float) -> Vector:
        """Scale vector with scalar."""
	pass

    def __rmul__(self, other: float) -> Vector
	return self * other
```

Note that we don't make a distinction here between a state vector and a vector representing a change in state. This may change in the future.

An ODE is then given as a function taking a `Vector` and a `float` returning a `Vector`. We define the type `Problem`:

``` {.python #abstract-types}
Problem = Callable[[Vector, float], Vector]
```

If we have a `Problem`, we're after a `Solution`: a function that, given an initial `Vector`, initial time and final time, gives the resulting `Vector`.

``` {.python #abstract-types}
Solution = Callable[[Vector, float, float], Vector]
```

Then the forward Euler method (+@eq:euler-method), is given by

``` {.python file=paranoodles/forward_euler.py}
from .abstract import (Vector, Problem, Solution)

def forward_euler(f: Problem) -> Solution:
    def step(y: Vector, t_0: float, t_1: float) -> Vector:
        """Stepping function of Euler method."""
        return y + (t_1 - t_0) * f(y, t_0)
    return step
```

Any existing solution can be iterated over to provide a solution over a larger time interval. The `iterate_solution` function runs a given solution with a step-size fixed to $\Delta t = h$.

<!--$${\rm Iter}[S, h]\Big|_{t_0, y = y}^{t_1} = \begin{cases}-->
<!--y & t_0 = t_1 \\-->
<!--{\rm Iter}[S, h]\big|_{t_0 + h, y = S(y, t_0, t_0 + h)}^{t_1} & {\rm otherwise}-->
<!--\end{cases}.$$-->

``` {.python file=paranoodles/iterators.py}
from .abstract import (Vector, Problem, Solution)
import numpy as np

def iterate_solution(step: Solution, h: float) -> Solution:
    def iter_step(y: Vector, t_0: float, t_1: float) -> Vector:
        """Stepping function of iterated solution."""
        n = math.ceil((t_1 - t_0) / h)
        steps = np.arange(t_0, t_1, n + 1)
        for t_a, t_b in zip(steps[:-1], steps[1:])
            y = step(y, t_a, t_b)
        return y         
    return iter_step
```

## Example: damped harmonic oscillator

The harmonic oscillator can model the movement of a pendulum or the vibration of a mass on a string. 

$$y'' + 2\zeta \omega_0 y' + \omega_0^2 y = 0,$$

where $\omega_0 = \sqrt{k/m}$ and $\zeta = c / 2\sqrt{mk}$, $k$ being the spring constant, $m$ the test mass and $c$ the friction constant.

To solve this second order ODE we need to introduce a second variable to solve for. Say $q = y$ and $p = y'$.

$$\begin{aligned}
    q' &= p\\
    p' &= -2\zeta \omega_0 p + \omega_0^2 q
\end{aligned}$$ {#eq:harmonic-oscillator}

The `Problem` is then given as

``` {.python file=paranoodles/harmonic_oscillator.py}
from .abstract import (Problem)

def harmonic_oscillator(omega_0: float, zeta: float) -> Problem:
    def f(y, t):
        return np.r_[x[1], -2 * zeta * omega_0 * x[1] - omega_0**2 * x[0]]
    return f
    
<<harmonic-oscillator-solution>>
```

### Exact solution
The damped harmonic oscillator has an exact solution, given the ansatz $y = A \exp(z t)$, we get
$$z = \omega_0\left(-\zeta \pm \sqrt{\zeta^2 - 1}\right),$$
and in the case of underdamped motion ($\zeta < 1$), we can parametrize $-\zeta = {\rm Re} \exp(i \xi)$ and $\pm \sqrt{1 - \zeta^2} = {\rm Im} \exp(i \xi)$, writing $z = \omega_0 \exp(i\xi)$.

$$y = A\quad \underbrace{\exp(-\omega_0\zeta t)}_{\rm dampening}\quad\underbrace{\exp(\pm i \omega_0 \sqrt{1 - \zeta^2} t)}_{\rm oscillation},$$
where $A$ is a complex scalar, giving the amplitude and phase of the solution.

Also,

$$p = A z\ \exp(z t) = A \omega_0\ \exp(i\xi) \exp(-\omega_0\zeta t)\ \exp(\pm i \omega_0 \sqrt{1 - \zeta^2} t),$$

effectively scaling the solution for $q$ with a factor $\omega_0$ and phase-shifting it with $\xi$.

Given an initial condition $q_0 = 1, p_0 = 0$, the solution is computed as

``` {.python #harmonic-oscillator-solution}
def solution(omega_0, zeta):
    A = 1 / np.sqrt(1 - zeta**2)
    xi = np.arccos(-zeta)
    phi = np.pi/2 - xi
    freq = omega_0 * np.sqrt(1 - zeta**2)
    
    def f(t):
        dampening = np.exp(-omega_0*zeta*t)
        q = A * dampening * np.cos(freq * t + phi)
        p = - A * omega_0 * dampening * np.sin(freq*t)
        return np.c_[q, p]
    return f
```




