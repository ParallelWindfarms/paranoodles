## ------ language="Python" file="paranoodles/harmonic_oscillator.py"
from .abstract import (Problem)

def harmonic_oscillator(omega_0: float, zeta: float) -> Problem:
    def f(y, t):
        return np.r_[x[1], -2 * zeta * omega_0 * x[1] - omega_0**2 * x[0]]
    return f
    
## ------ begin <<harmonic-oscillator-solution>>[0]
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
## ------ end
## ------ end
