# ~\~ language=Python filename=paranoodles/__init__.py
# ~\~ begin <<lit/paranoodles.md|paranoodles/__init__.py>>[0]
from .tabulate_solution import tabulate
from .parareal import parareal, parareal_mapping
from . import abstract

from noodles import schedule
from noodles.run.threading.sqlite3 import run_parallel as run

__all__ = ["tabulate", "parareal", "parareal_mapping", "schedule",
           "run", "abstract", "schedule"]
# ~\~ end
