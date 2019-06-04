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

