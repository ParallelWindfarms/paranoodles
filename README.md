---
title: ParaNoodles
author: Johan Hidding
---

ParaNoodles is an implementation of the Parareal on top of the Noodles framework in Python. **Parareal** is an algorithm for Parallel-in-time integration of ODEs (or PDEs through method of lines). **Noodles** is a framework for parallel programming in Python.

# Installing

## Cloning
To clone this repo including the `bootstrap` submodule (for rendering the documentation),

```
git clone --recursive git@github.com:parallelwindfarms/paranoodles
```

If you forgot to use `--recursive` and want the submodule after all:

```
git submodule update --init --recursive
```

## Building ParaNoodles
ParaNoodles is 100% Python.

```
pip install .
```

To run on a cluster environment (with the Xenon runner) you need `pyxenon` installed and a decently recent Java Runtime present.

```
pip install pyxenon
```

## Building documentation
This is a literate code. To build the documentation you need [Pandoc &le; 2.7](https://pandoc.org), and several Pandoc filters installed,

```
pip install pandoc-eqnos pandoc-fignos entangled-filters
```

