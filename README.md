Copyright © 2025 SB Technology, Inc. (SandboxAQ). All Rights Reserved.
This software and its associated documentation are the confidential and proprietary property of SandboxAQ (“Company”) and are provided subject to the terms, conditions, and restrictions of the license agreement between the recipient and the Company. Any use, reproduction, modification, or distribution of this software outside the scope of the license agreement is strictly prohibited.
THIS SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COMPANY BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# jax_ib
Immersed Boundary implementation in jax-cfd library


## Overview

This repository contains an implementation of an immersed boundary method in jax-cfd package.The immersed boundary method is a numerical technique used to simulate the interaction between fluid flow and immersed objects, such as particles, structures, or boundaries.

## Features

- Simulates transport of multiple rigid bodies

## Installation

This code requires the GPU-aware jax release `"jax[cuda12]==0.4.35"`. Later release might work, but have not been tested.
We recommend installation using the `conda`, `mamba` or `micromamba` service through the provided `environment.yml` file
in this repo. From the root directory of jax_ib, run

```bash
micromamba create -n jax-cfd --file environment.yml
micromamba activate jax-cfd
pip install -e .
python scripts/flap-opt.py --config scripts/config.yml
```

## Usage
The main script is located in `scripts/flap-opt.py`. Ths script implements two examples:
1. forward simulation of a moving cylinder
2. optimizing swimming efficiency of an ellipse
Both examples are currently limited to 2d and periodic boundary conditions.

The code implements a 2d domain decomposition to parallelize the simulation. The simulation is
mapped accross a 2d grid of devices using `jax.lax` collectives and `jax`'s `shard_map` API.

Run
```bash
python flap-opt.py --help
```
for more information regarding parameter settings. Parameters can be modified through command line,
or by passing in a yml file (see script/config.yml for an example), i.e.
```bash
python flap-opt.py --config config.yml --N1 128 --N2 128
```
In this case the value for `N1` and `N2` in the `config.yml` fle will be overriden be the values passed
in through the command line.


## Docker

This repository ships with a Dockerfile which can be used for containerization of the application.
To build the container run
```bash
docker build -t jax-cfd .
```

The container entrypoint is a shell script calling `scripts/flap-opt.py` with the default
configuration `scripts/config.yml` inside the container. The user can pass additional parameters
to the container which will override defaults in `config.yml`. To store results to disk the user
can mount a local directory `local-dir` to the container (accessed as a user-defined name`<container-dir>`
inside the container` to which the results will be stored:
```bash
docker run --gpus all --shm-size=256GB -v local-dir:<container-dir> -it jax-cfd --path <container-dir> --N1 128 --N2 128 # more parameters can be passed
```
The default value for `--path` used by the entrypoint is `/out-path`.
The user can run
```bash
docker run --gpus all --shm-size=2GB -v local-dir:<container-dir> -it jax-cfd --help
```
to print a list of parameters with their default values.

Note the flags `--gpus` and `--shm-size` in the `docker run` call. The first one exposes the available GPUs to the container, while
the second increases the size of `/dev/shm` from a default of 64MB to 2GB. Too small sizes of `/dev/shm` can result in core dumps
of the application.


## Citing External Packages
If you use this code in your research, please ensure to cite the relevant works. Here are the citations for the packages used in this project:

Package 1: jax-cfd
```bash
@article{Kochkov2021-ML-CFD,
  author = {Kochkov, Dmitrii and Smith, Jamie A. and Alieva, Ayya and Wang, Qing and Brenner, Michael P. and Hoyer, Stephan},
  title = {Machine learning{\textendash}accelerated computational fluid dynamics},
  volume = {118},
  number = {21},
  elocation-id = {e2101784118},
  year = {2021},
  doi = {10.1073/pnas.2101784118},
  publisher = {National Academy of Sciences},
  issn = {0027-8424},
  URL = {https://www.pnas.org/content/118/21/e2101784118},
  eprint = {https://www.pnas.org/content/118/21/e2101784118.full.pdf},
  journal = {Proceedings of the National Academy of Sciences}
}
```
Package 2: jax-md
```bash
@inproceedings{jaxmd2020,
 author = {Schoenholz, Samuel S. and Cubuk, Ekin D.},
 booktitle = {Advances in Neural Information Processing Systems},
 publisher = {Curran Associates, Inc.},
 title = {JAX M.D. A Framework for Differentiable Physics},
 url = {https://papers.nips.cc/paper/2020/file/83d3d4b6c9579515e1679aca8cbc8033-Paper.pdf},
 volume = {33},
 year = {2020}
}
```



