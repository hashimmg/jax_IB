Copyright © 2025 SB Technology, Inc. (SandboxAQ). All Rights Reserved.
This software and its associated documentation are the confidential and proprietary property of SandboxAQ (“Company”) and are provided subject to the terms, conditions, and restrictions of the license agreement between the recipient and the Company. Any use, reproduction, modification, or distribution of this software outside the scope of the license agreement is strictly prohibited.
THIS SOFTWARE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND NON-INFRINGEMENT. IN NO EVENT SHALL THE COMPANY BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT, OR OTHERWISE, ARISING FROM, OUT OF, OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# jax_ib
Immersed Boundary implementation in jax-cfd library


## Overview

This repository contains an implementation of an immersed boundary method in jax-cfd package. The immersed boundary method is a numerical technique used to simulate the interaction between fluid flow and immersed objects, such as particles, structures, or boundaries.

## Features

- Simulates transport of multiple rigid bodies 
- Combines Brownian dynamics integration with CFD for passive particle transport simulations

## Installation

To use a local version of the code, follow these steps:

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/hashimmg/jax_ib.git
    ```

2. Navigate to the project directory:

    ```bash
    cd jax_ib
    ```

3. ```bash
   pip install -e .
   ```
   
### Example

The repository contains two examples:

- [Flapping of an ellipse airfoil](https://github.com/hashimmg/jax_ib/blob/main/jax_ib/notebooks/Flapping_Demo.ipynb)
- [Mixing in Journal bearing](https://github.com/hashimmg/jax_ib/blob/main/jax_ib/notebooks/journal_bearing_demo.ipynb)
- [Taylor Dispersion](https://github.com/hashimmg/jax_ib/blob/main/jax_ib/notebooks/taylor_dispersion_demo.ipynb)

### Other Packages Used
This project relies on the following external packages:

jax-cfd, jax-md

Citing External Packages
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



