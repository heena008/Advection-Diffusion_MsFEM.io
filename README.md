# Advection-Diffusion_MsFEM.io

A C++ solver for time-dependent advection-diffusion problems, built on the [deal.II](https://www.dealii.org/) finite element library. It contains two solvers that can be compared against each other:

- a **fine-scale finite element solver** that works on the full mesh, and
- a **multiscale finite element (MsFEM) solver** that builds special basis functions on a coarse mesh to capture small-scale behaviour at a lower cost.

The code runs in parallel with MPI and uses Trilinos for the linear algebra.

## The problem

The code solves the equation

```
du/dt - div(A grad u) + b . grad u = f
```

where

| Symbol | Meaning |
|--------|---------|
| `u` | the unknown quantity (for example, concentration of a substance) |
| `A` | the diffusion coefficient (`include/diffusion_coefficient.hpp`, `include/matrix_coeff.hpp`) |
| `b` | the advection field, i.e. the "wind" that transports `u` (`include/advection_field.hpp`) |
| `f` | the source term (`include/right_hand_side.hpp`) |

Time stepping uses the theta scheme. Setting `theta = 1` gives implicit Euler (the default), `theta = 0.5` gives Crank-Nicolson, and `theta = 0` gives explicit Euler. The default time step is `1/100`.

The problem can be run on a domain with periodic boundary conditions or with mixed Dirichlet-Neumann boundary conditions. The example in `main` uses periodic boundaries in 2D, with an initial condition made of two Gaussian bumps (`include/initial_value.hpp`).

## Project structure

```
.
├── include/                              Header files (.hpp, .h)
│   ├── advectiondiffusion_problem.hpp        Fine-scale solver
│   ├── advectiondiffusion_multiscale.hpp     Multiscale (MsFEM) solver
│   ├── advectiondiffusion_basis.hpp          Multiscale basis functions
│   ├── basis_interface.hpp                   Common interface for basis types
│   ├── basis_frame_work.hpp                  Basis framework
│   ├── q1basis.hpp                           Q1 basis functions
│   ├── semilagrangian.hpp                    Semi-Lagrangian method for the advection part
│   ├── runge_kutta.hpp                       Runge-Kutta time integration helper
│   ├── reconstruction_base.hpp               Reconstruction of the fine-scale solution
│   ├── reconstruction_assembler.hpp          Assembly of reconstruction operators
│   ├── H1_conformal.hpp, L2_conformal.hpp    Conforming H1 / L2 spaces
│   ├── Multiscale_FEFieldFunction.hpp        Evaluating a multiscale solution as a function
│   ├── advection_field.hpp                   Advection field b
│   ├── diffusion_coefficient.hpp             Diffusion coefficient
│   ├── matrix_coeff.hpp                      Matrix-valued diffusion coefficient A
│   ├── reaction_rate.hpp                     Reaction rate
│   ├── right_hand_side.hpp                   Source term f
│   ├── initial_value.hpp                     Initial condition
│   ├── boundary_value.hpp                    Boundary values
│   ├── dirichlet_bc.hpp, neumann_bc.hpp      Boundary conditions
│   ├── solution_gaussian.hpp                 Gaussian reference solution
│   ├── parameter.h, advection_diffusion_parameter.h   Parameter handling
│   ├── referencevalue.h, coefficients.h, config.h     Constants and settings
│   └── ...
├── source/
│   ├── AdvectionDiffusionProblem.cc          Main program (entry point)
│   ├── mpi.cc                                MPI and library header includes
│   ├── intlog.h                              Logging helper
│   └── parameters.prm                        Runtime parameters
├── LICENSE                                   GNU GPL v3
└── README.md
```

## Requirements

- A C++ compiler supported by your deal.II version
- [deal.II](https://www.dealii.org/) configured with **MPI** and **Trilinos** (the code stops with an error if Trilinos is missing)
- [p4est](https://www.dealii.org/developer/external-libs/p4est.html) for distributed meshes
- Boost (filesystem and iostreams are used)
- CMake

Some includes in `source/mpi.cc` also refer to PETSc, SLEPc, and Zoltan, so a deal.II build that includes these is the safest choice.

## Building

The repository does not yet include a working `CMakeLists.txt` (the two files in `include/` and `source/` are empty). A minimal deal.II project file placed in the repository root looks like this:

```cmake
cmake_minimum_required(VERSION 3.13.4)

find_package(deal.II REQUIRED HINTS ${DEAL_II_DIR} $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

project(advection_diffusion_msfem)

include_directories(${CMAKE_SOURCE_DIR}/include ${CMAKE_SOURCE_DIR}/source)

add_executable(advection_diffusion source/AdvectionDiffusionProblem.cc)
deal_ii_setup_target(advection_diffusion)
```

Then build:

```bash
git clone https://github.com/heena008/Advection-Diffusion_MsFEM.io.git
cd Advection-Diffusion_MsFEM.io
mkdir build && cd build
cmake -DDEAL_II_DIR=/path/to/dealii ..
make
```

This CMake file is a starting point based on the standard deal.II setup and has not been tested against this repository.

## Running

Run the program with MPI:

```bash
mpirun -np 4 ./advection_diffusion
```

When it starts, the program first runs the fine-scale solver on a coarse mesh, then on a finer mesh, and then runs the multiscale solver. The mesh refinement levels and the periodic-boundary option are set in `main` in `source/AdvectionDiffusionProblem.cc`:

```cpp
bool         is_periodic = true;
unsigned int n_refine    = 4;
```

## Parameters

Runtime output settings are read from `source/parameters.prm`:

| Parameter | Meaning |
|-----------|---------|
| `Output filename` | Name of the output file, without extension |
| `Output format` | Output format, for example `vtu` |
| `Output directory` | Folder where results are written |

The default `Output directory` in the file is an absolute path from the original author's computer. Change it to a folder that exists on your machine before running.

The coefficients (`A`, `b`, `f`), the initial condition, `theta`, and the time step are set in the header files in `include/`.

## Output

Results are written as `.vtu` files, one per time step, and can be opened with [ParaView](https://www.paraview.org/) or [VisIt](https://visit-dav.github.io/visit-website/).

## Background reading

- Biezemans, Le Bris, Legoll, Lozinski. *MsFEM for advection-dominated problems in heterogeneous media: Stabilization via nonconforming variants.* [arXiv:2406.13437](https://arxiv.org/abs/2406.13437)
- Le Bris, Legoll, Madiot. *Multiscale Finite Element methods for advection-dominated problems in perforated domains.* [arXiv:1710.09331](https://arxiv.org/abs/1710.09331)

## Contributing

Contributions are welcome. For anything beyond a small fix, please open an issue first to discuss the change.

## License

This project is licensed under the GNU General Public License v3.0. See the [LICENSE](LICENSE) file for details.

## Author

Heena ([@heena008](https://github.com/heena008))
