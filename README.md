# Master Thesis – OpenMOC / BANDI-60 Neutronics

This repository contains the neutronics work developed for my Master's thesis in Computational Science: Physics. The main objective is to build deterministic neutron-transport models that can later be used to generate data for reduced-order / machine-learning models and digital-twin applications.

The deterministic solver used here is [OpenMOC](https://github.com/mit-crpg/OpenMOC), based on the Method of Characteristics (MOC). OpenMC is also used to generate multigroup cross sections for the BANDI-60 model.

## Repository structure

```text
Master_Thesis/
├── BANDI60/
│   ├── geometry.py
│   ├── materials-hdf5.py
│   ├── materials.h5
│   ├── multi-group_cross-sections/
│   └── plots/
└── sample-input/
    ├── pin-cell/
    └── fixed-source-pin-cell/
```

## BANDI-60 OpenMOC model

`BANDI60/geometry.py` contains the current OpenMOC model of the Pyrex-loaded BANDI-60 core, based mainly on Kim et al. (2020, 2022, 2024) and on a previous OpenMC implementation.

### Multigroup cross sections

The folder

```text
BANDI60/multi-group_cross-sections/
```

contains the OpenMC workflow used to calculate material-wise multigroup macroscopic cross sections.

`calculate_material_mgxs.py`:

1. Loads the OpenMC BANDI-60 geometry and materials.
2. Creates material-wise MGXS tallies.
3. Runs OpenMC.
4. Extracts total, scattering, fission, nu-fission and chi data.
5. Writes copy/paste-ready arrays to `mgxs_arrays.txt`.

The current model uses a provisional 7-group structure compatible with the OpenMOC/C5G7-style HDF5 workflow.

`materials-hdf5.py` builds `materials.h5`, which is loaded by the OpenMOC BANDI-60 geometry.

### Running the BANDI-60 geometry

From the repository environment with OpenMOC installed:

```bash
cd BANDI60
python materials-hdf5.py
python geometry.py
```

Generated geometry plots are written to:

```text
BANDI60/plots/
```

## Earlier OpenMOC work

`sample-input/pin-cell/` contains introductory OpenMOC pin-cell and track/flux studies.

`sample-input/fixed-source-pin-cell/` contains fixed-source transport experiments in which spatially varying Gaussian Random Field (GRF) source distributions are generated and solved with OpenMOC. The saved source/flux pairs are part of the preliminary workflow for later machine-learning training-data generation.

