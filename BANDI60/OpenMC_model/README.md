# OpenMC Reference Model and MGXS Generation

This directory contains the **OpenMC representation of the BANDI-60 core**, together with the workflow used to generate the **7-group macroscopic cross sections (MGXS)** consumed by the corresponding OpenMOC model.

The directory therefore has two related purposes:

1. Run the **continuous-energy OpenMC reference calculation** for the BANDI-60 geometry.
2. Collapse the OpenMC solution into a **7-group material-wise MGXS library** for OpenMOC.

It also contains the OpenMC axial fission-rate tally used to derive the **normalized axial power distribution** for comparison with OpenMOC and with the BANDI-60 results reported by Kim et al. (2024).

---

## Directory structure

```text
OpenMC_model/
├── log/
│    └── axial_power_openmc_dz*.csv
│
├── mgxs-run/
│    ├── geometry.xml
│    ├── materials.xml
│    ├── settings.xml 
│    ├── tallies.xml
│    └── summary.h5
│
├── calculate_material_mgxs.py
├── geometry.py
├── geometry.xml
├── materials.py
├── materials.xml
├── settings.xml
│
├── materials.h5
├── materials-hdf5.py
│
└── results_record.txt
```

The exact set of generated files can vary depending on the run history.

---

# Main workflow

The principal script is:

```text
calculate_material_mgxs.py
```

It performs the following sequence:

1. Loads the OpenMC model from:
   - `materials.xml`
   - `geometry.xml`
   - `settings.xml`

2. Defines the 7-group energy structure used for the OpenMOC model.

3. Creates a material-wise OpenMC MGXS library containing:
   - total cross section
   - scattering matrix
   - fission cross section
   - nu-fission cross section
   - fission spectrum, chi

4. Adds the MGXS tallies to the OpenMC calculation.

5. Adds an **axial fission-rate tally** over the active core for later calculation of the normalized axial power distribution.

6. Runs OpenMC.

7. Reads the final statepoint and extracts:
   - the MGXS data
   - the axial fission-rate profile

8. Writes the OpenMOC-compatible multigroup library:

```text
materials.h5
```

9. Writes a reproducible Python generator:

```text
materials-hdf5.py
```

which can recreate `materials.h5` from the extracted numerical arrays.

10. Writes the OpenMC axial-power information into:

```text
log/
```

11. Removes the temporary OpenMC run directory when the calculation completes successfully.

---

# Important source files

## `calculate_material_mgxs.py`

Main OpenMC execution and MGXS-generation script.

The current calculation uses a **7-group structure**:

```text
1.0e-5
6.35e-2
1.0e1
1.0e2
1.0e3
5.0e5
1.0e6
2.0e7 eV
```

The script generates material-wise macroscopic group constants suitable for loading into OpenMOC.

It also tallies the full-core axial fission-rate distribution over the 200 cm active height.

Current axial-power settings are:

```text
Active core: z = -100 cm to +100 cm
Axial bin width: 5 cm
Number of bins: 40
```

The resulting normalized axial power is calculated from the fission-rate distribution.

---

## `geometry.py`

Python definition of the OpenMC BANDI-60 geometry.

This should be treated as one of the primary source files for the physical model.

The corresponding:

```text
geometry.xml
```

is the OpenMC XML representation exported from the Python model.

---

## `materials.py`

Python definition of the OpenMC materials used in the BANDI-60 model.

The corresponding:

```text
materials.xml
```

contains the exported OpenMC material definitions.

---

## `settings.xml`

OpenMC run settings used by `calculate_material_mgxs.py`.

The current reference calculations are intended to reproduce the BANDI-60 beginning-of-cycle / hot-full-power configuration as closely as possible while maintaining consistency with the OpenMOC model.

---

# Generated MGXS files

## `materials.h5`

This is the **main interface between OpenMC and OpenMOC**.

It contains the 7-group material-wise macroscopic cross sections in the HDF5 structure expected by:

```python
openmoc.materialize.load_from_hdf5(...)
```

The OpenMOC model should use this file rather than independently defining multigroup cross sections.

The current HDF5 hierarchy is conceptually:

```text
/material/
    UO2/
        total
        scatter matrix
        fission
        nu-fission
        chi
    water/
        ...
    ...
```

A near-void material is also included for the external region because OpenMC represents that region as true void, whereas OpenMOC requires a material object.

---

## `materials-hdf5.py`

Auto-generated Python script containing the extracted MGXS arrays.

Its purpose is reproducibility: it can recreate `materials.h5` without repeating the expensive OpenMC Monte Carlo calculation.

---

# Axial power data

The OpenMC calculation includes an axial fission-rate tally over the active core.

The output is written to:

```text
log/
```

with a filename of the form:

```text
axial_power_openmc_dz5.csv
```

The CSV contains:

```text
z_lower_cm
z_upper_cm
z_mid_cm
core_height_cm
fission_rate
std_dev
normalized_axial_power
```

The normalized axial power is currently calculated from the fission-rate profile.

For equal-width axial bins:

```text
normalized axial power
    = local fission rate / mean fission rate
```

This produces a profile with an average value of approximately 1 over the active core.

The file is intended for use by a separate plotting / comparison script located at the top level of the `BANDI60` project.

---

# `log/`

Includes the axial fission-rate / normalized axial-power CSV.

```text
log/
├── axial_power_openmc_dz5.csv
└── ...
```

---

# `mgxs-run/`

Temporary working directory used by `calculate_material_mgxs.py`.

During an OpenMC MGXS calculation it can contain files such as:

```text
geometry.xml
materials.xml
settings.xml
tallies.xml
statepoint.*.h5
summary.h5
```

These files belong to the temporary OpenMC run used to generate the MGXS library.

If the calculation fails, retaining this directory is useful for debugging.

---

# `results_record.txt`

Manual record of OpenMC benchmark runs and timing results.

This file is useful for tracking:

- particle histories
- batches
- inactive batches
- CPU-thread count
- final `k_eff`
- Monte Carlo statistical uncertainty
- leakage
- runtime

---

# Running the OpenMC/MGXS calculation

From this directory, activate the OpenMC Python environment and run:

```bash
python calculate_material_mgxs.py
```

The script should:

1. construct the MGXS tallies;
2. execute OpenMC;
3. extract the final statepoint;
4. save the axial-power CSV in `log/`;
5. generate `materials.h5`;
6. generate `materials-hdf5.py`;

---

# Relationship to the OpenMOC model

The corresponding OpenMOC model is stored separately in:

```text
BANDI60/
└── OpenMOC_model/
```

The basic workflow is:

```text
OpenMC continuous-energy model
        │
        ├── reference k_eff
        ├── reference axial fission-rate distribution
        │
        └── material-wise MGXS collapse
                    │
                    ▼
              materials.h5
                    │
                    ▼
              OpenMOC model
                    │
                    ├── deterministic k_eff
                    └── FSR fission-rate distribution
```

This separation is important for the thesis:

- OpenMC provides the continuous-energy Monte Carlo reference calculation.
- OpenMOC solves the corresponding reduced multigroup deterministic transport problem.
- `materials.h5` is the bridge between the two models.

---
