# EOSNet: Embedded Overlap Structures for Graph Neural Networks

<p align="center">
  <img src="https://raw.githubusercontent.com/Rutgers-ZRG/EosNet/master/EOSnet_TOC.png" width="65%" alt="EOSNet TOC">
</p>

**Note**: This EOSNet package is inherited from the [CGCNN](https://github.com/txie-93/cgcnn) framework, and there are some major changes.

## Change log

- **Extended to support tensor properties prediction** (elasticity tensors, dielectric tensors, etc.)
  - Added `--out-dim` argument to specify number of output components (1 for scalar, 3 for vector, 6/21/36 for tensors)
  - Introduced `TensorTargetData` class to handle multi-column CSV format for tensor properties
  - Updated model architecture to output configurable dimensions
  - Updated `Normalizer` to handle component-wise normalization for tensor targets
- Using atomic-centered Gaussian Overlap Matrix (GOM) Fingerprint vectors as atomic features
- Switch reading pymatgen structures from CIF to POSCAR
- Add `drop_last` option in `get_train_val_test_loader`
- Take data imbalance into account for classification job
- Clip `lfp` (Long FP) and `sfp` (Contracted FP) length for arbitrary crystal structures
- Add MPS support to accelerate training on MacOS, for details see [PyTorch MPS Backend](https://pytorch.org/docs/stable/notes/mps.html) and [Apple Metal acceleration](https://developer.apple.com/metal/pytorch/) \
  **Note**: For classification jobs you may need to modify [line 227 in WeightedRandomSampler](https://github.com/pytorch/pytorch/blob/main/torch/utils/data/sampler.py#L227) to `weights_tensor = torch.as_tensor(weights, dtype=torch.float32 if weights.device.type == "mps" else torch.float64)` when using MPS backend. To maximize the efficiency of training while using MPS backend, you may want to use only single core (`--workers 0`) of the CPU to load the dataset.
- Add `save_to_disk` option, `disable_mps` option and FP related arg options
- Introduce `IdTargetData` class to do efficient sampling on the dataset
- Update `collate_pool` to handle both `IdTargetData` and `StructData` type dataset
- Complete refrom the `StructData` with batch loading and processing, and add `dataset.clear_cache()` to release memory
- Move instancing of `StructData` to `tain()` and `validate()` seprately instead of in `main()`
- Use `IdTargetData` for `get_train_val_test_loader()`, and get `struct_data` from `StructData` by batches
- Saving the `processed_data` to multiple `npz` files under `saved_npz_files` directory instead of one big file
- Save both `train_results.csv` and `test_results.csv` at the end of training
- Switched to [libfp](https://github.com/Rutgers-ZRG/libfp) (C implementation with Python interface) for faster fingerprint calculation

This package is based on the [Crystal Graph Convolutional Neural Networks]((https://link.aps.org/doi/10.1103/PhysRevLett.120.145301)) that takes an arbitary crystal structure to predict material properties. 

The package provides two major functions:

- Train a EOSNet model with a customized dataset.
- Predict material properties of new crystals with a pre-trained EOSNet model.

##  Dependencies

This package requires:

- [libfp](https://github.com/Rutgers-ZRG/libfp) - C implementation of the fingerprint library with Python interface
- [PyTorch](http://pytorch.org)
- [scikit-learn](http://scikit-learn.org/stable/)
- [pymatgen](http://pymatgen.org)
- [ASE](https://wiki.fysik.dtu.dk/ase/)

If you are new to Python, please use [conda](https://conda.io/docs/index.html) to manage Python packages and environments.

### Installation Steps

1. **Create and activate a conda environment:**
```bash
conda create -n eosnet python=3.10 pip
conda activate eosnet
python3 -m pip install -U pip setuptools wheel
```

2. **Install libfp (fingerprint library):**

   **Option A: Install from PyPI (recommended):**
   ```bash
   pip install libfp
   ```

   **Option B: Install from source:**
   ```bash
   git clone https://github.com/Rutgers-ZRG/libfp.git
   cd libfp
   pip install .
   cd ..
   ```

   **Prerequisites for libfp:**
   - C compiler (gcc, clang, etc.)
   - Python development headers
   - LAPACK/OpenBLAS/MKL

   If you encounter LAPACK-related issues, install OpenBLAS:
   ```bash
   # Using conda
   conda install -c conda-forge openblas
   
   # Using Homebrew (macOS)
   brew install openblas
   ```

3. **Install remaining dependencies:**
```bash
python3 -m pip install numpy>=1.21.4 scipy>=1.8.0 ase==3.22.1
python3 -m pip install scikit-learn torch==2.2.2 torchvision==0.17.2 pymatgen==2024.3.1
```

The above environment has been tested stable for both M-series macOS and Linux clusters.

## Check your strcuture files before use EOSNet

To catch the erroneous POSCAR file you can use the following `check_fp.py` in the `root_dir`:
```python

#!/usr/bin/env python3

import os
import sys
import numpy as np
from functools import reduce
import libfp
from ase.io import read as ase_read

def get_ixyz(lat, cutoff):
    lat = np.ascontiguousarray(lat)
    lat2 = np.dot(lat, np.transpose(lat))
    vec = np.linalg.eigvals(lat2)
    ixyz = int(np.sqrt(1.0/max(vec))*cutoff) + 1
    ixyz = np.int32(ixyz)
    return ixyz

def check_n_sphere(rxyz, lat, cutoff, natx):
    
    ixyzf = get_ixyz(lat, cutoff)
    ixyz = int(ixyzf) + 1
    nat = len(rxyz)
    cutoff2 = cutoff**2

    for iat in range(nat):
        xi, yi, zi = rxyz[iat]
        n_sphere = 0
        for jat in range(nat):
            for ix in range(-ixyz, ixyz+1):
                for iy in range(-ixyz, ixyz+1):
                    for iz in range(-ixyz, ixyz+1):
                        xj = rxyz[jat][0] + ix*lat[0][0] + iy*lat[1][0] + iz*lat[2][0]
                        yj = rxyz[jat][1] + ix*lat[0][1] + iy*lat[1][1] + iz*lat[2][1]
                        zj = rxyz[jat][2] + ix*lat[0][2] + iy*lat[1][2] + iz*lat[2][2]
                        d2 = (xj-xi)**2 + (yj-yi)**2 + (zj-zi)**2
                        if d2 <= cutoff2:
                            n_sphere += 1
                            if n_sphere > natx:
                                raise ValueError()


def read_types(cell_file):
    buff = []
    with open(cell_file) as f:
        for line in f:
            buff.append(line.split())
    try:
        typt = np.array(buff[5], int)
    except:
        del(buff[5])
        typt = np.array(buff[5], int)
    types = []
    for i in range(len(typt)):
        types += [i+1]*typt[i]
    types = np.array(types, int)
    return types

if __name__ == "__main__":
    current_dir = './'
    for filename in os.listdir(current_dir):
        f = os.path.join(current_dir, filename)
        if os.path.isfile(f) and os.path.splitext(f)[-1].lower() == '.vasp':
            atoms = ase_read(f)
            lat = atoms.cell[:]
            rxyz = atoms.get_positions()
            chem_nums = list(atoms.numbers)
            znucl_list = reduce(lambda re, x: re+[x] if x not in re else re, chem_nums, [])
            typ = len(znucl_list)
            znucl = np.array(znucl_list, int)
            types = read_types(f)
            cell = (lat, rxyz, types, znucl)

            natx = int(256)
            lmax = int(0)
            cutoff = np.float64(int(np.sqrt(8.0))*3) # Shorter cutoff for GOM
            
            try:
                check_n_sphere(rxyz, lat, cutoff, natx)
            except ValueError:
                print(str(filename) + " is glitchy !")
            
            if len(rxyz) != len(types) or len(set(types)) != len(znucl):
                print(str(filename) + " is glitchy !")
            else:
                fp = libfp.get_lfp(cell, cutoff=cutoff, natx=natx, log=False) # Long Fingerprint
                # fp = libfp.get_sfp(cell, cutoff=cutoff, natx=natx, log=False)   # Contracted Fingerprint         
```

## Usage

### Define a customized dataset 

To input crystal structures to EOSNet, you will need to define a customized dataset. Note that this is required for both training and predicting. 

Before defining a customized dataset, you will need:

- [POSCAR](https://www.vasp.at/wiki/index.php/POSCAR) files recording the structure of the crystals that you are interested in
- The target properties for each crystal (not needed for predicting, but you need to put some random numbers in `id_prop.csv`)

You can create a customized dataset by creating a directory `root_dir` with the following files: 

1. `id_prop.csv`: a [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) file with two columns. The first column recodes a unique `ID` for each crystal, and the second column recodes the value of target property. If you want to predict material properties with `predict.py`, you can put any number in the second column. (The second column is still needed.)

2. `ID.vasp` a [POSCAR](https://www.vasp.at/wiki/index.php/POSCAR) file that recodes the crystal structure, where `ID` is the unique `ID` for the crystal.

The structure of the `root_dir` should be:

```
root_dir
├── id_prop.csv
├── atom_init.json
├── id0.vasp
├── id1.vasp
├── ...
```

### Tensor Properties Support

EOSNet now supports predicting crystal-level tensor properties (e.g., elasticity tensors, dielectric tensors) in addition to scalar properties.

#### Data Format for Tensor Properties

For tensor properties, the `id_prop.csv` file should contain multiple columns:

```csv
struct_id, component_1, component_2, ..., component_n
mp-001, 142.81, 54.37, 47.75, 0.0, 0.0, 0.0, ...
mp-002, 215.33, 89.12, 78.44, 0.5, -0.2, 0.1, ...
```

**Common tensor types and their dimensions:**

| Target Type | `out_dim` | Format | Example |
|-------------|-----------|--------|---------|
| Scalar (energy, band gap) | 1 | `id,val` | Formation energy |
| Vector (polarization) | 3 | `id,Px,Py,Pz` | Electric polarization |
| 2nd-rank symmetric (dielectric, stress) | 6 | `id,xx,yy,zz,yz,xz,xy` | Voigt notation |
| 4th-rank elasticity (symmetric) | 21 | `id,C11,C12,...,C66` | Independent components |
| 4th-rank elasticity (full) | 36 | `id,C11,...,C66` | Full 6×6 matrix |

For elasticity tensors, the 21 independent components in Voigt notation are stored as:
```
C11, C12, C13, C14, C15, C16,
     C22, C23, C24, C25, C26,
          C33, C34, C35, C36,
               C44, C45, C46,
                    C55, C56,
                         C66
```

### Train a GNN model

Before training a new GNN model, you will need to:

- Define a customized dataset at `root_dir` to store the structure-property relations of interest.

Then, in directory `EOSNet`, you can train a GNN model for your customized dataset by:

```bash
python3 train.py root_dir
```

For detailed info of setting tags you can run

```bash
python3 train.py -h
```

**Scalar properties:**
```bash
python3 train.py root_dir --save_to_disk true --disable-mps --task regression --workers 7 --epochs 500 --batch-size 64 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --n-conv 3 --n-h 1 --lr 1e-3 --warmup-epochs 20 --lr-milestones 100 200 400 --weight-decay 0 | tee EOSNet_log.txt
```

**Tensor properties (e.g., elasticity):**
```bash
python3 train.py ./mp_elast_2000 --out-dim 21 --task regression \
  --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
  --epochs 500 --batch-size 64 --lr 1e-3 --workers 0 --disable-mps \
  --save_to_disk true | tee elasticity_log.txt
```

The `--out-dim` argument specifies the number of output components:
- `--out-dim 1`: Scalar (default)
- `--out-dim 3`: Vector properties
- `--out-dim 6`: Symmetric 2nd-rank tensor (Voigt)
- `--out-dim 21`: Elasticity tensor (21 independent components)
- `--out-dim 36`: Full 6×6 elasticity tensor

To resume from a previous `checkpoint`

```bash
python3 train.py root_dir --save_to_disk false --disable-mps --resume ./checkpoint.pth.tar --task regression --workers 7 --epochs 500 --batch-size 64 --optim 'Adam' --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 --n-conv 3 --n-h 1 --lr 1e-3 --warmup-epochs 20 --lr-milestones 100 200 400 --weight-decay 0 >> EOSNet_log.txt
```

After training, you will get three files in `EOSNet` directory.

- `model_best.pth.tar`: stores the GNN model with the best validation accuracy.
- `checkpoint.pth.tar`: stores the GNN model at the last epoch.
- `test_results.csv`: stores the `ID`, target value, and predicted value for each crystal in test set.

### Predict material properties with a pre-trained GNN model

In directory `EOSNet`, you can predict the properties of the crystals in `root_dir`:

```bash
python predict.py pre-trained.pth.tar --save_to_disk false --test root_dir
```

**Note**: you need to put some random numbers in `id_prop.csv` and the `struct_id`s are the structures you want to predict.

For tensor properties, the pre-trained model automatically detects the output dimension from its saved configuration. The output CSV will contain all predicted components:
```csv
id,target,prediction
mp-001,142.81 54.37 47.75 ...,145.23 52.11 49.34 ...
```

## Performance Notes

### Typical Performance for Elasticity Prediction

When training on Materials Project elasticity data (~2000 structures):
- **Training time**: ~5-10 minutes per epoch (M-series Mac with MPS disabled, 0 workers)
- **Typical MAE**: 30-40 GPa per component (good performance)
- **Excellent MAE**: < 20 GPa
- **Acceptable MAE**: 40-60 GPa

The MAE is reported in the physical units (GPa for elasticity) while the loss is computed on normalized data for stable training.

### Segmentation Fault Issues on macOS

If you encounter segmentation faults when processing structures (especially on macOS), this is typically caused by **conflicting OpenMP runtimes** between conda-forge's OpenBLAS and Apple's system libraries.

**Root Cause:** The default conda-forge OpenBLAS uses OpenMP threading (`openmp_` build), which conflicts with Apple's libiomp5 and causes segfaults in `libfp`'s C extensions.

**Recommended Solution:** Switch to pthread-based OpenBLAS

1. **Replace OpenMP OpenBLAS with pthread build:**
   ```bash
   conda activate fplibenv  # or your environment name
   conda install "libopenblas=*=*pthread*" -c conda-forge --force-reinstall
   ```

2. **Verify the installation:**
   ```bash
   conda list | grep libopenblas
   # Should show: libopenblas  X.X.XX  pthreads_hXXXXXXX_X
   ```

3. **Reinstall libfp to link against new pthread OpenBLAS:**
   ```bash
   pip uninstall libfp -y
   pip install libfp --no-cache-dir
   ```


## How to cite

Please cite the our newly published work if you use EOSNet in your research:

```
@article{taoEOSnetEmbeddedOverlap2025,
  author = {Tao, Shuo and Zhu, Li},
  title = {EOSnet: Embedded Overlap Structures for Graph Neural Networks in Predicting Material Properties},
  journal = {J. Phys. Chem. Lett.},
  volume = {16},
  number = {XXX},
  pages = {717-724},
  year = {2025},
  doi = {10.1021/acs.jpclett.4c03179},
  URL = { https://doi.org/10.1021/acs.jpclett.4c03179},
  eprint = { https://doi.org/10.1021/acs.jpclett.4c03179}
}
```

For CGCNN framework, please cite:
```
@article{PhysRevLett.120.145301,
  title = {Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties},
  author = {Xie, Tian and Grossman, Jeffrey C.},
  journal = {Phys. Rev. Lett.},
  volume = {120},
  issue = {14},
  pages = {145301},
  numpages = {6},
  year = {2018},
  month = {Apr},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevLett.120.145301},
  url = {https://link.aps.org/doi/10.1103/PhysRevLett.120.145301}
}
```

If you use the [libfp library](https://github.com/Rutgers-ZRG/libfp), please cite:
```
@article{taoEOSnetEmbeddedOverlap2025,
  title = {EOSnet: Embedded Overlap Structures for Graph Neural Networks in Predicting Material Properties},
  author = {Tao, Shuo and Zhu, Li},
  journal = {J. Phys. Chem. Lett.},
  volume = {16},
  pages = {717-724},
  year = {2025},
  doi = {10.1021/acs.jpclett.4c03179},
  url = {https://doi.org/10.1021/acs.jpclett.4c03179}
}
```

And the original fingerprint algorithm paper:
```
@article{zhuFingerprintBasedMetric2016,
  title = {A Fingerprint Based Metric for Measuring Similarities of Crystalline Structures},
  author = {Zhu, Li and Amsler, Maximilian and Fuhrer, Tobias and Schaefer, Bastian and Faraji, Somayeh and Rostami, Samare and Ghasemi, S. Alireza and Sadeghi, Ali and Grauzinyte, Migle and Wolverton, Chris and Goedecker, Stefan},
  year = {2016},
  month = jan,
  journal = {The Journal of Chemical Physics},
  volume = {144},
  number = {3},
  pages = {034203},
  doi = {10.1063/1.4940026},
  url = {https://doi.org/10.1063/1.4940026}
}
```

For GOM Fingerprint methodology, please cite:
```
@article{sadeghiMetricsMeasuringDistances2013,
  title = {Metrics for Measuring Distances in Configuration Spaces},
  author = {Sadeghi, Ali and Ghasemi, S. Alireza and Schaefer, Bastian and Mohr, Stephan and Lill, Markus A. and Goedecker, Stefan},
  year = {2013},
  month = nov,
  journal = {The Journal of Chemical Physics},
  volume = {139},
  number = {18},
  pages = {184118},
  doi = {10.1063/1.4828704},
  url = {https://pubs.aip.org/aip/jcp/article/317391}
}
```

