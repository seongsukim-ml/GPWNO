# Getting Atom Information

By Chaoran Cheng, Oct 1, 2023

This folder contains the atom information for the datasets.

Some baselines like CNN and GKN need atom information for building the initial input feature function. To let these models capture the different atom types, we applied atom-specific Gaussian parameters. The atom information are store in the JSON file with fields of `name` (chemical symbol), `atom_num` (atomic number) and `radius` (covalent radius). The information can be obtained with `RDKit` package with the following example code:

```python
from rdkit import Chem

pt = Chem.GetPeriodicTable()
atoms = ['C', 'H', 'N', 'O', 'F']
atom_info = [
    {
        'name': a,
        'atom_num': pt.GetAtomicNumber(a),
        'radius': round(pt.GetRcovalent(a) / 0.529177, 5)  # convert to Bohr
    } for a in atoms
]
```
