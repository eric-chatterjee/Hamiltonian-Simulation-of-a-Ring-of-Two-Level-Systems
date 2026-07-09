# Hamiltonian Simulation of a Ring of Two-Level Systems

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib as mpl
```

# Goal

We seek to simulate the time-evolution of $n$ 2-level sites (e.g. atoms with 2 energy levels, spins), each with energy $\hbar \omega$ in a ring with nearest-neighbor coupling $\hbar g$.
