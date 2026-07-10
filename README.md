# Hamiltonian Simulation of a Ring of Two-Level Systems

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib as mpl
```

# Problem Statement and Hamiltonian Decomposition

We seek to simulate the time-evolution of $n$ 2-level sites (e.g. atoms with 2 energy levels, spins), each with energy gap $\hbar \omega$ in a ring with nearest-neighbor coupling $\hbar g$. For purposes of convenience, we will set $\hbar = 1$ throughout this process. If we set the reference energy to halfway between the ground and excited states for each site, then the Hamiltonian takes the following form:

$H = \sum_{l = 0}^{n - 1} \bigg(\frac{\omega}{2} (2 a_l^{\dag} a_l - 1) + g(a_l^{\dag} a_{l+1} _ a_l a_{l+1}^{\dag})$,

where $a_l^{(\dag)}$ is the annihilation (creation) operator for the site $l$. The first and second terms represent the self-energy of each site and the site-to-site coupling, respectively. Note that for $l = n-1$, the coupling with the site $l+1$ is actually with the $0^\mathrm{th}$ site, since the sites are organized in a ring. 
