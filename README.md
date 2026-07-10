# Hamiltonian Simulation of a Ring of Two-Level Systems

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib as mpl
```

## Problem Statement and Hamiltonian Decomposition

We seek to simulate the time-evolution of $n$ 2-level sites (e.g. atoms with 2 energy levels, spins), each with energy gap $\hbar \omega$ in a ring with nearest-neighbor coupling $\hbar g$. For purposes of convenience, we will set $\hbar = 1$ throughout this process. If we set the reference energy to halfway between the ground and excited states for each site, then the Hamiltonian takes the following form:

$$
H = \sum_{l = 0}^{n - 1} \bigg(\frac{\omega}{2} \Big(2 a_l^{\dagger} a_l - 1\Big) + g\Big(a_l^{\dagger} a_{l+1} + a_l a_{l+1}^{\dagger}\Big)\bigg),
$$

where $a_l^{(\dagger)}$ is the annihilation (creation) operator for the site $l$. The first and second terms represent the self-energy of each site and the site-to-site coupling, respectively. Note that for $l = n-1$, the coupling with the site $l+1$ is actually with the $0^\mathrm{th}$ site, since the sites are organized in a ring. 

We will represent the system using $n$ data bits, with the $l^\mathrm{th}$ bit (counting from right to left and zero-indexing) representing the state (ground = 0, excited = 1) of site $l$. For our example, we use $n = 8$:

```python
n = 8
```

We also add m+3 ancillary bits, where $m = \lceil \log_2(n) \rceil$. The reasoning for this will be explained in the next section:

```python
m = int(np.ceil(np.log(n)/np.log(2)))
qc = QuantumCircuit(n+m+3)
```

The ancillary bits will be used as controls in manipulating the data bits as we apply the gates sequentially. For our physical system, we start with sites 0 and 4 excited, leading us to prepare our input state by applying X gates to data bits 0 and 4:

```python
qc.x(0)
qc.x(4)
```

## LCU: Block-Encoding the Hamiltonian

The key step in the simulation process will be to block-encode the Hamiltonian into a unitary operator, which will require us to decompose the Hamiltonian into a linear combination of unitaries (LCU). To that end, the nature of $H$ lends itself to decomposition into a linear combination of Pauli strings:

$$
H = \frac{1}{2} \sum_{l = 0}^{n - 1} \Big(Z_l + X_l X_{l+1} + Y_l Y_{l+1}\Big).
$$

As this expression shows, $H$ is composed of $3n$ Pauli strings: $n$ strings of each of 3 categories: $Z_l$, $X_l X_{l+1}$, and $Y_l Y_{l+1}$. The amplitudes (square-root of the coefficients) corresponding to these string categories are proportional to $\sqrt{\omega}$, $\sqrt{g}$, and $\sqrt{g}$, respectively. We use the rightmost 2 ancillary bits to represent the string categories, with $\ket{00} \leftrightarrow Z$, $\ket{01} \leftrightarrow X$, and $\ket{11} \leftrightarrow Y$. To encode the string category amplitudes, we apply the rotation operator $R_Y(2\theta)$, where $\theta = \cos^{-1}(\sqrt{\omega}/\sqrt{\omega + 2g})$ on the initial 2-ancillary-bit state $\ket{00}$. 
