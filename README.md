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

<img width="191" height="947" alt="InitialState" src="https://github.com/user-attachments/assets/94a089fe-5d86-4966-988c-eb3a05477913" />

$\ket{000000}\ket{00010001}$

## LCU: Block-Encoding the Hamiltonian

The key step in the simulation process will be to block-encode the Hamiltonian into a unitary operator, which will require us to decompose the Hamiltonian into a linear combination of unitaries (LCU). To that end, the nature of $H$ lends itself to decomposition into a linear combination of Pauli strings:

$$
H = \frac{1}{2} \sum_{l = 0}^{n - 1} \Big(Z_l + X_l X_{l+1} + Y_l Y_{l+1}\Big).
$$

As this expression shows, $H$ is composed of $3n$ Pauli strings: $n$ strings of each of 3 categories: $Z_l$, $X_l X_{l+1}$, and $Y_l Y_{l+1}$. This form lends itself to block-encoding the Hamiltonian as $H = (PREP^{\dagger}) (SELECT) (PREP)$, where PREP converts the ancilla to the basis of desired strings and SELECT is a block-diagonal unitary that applies the desired string based on the ancilla values.

### Applying the PREP Gate

The amplitudes (square-root of the coefficients) corresponding to the string categories are proportional to $\sqrt{\omega}$, $\sqrt{g}$, and $\sqrt{g}$, respectively. We use the rightmost 2 ancillary bits to represent the string categories, with $\ket{00} \leftrightarrow Z$, $\ket{01} \leftrightarrow X$, and $\ket{11} \leftrightarrow Y$. If we just consider the rightmost bit (i.e., bit $n$), we thus need an amplitude proportional to $\sqrt{\omega}$ and $\sqrt{2g}$ for the states $\ket{0} and $\ket{1}, respectively. This is encoded by applying the rotation operator $R_Y(2\theta)$, where $\theta = \cos^{-1}(\sqrt{\omega}/\sqrt{\omega + 2g})$, on the rightmost bit. For our case, we define $\omega = 10^{10} \textrm{ s}^{-1}$ and $g = 5 \times 10^9 \textrm{ s}^{-1}$:

```python
w = 10**10 # self-energy of each site
g = 5*10**9 # site-site coupling

theta = np.arccos(np.sqrt(w/(w + 2*g))) # desired rotation angle for rightmost bit

qc.ry(2*theta,n)
```

<img width="191" height="947" alt="AfterFirstRotationGate" src="https://github.com/user-attachments/assets/cd6dcc60-953d-474c-bfb5-bb9d85b2b459" />

$\frac{1}{\sqrt{2}} \ket{000000}\ket{00010001} + \frac{1}{\sqrt{2}} \ket{000001}\ket{00010001}$

We have thus encoded the composite amplitude of the Z strings with the state $\ket{0}$ on the rightmost bit, while the composite amplitude of the X and Y strings are encoded with the state $\ket{1}$. To separate the X (i.e., $\ket{01}$) and Y (i.e., $\ket{11}$) strings from each other, we use the fact that the X and Y strings feature identical amplitudes to apply a Hadamard on the second-to-rightmost ancillary bit if the rightmost ancillary bit is 1 (in other words, a control-Hadamard on bit $n+1$ conditioned on bit $n$):

```python
qc.ch(n,n+1)
```

<img width="255" height="947" alt="AfterFirstHadamardGate" src="https://github.com/user-attachments/assets/560100d4-58c0-4a38-a99f-f50dbe659ba4" />

$\frac{1}{\sqrt{2}} \ket{000000}\ket{00010001} + \frac{1}{2} \ket{000001}\ket{00010001} + \frac{1}{2} \ket{000011}\ket{00010001}$

The next step is to establish the individual strings for each string category. Here, we need enough ancillary bits to address every site in the physical ring. This explains why we defined $m = \lceil \log_2(n) \rceil$ above, with bits $n+2$ through $n+1+m$ serving to address individual data bits (corresponding to the individual sites). Since all $n$ strings for each given category feature the same amplitude, we apply Hadamard gates to this $m$-bit string:

```python
qc.h(range(n+2,n+m+2))
```

<img width="255" height="947" alt="AfterSecondHadamardGate" src="https://github.com/user-attachments/assets/05ab0c5d-4bfb-41f0-95f2-476e1f151af2" />

$\frac{1}{4} \ket{000000}\ket{00010001} + \frac{1}{4\sqrt{2}} \ket{000001}\ket{00010001} + \frac{1}{4\sqrt{2}} \ket{000011}\ket{00010001} + \frac{1}{4} \ket{000100}\ket{00010001} + \frac{1}{4\sqrt{2}} \ket{000101}\ket{00010001} + \frac{1}{4\sqrt{2}} \ket{000111}\ket{00010001} + ... + \frac{1}{4\sqrt{2}} \ket{011001}\ket{00010001} + \frac{1}{4\sqrt{2}} \ket{011011}\ket{00010001} + \frac{1}{4} \ket{011100}\ket{00010001} + \frac{1}{4\sqrt{2}} \ket{011101}\ket{00010001} + \frac{1}{4\sqrt{2}} \ket{011111}\ket{00010001}$

Note that we have so far only operated on the ancillas, with the state encoded by the data bits staying intact. The set of sub-processes above (specifically, rotation on the rightmost ancillary bit, control-Hadamard on the next ancillary bit conditioned on the rightmost ancillary bit, and Hadamard on the next m ancilla) thus induce the following mapping:

$\ket{0} \otimes \ket{0}^{\otimes m} \otimes \ket{\psi} \rightarrow \frac{1}{\sqrt{2^m (\omega+ 2g)}} \ket{0} \otimes \Big(\sum_{l = 0}^{n-1} \ket{l}\Big) \otimes (\sqrt{\omega} \ket{00} + \sqrt{g} \ket{01} + \sqrt{g} \ket{11}) \otimes \ket{\psi}$,

where $\ket{l}$ is the $m$-bit address for the site $l$.

### Applying the SELECT Gate

Now that each ancillary bit combination represents a unique string, we are ready to use the SELECT gate. Broadly, this consists of applying the individual strings on the data-bit string $\ket{\psi}$, with the exact string being determined by the ancilla. 
