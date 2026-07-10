# Hamiltonian Simulation of a Ring of Two-Level Systems

```python
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib as mpl
import scipy as sp
```

## Problem Statement and Hamiltonian Form

We seek to simulate the time-evolution of $n$ 2-level sites (e.g. atoms with 2 energy levels, spins), each with energy gap $\hbar \omega$ in a ring with nearest-neighbor coupling $\hbar g$. For purposes of convenience, we will set $\hbar = 1$ throughout this process. If we set the reference energy to halfway between the ground and excited states for each site, then the Hamiltonian takes the following form:

$$
H = \sum_{l = 0}^{n - 1} \bigg(\frac{\omega}{2} \Big(2 a_l^{\dagger} a_l - 1\Big) + g\Big(a_l^{\dagger} a_{l+1} + a_l a_{l+1}^{\dagger}\Big)\bigg),
$$

where $a_l^{(\dagger)}$ is the annihilation (creation) operator for the site $l$. The first and second terms represent the self-energy of each site and the site-to-site coupling, respectively. Note that for $l = n-1$, the coupling with the site $l+1$ is actually with the $0^\mathrm{th}$ site, since the sites are organized in a ring. 

## Establishing the Qubit Register

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

As this expression shows, $H$ is composed of $3n$ Pauli strings: $n$ strings of each of 3 categories: $Z_l$, $X_l X_{l+1}$, and $Y_l Y_{l+1}$. This form lends itself to block-encoding the Hamiltonian as the following:

$$
\ket{0}_\mathrm{ancilla} \bra{0}_\mathrm{ancilla} H/\alpha \ket{\psi} = \bra{0}_\mathrm{ancilla} (\mathrm{PREP})^{\dagger} (\mathrm{SELECT}) (\mathrm{PREP}) \ket{0}_\mathrm{ancilla} \ket{0}_\mathrm{ancilla} \ket{\psi},
$$

where PREP converts the ancilla to the basis of desired strings, SELECT is a block-diagonal unitary that applies the desired string based on the ancilla values, and $\alpha$ is a scaling factor that keeps all elements of $H/\alpha$ below 1 (which will directly result from establishing the PREP gate, as will be shown below).

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

We have thus encoded the composite amplitude of the $Z$ strings with the state $\ket{0}$ on the rightmost bit, while the composite amplitude of the $X$ and $Y$ strings are encoded with the state $\ket{1}$. To separate the $X$ (i.e., $\ket{01}$) and $Y$ (i.e., $\ket{11}$) strings from each other, we use the fact that the $X$ and $Y$ strings feature identical amplitudes to apply a Hadamard on the second-to-rightmost ancillary bit if the rightmost ancillary bit is 1 (in other words, a control-Hadamard on bit $n+1$ conditioned on bit $n$):

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

$\ket{0} \otimes \ket{0}^{\otimes m} \otimes \ket{\psi} \rightarrow \frac{1}{\sqrt{2^m (\omega + 2g)}} \ket{0} \otimes \Big(\sum_{l = 0}^{n-1} \ket{l}\Big) \otimes (\sqrt{\omega} \ket{00} + \sqrt{g} \ket{01} + \sqrt{g} \ket{11}) \otimes \ket{\psi}$,

where $\ket{l}$ is the $m$-bit address for the site $l$. Note that the amplitude-squared of each $Z$ Pauli string is $\omega/(2^m (\omega + 2g))$, while the amplitude of each $X$ or $Y$ string is $g/(2^m (\omega + 2g))$. On the other hand, recall that the respective LCU coefficients in the Hamiltonian are $\omega/2$ and $\gamma/2$. As a result, the above-discussed attenuation coefficient in the block-encoding becomes $\alpha = 2^{m-1} (\omega + 2g)$.

### Applying the SELECT Gate

Since each ancillary bit combination now represents a unique string, we are ready to use the SELECT gate. Broadly, this consists of applying the individual strings on the data-bit string $\ket{\psi}$, with the exact string being determined by the ancilla. Here, we note that the leftmost ancillary bit is used as a placeholder for rotation operators that will be applied later during the Generalized Quantum Signal Processing (GQSP) steps. This finishes the explanation of why we use $m+3$ ancillary bits overall. In preparation for GQSP, we need to use this ancillary bit as an anti-control for the block-encoding process (i.e., the SELECT gate is only applied if the leftmost ancillary bit input is 0). The method we use is to flip this bit and use it as a regular control bit in all SELECT operations:

```python
qc.x(n+m+2)
```

The strategy that we will use is to individually set each combination for the $m$-bit ancillary string addressing the physical site (i.e., bits $n+2$ through $n+m+1$) and the 2-bit ancillary string addressing the string type (i.e., bits $n$ and $n+1$) to all 1s, followed by applying the corresponding string to the data bits in a multi-controlled manner conditioned on all ancillary bits. To do this, we run a for loop over all possible combinations in the $m$-bit string, applying an $X$ gate to the $k^\mathrm{th}$ bit in that string (where we are zero-indexing and counting from right to left) every $2^k$ runs (since the $k^\mathrm{th}$ binary digit switches every $2^k$ counts). For a given run corresponding to site $l$, we first apply multi-controlled $Y$ gates to sites $l$ and $l+1$ (since the rightmost 2 ancilla being in the all-1 state corresponds to the $Y$ string type), then we apply the $X$ gate to flip the second-to-rightmost ancillary bit (i.e., bit $n+1$) and apply the multi-controlled $X$ gates to sites $l$ and $l+1$, and finally we apply the $X$ gate to flip the rightmost ancillary bit (i.e., bit $n$) and apply the multi-controlled $Z$ gates to site $l$. Note that we separate out the case $l = n$ from the main for loop, since $l+1 = 0$ in that case, not $n+1$:

```python
from qiskit.circuit.library import MCXGate
from qiskit.circuit.library import YGate
from qiskit.circuit.library import ZGate

mcy_gate = YGate().control(m+3)
mcz_gate = ZGate().control(m+3)

control_qubits = range(n,n+m+3)

for site in range(n-1): # This is not n, because we separately consider the (n-1)^st site and its coupling with the 0^th site
    
    for k in range(m):
        if np.mod(site,2**k) == 0:
            qc.x(n+2+k) # Maps m-bit ancillas marking the site to all 1s
    
    qc.append(mcy_gate,list(control_qubits) + [site]) # For the rightmost 2 ancillas, 11 represents the Y Pauli string category
    qc.append(mcy_gate,list(control_qubits) + [site+1])
    
    qc.x(n+1) # Mapping 01 (corresponding to the X Pauli string category) to 11 for the rightmost 2 ancillas
    
    qc.mcx(list(control_qubits),site)
    qc.mcx(list(control_qubits),site+1)
    
    qc.x(n) # Together with previous X gate, these map 00 (corresponding to the Z Pauli string category) to 11
    
    qc.append(mcz_gate,list(control_qubits) + [site])
    
# Now, we handle the n^{th} site

for site in range(n-1,n):
    
    for k in range(m):
        if np.mod(site,2**k) == 0:
            qc.x(n+2+k)
    
    qc.append(mcy_gate,list(control_qubits) + [site])
    qc.append(mcy_gate,list(control_qubits) + [0]) # Since the next site after n-1 is 0, not n
    
    qc.x(n+1)
    
    qc.mcx(list(control_qubits),site)
    qc.mcx(list(control_qubits),0)
    
    qc.x(n)
    
    qc.append(mcz_gate,list(control_qubits) + [site])
```

Since the SELECT operator is only supposed to apply to the data bits, with the ancilla just serving as controls, we need to revert the ancilla upon which we applied $X$ gates in this sequence. To do so, we simply rerun all of the ancillary-bit-flip operations in the above sequence, ensuring that in the overall SELECT operation, every ancillary bit has an even number of $X$ gates applied to it and thus remains unchanged:

```python
for site in range(n):
    for k in range(m):
        if np.mod(site,2**k) == 0:
            qc.x(n+2+k) # Reverts the m-bit register
    qc.x(n+1) # This and the next operation revert the rightmost 2 ancillas
    qc.x(n)
    
qc.x(n+m+2) # Reverting the lefmost ancilla
```

### Applying the Inverse PREP Gate

Finally, we apply $\mathrm{PREP}^{\dagger}$ by reversing the gates in PREP and taking the adjoint of each gate. Conveniently, all gates here are Hermitian except the rotation gate, for which the adjoint simply involves flipping the rotation angle:

```python
qc.h(range(n+2,n+m+2))

qc.ch(n,n+1)

qc.ry(-2*theta,n)
```

<img width="1736" height="2878" alt="AfterU" src="https://github.com/user-attachments/assets/8aa0da0f-bc9e-4294-a118-80b83cb3044b" />

$\frac{1}{16} \ket{000000}\ket{00001001} + \frac{1}{4} \ket{000000}\ket{00010001} + \frac{1}{16} \ket{000000}\ket{00010010} - \frac{1}{16} \ket{000000}\ket{00010111} + \frac{1}{16} \ket{000000}\ket{00100001} - \frac{1}{16} \ket{000000}\ket{01110001} + ... + \frac{1}{16} \ket{011101}\ket{10010000} - \frac{1}{16} \ket{011110}\ket{00011101} + \frac{1}{16} \ket{011110}\ket{11010001} - \frac{1}{16} \ket{011111}\ket{00011101} + \frac{1}{16} \ket{011111}\ket{11010001}$

Note that the data bits have changed, with the excited sites no longer being exclusively sites 0 and 4. Moreover, as desired, the leftmost ancillary bit for all states in this superposition is 0, thus enabling the block-encoding here to be used in the GQSP procedure.

## GQSP: From Hamiltonian to Propagator

Having block-encoded the attenuated Hamiltonian $H' = H/\alpha$, we now seek to block-encode the propagator $e^{-iHt} = e^{-iH'\tau}$, where $\tau = \alpha t$, which in turn can be expanded in powers of $H'$. To that end, Generalized Quantum Signal Processing (https://journals.aps.org/prxquantum/abstract/10.1103/PRXQuantum.5.020368) provides an elegant tool for obtaining polynomials $P(H')$ of $H'$ without the need for the polynomial to have a well-defined parity. The method consists of interleaving the anti-controlled $U = (\mathrm{PREP})^{\dagger} (\mathrm{SELECT}) (\mathrm{PREP})$ (conditioned on the leftmost ancillary bit) with rotation operators on that ancillary bit. The key is to calculate the phase angles for the varios rotation operators, which requires significant classical pre-processing, as we will now solve.

### Establishing Propagation Time and Polynomial Order

We first define $\tau$ (i.e., the time in units of $1/\alpha$) and the polynomial order cutoff $d$. For our example, we set $\tau$ to 10. For $d$, we use the relationship $d \sim \tau + \log{(1/\epsilon}/\log{\log(1/\epsilon)}$, where $\epsilon$ is the error tolerance. We set this error tolerance to $10^{-9}$ in our case:

```python
tau = 10

epsilonerror = 10**(-9) # error tolerance
epsiloninv = 1/epsilonerror

d = int(np.ceil(tau + np.log(epsiloninv)/np.log(np.log(epsiloninv))))
```

### Calculating Coefficients of P(x)

We now seek to determine the coefficients of $P(x)$, where $x$ is any given eigenvalue of $H'$. The Jacobi-Anger expansion of the propagator takes the following form:

$$
e^{-ix\tau} = J_0(\tau) + 2 \sum_{k = 1}^\infty (-i)^k J_k(\tau) T_k(x),
$$

where J_k and T_k are the Bessel and Chebyshev polynomials of the first kind, respectively. In general, T_0(x) = 1, T_1(x) = x, and all higher orders take the recursive form T_{k}(x) = 2xT_{k-1}(x) - T_{k-2}(x). The polynomial decomposition of each T_k can thus be encoded in an array (with the elements corresponding to the ascending range x^0 to x^d) and populated through a recursive for loop:

```python
t = {}

t[0] = np.array([1] + [0]*d)
t[1] = np.array([0] + [1] + [0]*(d-1))

for k in range(2,d+1):
    t[k] = np.array([0]*(d+1)) # Initializing for each n
    t[k][0] = -t[k-2][0] # Special case for i = 0, since t[n-1][m-1] does not exist for this case
    for i in range(1,d+1):
        t[k][i] = 2*t[k-1][i-1] - t[k-2][i]
```

A corresponding array can be found for the overall $P(x)$ by substituting these Chebyshev polynomials into the Jacobi-Anger expansion above. We map each $T_k$ onto the $k^\mathrm{th}$ term in the Jacobi-Anger expansion through a for loop and then sum over the terms:

```python
pxhold = {}

pxhold[0] = np.array([sp.special.jv(0,tau)] + [0]*d)

for i in range(1,d+1):
    pxhold[i] = 2*(-1j)**n*sp.special.jv(i,tau)*t[i]
    
px = sum(pxhold[i] for i in range(d+1))
```

### Converting from P(x) to P(z)

In general, $|x| < 1$, since all values in the Hamiltonian (and hence its eigenvalues) are less than the attenuation factor $\alpha$. On the other hand, if we consider the operator $U$ in which $H/\alpha$ is block-encoded, the eigenvalues $z$ take the form $x \pm i\sqrt{1-x^2}$ (with the latter term coming from the off-diagonal block elements in $U$), implying that $z = e^{\pm i\theta}$, where $\theta = \cos^{-1}(x)$. It is therefore important to convert the array representing the polynomial $P$ from the $x$ basis to the $z$ basis, for which we make the substitution $x = (z + z^{-1})/2$, which yields:

$$
\begin{align}
P(z) &= \sum_{s=0}^d P_s(x) x^s \\
&= \sum_{s=0}^d \frac{1}{2^s} P_s(x) (z + z^{-1})^s \\
&= \sum_{s=0}^d \frac{1}{2^s} P_s(x) \sum_{j=0}^{s} \frac{s!}{j!(s-j)!} z^{s-2j}
\end{align}
$$

From the binomial expansion, we can deduce that any $P_k(z)$ receives contributions from all orders of $P(x)$ at or above $|k|$. This is quantitatively confirmed by defining $k = s - 2j$ and imposing the condition that $0 \leq j \leq s$, corresponding to the requirements that $j \geq 0$ and $j \geq -k$. For non-negative values of $k$, the second requirement is automatically satisfied if the first is satisfied. Moreover, the fact that $P(z)$ is invariant in the reciprocal transformation of $z$ implies that $P_{-k}(z) = P_k(z)$ for all k. This common polynomial coefficient for $k$ and $-k$ thus becomes the following:

$$
P_|k|(z) = \sum_{j=0}^{\lfloor \frac{d-k}{2} \rfloor} \frac{1}{2^{k+2j}} \frac{(k+2j)!}{j!(k+j)!} P_{k+2j}(x)
$$

We use a for loop to map the $P(x)$ array onto the $P(z)$ array for non-negative orders, and then we expand the array to append the negative orders:

```python
pzhold = {}

# Starting with only non-negative values:

pzraw = np.zeros(d+1, dtype=complex)

for k in range(d+1):
    pzhold[k] = np.zeros(d+1, dtype=complex)
    for i in range(int(np.floor((d-k)/2))+1):
        pzhold[k][i] = 1/2**(k+2*i)*sp.special.factorial(k+2*i)/(sp.special.factorial(i)*sp.special.factorial(k+i))*px[k+2*i]
    pzraw[k] = sum(pzhold[k][i] for i in range(int(np.floor((d-k)/2))+1))
    
# Expanding to include negative k values (by symmetry P_{-k}(z) = P_k(z)) and shift from [-d,d] to [0,2d]:

pz = np.zeros(2*d+1, dtype=complex)

for kprime in range(d):
    k = kprime - d
    pz[kprime] = pzraw[-k]
    
for kprime in range(d,2*d+1):
    k = kprime - d
    pz[kprime] = pzraw[k]
```

### Determining Q(z)

To find the coefficients of $Q(z)$ (i.e., the complementary polynomial to $P(z)$), we use the complex polynomial code for the GQSP code file FindingQ.ipynb (https://github.com/Danimhn/GQSP-Code/blob/main/FindingQ.ipynb). The extra input array $x$ features the real and imaginary parts of a guess for the $Q(z)$ coefficients. The code then finds the distance of $|P(z)|^2 + |Q(z)|^2$ from 1, as well as the gradient of this distance over the elements of $x$. It adjusts the $x$ values accordingly until $|P(z)|^2 + |Q(z)|^2$ is within a negligible distance from 1. We modify the code to cut off the order of $Q$ at 64. To accommodate this, we also pad $P(z)$ with zeros up to that cutoff order:

```python
import torch
from torch.nn.functional import conv1d, pad
from torch.fft import fft
from torchaudio.transforms import Convolve, FFTConvolve
import time
import numpy as np

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

xarray = np.random.uniform(-0.1,0.1,4*d+2) # needs to be twice the size of P
x = torch.from_numpy(xarray)

def objective_torch(x, P):
    x.requires_grad = True

    real_part = x[:len(x) // 2]
    imag_part = x[len(x) // 2:]

    real_flip = torch.flip(real_part, dims=[0])
    imag_flip = torch.flip(-1*imag_part, dims=[0])

    conv_real_part = FFTConvolve("full").forward(real_part, real_flip)
    conv_imag_part = FFTConvolve("full").forward(imag_part, imag_flip)

    conv_real_imag = FFTConvolve("full").forward(real_part, imag_flip)
    conv_imag_real = FFTConvolve("full").forward(imag_part, real_flip)

    # Compute real and imaginary part of the convolution
    real_conv = conv_real_part - conv_imag_part
    imag_conv = conv_real_imag + conv_imag_real

    # Combine to form the complex result
    conv_result = torch.complex(real_conv, imag_conv)

    # Compute loss using squared distance function
    loss = torch.norm(P - conv_result)**2
    return loss

def complex_conv_by_flip_conj(x):
    real_part = x.real
    imag_part = x.imag

    real_flip = torch.flip(real_part, dims=[0])
    imag_flip = torch.flip(-1*imag_part, dims=[0])

    conv_real_part = FFTConvolve("full").forward(real_part, real_flip)
    conv_imag_part = FFTConvolve("full").forward(imag_part, imag_flip)

    conv_real_imag = FFTConvolve("full").forward(real_part, imag_flip)
    conv_imag_real = FFTConvolve("full").forward(imag_part, real_flip)

    # Compute real and imaginary part of the convolution
    real_conv = conv_real_part - conv_imag_part
    imag_conv = conv_real_imag + conv_imag_real

    # Combine to form the complex result
    return torch.complex(real_conv, imag_conv)

times = []
final_vals = []
num_iterations = []

for k in range(4, 20):
    N = 2 ** k
    
    if N > len(pz):
        padded_pz = np.pad(pz, (0, N - len(pz)), 'constant')
    else:
        padded_pz = pz
    poly = torch.from_numpy(padded_pz).to(device=device, dtype=torch.complex128)

    granularity = 2 ** 25
    P = pad(poly, (0, granularity - poly.shape[0]))
    ft = fft(P)

    # Normalize P
    P_norms = ft.abs()
    poly /= torch.max(P_norms)

    conv_p_negative = complex_conv_by_flip_conj(poly)*-1
    conv_p_negative[poly.shape[0] - 1] = 1 - torch.norm(poly) ** 2

    # Initializing Q randomly to start with
    initial = torch.randn(poly.shape[0]*2, device=device, requires_grad=True)
    initial = (initial / torch.norm(initial)).clone().detach().requires_grad_(True)

    optimizer = torch.optim.LBFGS([initial], max_iter=1000)

    t0 = time.time()

    def closure():
        optimizer.zero_grad()
        loss = objective_torch(initial, conv_p_negative)
        loss.backward()
        return loss

    optimizer.step(closure)

    t1 = time.time()

    total = t1-t0
    times.append(total)
    final_vals.append(closure().item())
    num_iterations.append(optimizer.state[optimizer._params[0]]['n_iter'])
    print(f'N: {N}')
    print(f'Time: {total}')
    print(f'Final: {closure().item()}')
    print(f"# Iterations: {optimizer.state[optimizer._params[0]]['n_iter']}")
    print("-----------------------------------------------------")
    
    if N == 64:
        
        final_q = initial.detach().cpu().numpy()
        
        # Split it back into its real and imaginary parts to rebuild Q(z)
        half = len(final_q) // 2
        q_real = final_q[:half]
        q_imag = final_q[half:]
        q_coefficients = q_real + 1j * q_imag
        
        break

print(times)
print(final_vals)
print(num_iterations)
```

### Calculating the Phase Angles

Now that we have the coefficients for both $P(z)$ and $Q(z)$, we can use the GQSP paper's Algorithm 1 to determine which rotation operator we need to apply for every interleaving step. 
