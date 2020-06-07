# optimized-ising-model
This repository aims to provide optimized ising model simulation.
This repository also provide extra feature like stacking and zooming which optimize the code by building "good" seed.

The report aims to give a short description of the Ising two-dimensional model and an introduction to Monte Carlo simulation. To draw results, we implemented the Metropolis-Hastings algorithm, which helps us to built likely configurations at a particular temperature. In order to approximate critical exponents, we did finite-size scaling. At the end of this report, we included optimized python code for doing the simulation.

The ising.py is fast code which works on following libraries
Numba, NumPy, Matplotlib, and tqdm.

The ising_low.py is slow code.
