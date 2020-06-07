import numpy as np
from random import random
import matplotlib.pyplot as plt
import time
import logging

logging.basicConfig(level=logging.INFO,filename='simulation.log', filemode='w',format='%(asctime)s - %(message)s',datefmt='%d-%b-%y %H:%M:%S')
np.seterr(all='warn')

#################################################
#                                               #
#              SIMULATION MACROS                #
#                                               #
#################################################
"""__________________________________________
Simulation MACROs:
T_max and T_min is range of temperature.
nt is number of Temperature points.
sweeps are number of mc steps per spin.
min_meas is minimum number Measurement.
j_knife_factor is jack knife factor is used when number of measurement interval < 2 x Correlation time.
All some_variables0 are default value.
------------------------------------------"""
logging.info("Starting Ising Model Simulation")
T_min = 1.5; T_max = 3
nt = int((T_max-T_min)*10+1)
sweeps0 = 1000
max_sweeps = sweeps0*10
min_meas = 100
j_knife_factor0 = 1
startTime = time.time()
T = np.linspace(T_min, T_max, nt)
"""
We will work with expanding lattices. We will store expanded lattice for particular temperature. Stored lattice would be used as initial configuration for higher dimenssion lattic size. We have two methods for expanding lattice: zooming and stacking. We recommend stacking for use.
"""
states = {_: None for _ in T}
#lattice_sizes = 3**(np.arange(2,5))
################OR##################
lattice_sizes = 2**(np.arange(4,8))

#################################################
#                                               #
#                   FUNCTIONS                   #
#                                               #
#################################################
"""Onsagar's solutions"""
def onsagar_specific_heat(X):
    const = -(2/2.269)**2*2/np.pi
    return const*np.log(abs(np.ones(len(X))-X/2.269))
def onsagar_mag(X):
    lst1 = (1-(np.sinh(np.log(1+np.sqrt(2))*2.269/X[X<2.269]))**(-4))**(1/8)
    lst2 = 0*X[X>=2.269]
    return np.concatenate((lst1,lst2))


"""Monte Carlo Metropolis algorithm"""
def monteCarlo(n, state, energy, mag, beta, sweeps,max_sweeps):
    if sweeps > max_sweeps:
        sweeps = max_sweeps
    exp_betas = np.exp(-beta*np.arange(0,9))
    energies, mags = np.zeros(sweeps), np.zeros(sweeps)
    # random state indices
    J = np.random.randint(0, n, size=(sweeps, n*n))
    K = np.random.randint(0, n, size=(sweeps, n*n))
    #loop
    for t in range(sweeps):
        for tt in range(n*n):
            # random indices
            j, k = J[t, tt], K[t, tt]
            s = state[j,k]
            neighbour_sum = (state[(j-1)%n, k] +
                             state[j, (k-1)%n] + state[j, (k+1)%n] +
                             state[(j+1)%n, k])
            energy_diff = 2*s*neighbour_sum
            if energy_diff < 0 or random() < exp_betas[energy_diff]:
                s *= -1
                energy += energy_diff
                mag += 2*s
            state[j, k] = s
        energies[t], mags[t] = energy, mag
    return energies, mags


"""Calculation of auto-correlation""" 
def autocorrelation(M):
    start_time = time.time()
    tau = 1
    sweeps = len(M)
    auto = np.zeros(sweeps)
    for t in range(sweeps):
        some_time = sweeps-t
        first_term = np.average(M[:some_time]*M[t:sweeps])
        S1 = np.average(M[:some_time])
        S2 = np.average(M[t:sweeps])
        auto_temp = first_term - S1*S2
        if auto_temp > 0:
            auto[t] = auto_temp
        else:#remove oscillating part
            break 
    if auto[0] != 0:
        auto = auto[auto>0]
        auto = auto/auto[0] #normalization
        len_auto = len(auto)
        if len_auto > 1: #draw a straight line if you have atleast two points
            tau = int(-1/np.polyfit(np.arange(len_auto), np.log(auto), 1, w=np.sqrt(auto))[0])
    tau = max(tau,1)
    logging.info(f"Correlation time = {tau}")
    return tau


"""
Calculation of specific heat or Susceptibility and errorbar.
CX is Specific Heat or Susceptibility.
CX_i is Specific Heat or Susceptibility without i-th measurement.
"""
def jackKnife(EM,factor=1):
    n = len(EM)
    CX = np.var(EM)
    CX_i = np.zeros(n)
    for i in range(n):
        CX_i[i] = np.var(np.delete(EM,i))
    under = np.sum(np.square(np.full(n,CX) - CX_i))
    CX_err = np.sqrt(under*factor)
    return CX, CX_err

"""
Stacking Lattices: Stacking z lattice and taking advantage of periodic boundary condition. The energy and magnetization would also increase as system size increase as they are extensive state variables. Other trick to explore is Zoom.
"""
def stackLattice(z,state,energy,mag):
    h_stack_state = state
    for _ in range(z-1):
        h_stack_state = np.hstack((h_stack_state,state))
    v_stack_state = h_stack_state
    for _ in range(z-1):
        v_stack_state = np.vstack((v_stack_state,h_stack_state))
    return (v_stack_state, z*z*energy, z*z*mag)

#################################################
#                                               #
#                     MAIN                      #
#                                               #
#################################################
"""we will plot the following wrt temperature, T"""
plotEnergy = np.zeros(nt)
plotMag = np.zeros(nt)
plotChi = np.zeros(nt)
plotChi_err = np.zeros(nt)
plotSH = np.zeros(nt)
plotSH_err = np.zeros(nt)
plotCorrelation = np.zeros(nt)


"""
Preparing n x n lattice with all spins up.
Here, z is a zoom factor or a stacking factor.
"""
n = min(lattice_sizes)
N = n*n
z = lattice_sizes[1]//lattice_sizes[0]
state = np.ones((n,n),dtype="int")
energy, mag = -N, N
"""lattice size loop"""
for n in lattice_sizes:
    logging.info(f"Lattice size is {n}x{n}")
    print(f"Lattice size is {n}x{n}")
    N = n*n
    """temperature loop"""
    for k in range(nt):
        temp = T[k]
        Beta=1/temp
        if states[temp] !=  None:
            (state,energy,mag) = states[temp]
        logging.info("_"*35)
        logging.info("Temperature is %0.2f, time elapsed %d" %(temp,time.time()-startTime))
        sweeps = sweeps0; j_knife_factor = j_knife_factor0; measurements = 0
        E, M = np.zeros(0), np.zeros(0)
        while measurements < min_meas:
            energies, mags = monteCarlo(n, state, energy, mag, Beta, sweeps, max_sweeps//10)
            energy, mag = energies[-1], mags[-1]
            E = np.concatenate((E,energies))
            M = np.concatenate((M,mags))
            delta_int = eq_time = 2*autocorrelation(M)
            measurements = len(E[eq_time::delta_int])
            logging.info(f"{measurements} measurements are possible")
            if  measurements < min_meas:
                _energies_ = len(E)
                if _energies_ < max_sweeps:
                    sweeps = delta_int*(min_meas-measurements)
                    logging.info(f"\tdoing {sweeps} more sweeps")
                else:
                    delta_int = (_energies_-eq_time)//min_meas
                    j_knife_factor = eq_time/delta_int
                    measurements = len(E[eq_time::delta_int])
                    logging.info(f"We will do {measurements} measurements")
        
        
        #doing measurements
        E = E[eq_time::delta_int]
        M = M[eq_time::delta_int]
        plotMag[k] = np.average(M)/N
        Chi, Chi_err = jackKnife(M,j_knife_factor)
        plotChi[k] =Chi*Beta/N
        plotChi_err[k] =Chi_err*Beta/N
        plotEnergy[k] = np.average(E)/N
        sp_heat, sp_heat_err = jackKnife(E,j_knife_factor)
        plotSH[k] = sp_heat*Beta*Beta/N
        plotSH_err[k] = sp_heat_err*Beta*Beta/N
        plotCorrelation[k] = eq_time//2
        
        
        #lattice expansion
        states[temp] = stackLattice(z,state,energy,mag)
        #states[temp] = zoomLattice(z,state,energy,mag)
        
        
    #PLOTS##PLOTS##PLOTS##PLOTS##PLOTS##PLOTS##PLOTS##PLOTS#
    f = plt.figure(figsize=(16, 9));
    title_name = "Size:"+str(n)+"x"+str(n)
    plt.title(title_name, color='b');

    sp =  f.add_subplot(2, 2, 1 );
    plt.scatter(T, plotEnergy, s=50, marker='o', color='IndianRed')
    plt.xlabel("Temperature (T)", fontsize=20);
    plt.ylabel("Energy ", fontsize=20); plt.axis('tight');

    sp =  f.add_subplot(2, 2, 2 );
    plt.scatter(T, abs(np.array(plotMag)), s=50, marker='o', color='IndianRed', label = "data")
    temp_list = np.linspace(T_min, T_max, 10000)
    plt.plot(temp_list, onsagar_mag(temp_list) , color='blue', label = "Onsager Solution") 
    plt.legend()
    plt.xlabel("Temperature (T)", fontsize=20); 
    plt.ylabel("Magnetization ", fontsize=20);   plt.axis('tight');

    sp =  f.add_subplot(2, 2, 3 );
    plt.errorbar(T, plotSH, yerr = plotSH_err, fmt='o', color='IndianRed', label = "data")
    plt.plot(temp_list, onsagar_specific_heat(temp_list), color='RoyalBlue', label = "Onsager Solution") 
    plt.legend()
    plt.xlabel("Temperature (T)", fontsize=20);  
    plt.ylabel("Specific Heat ", fontsize=20);   plt.axis('tight');   

    sp =  f.add_subplot(2, 2, 4 );
    plt.errorbar(T, plotChi, yerr = plotChi_err, fmt='o', color='IndianRed', label = "data")
    plt.xlabel("Temperature (T)", fontsize=20); 
    plt.ylabel("Susceptibility", fontsize=20);   plt.axis('tight');

    timeIs = time.strftime("%H-%M-%S")
    plt.savefig(timeIs+'.pdf')
    
    #storing measurements in in a file
    with open(str(n)+"data","w") as file:
        file.write("##Temp\tEnergy\tMag\tSp_ht\tSp_ht_err\tChi\tChi_err\ttau\n")
        for i in range(nt):
            file.write(str(T[i])+"\t"+str(plotEnergy[i])+"\t"+str(plotMag[i])+"\t"+str(plotSH[i])+"\t"+str(plotSH_err[i])+"\t"+str(plotChi[i])+"\t"+str(plotChi_err[i])+"\t"+str(plotCorrelation[i])+"\t"+"\n")
