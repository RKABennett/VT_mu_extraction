from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random
import os
import sys
from VT_mu_extraction import extraction

# get the current working directory
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path + '/../')

data = dir_path + '/IdVd_data'            # directory with Id vs. Vds data 
Lchs = [0.2, 0.4, 0.6, 0.8, 1.0]          # channel lengths in um
vgs_vals = [3.1, 3.2, 3.3, 3.4, 3.5] # gate voltages in V
EOT = 10                                  # equivalent oxide thickness in nm

np.random.seed(0) # seed for reproducibility

IDT = 1e-6
NMC = 500

mu, mu_error, VT, VT_error = extraction(
                       data,
                       Lchs,
                       vgs_vals,
                       EOT,
                       IDT,
                       NMC,
                       plot_Vdsi_extractions = True,
                       plot_deltaVC_extractions = True,
                       plot_histograms = True,
                       )

print('Estimated mobility: {} cm^2 V^s-1 s^-1'.format(mu))
print('Estimated mobility error: {} cm^2 V^s-1 s^-1'.format(mu_error))
print('Estimated VT: {} V'.format(VT))
print('Estimated VT error: {} V'.format(VT_error))
