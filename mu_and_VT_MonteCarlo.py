from math import sqrt
import numpy as np
import random
from regressions import linear_regression
from scipy.stats import norm

def build_data_matrix(Lch_list, Vgs_list, data_list):
    """
    Builds a data matrix to pass into the Monte Carlo step function. 

    Inputs:
        Lch_list; a vector of channel lengths used

        Vgs_list; a vector of Vgs values used

        data_list; a matrix formatted as
            [Vds_targets, Vchs, Vgsps, error_b's, mvals]

            where:
                Vds_targets is a vector of Vds(i) values
                
                Vchs is a vector of Vds(i)' values
                
                Vgsps is a vector of Vgs' values used
                
                error_b is a vector of the y intercept error from when 
                extracting delta V_ch

                mvals is a vector containing slopes from each fit



    The output matrix xy_mat is formatted as
        xy_mat = [xvals, error_x's, yvals, error_y's, error_b's]
    
    where: 
        xvals is a vector of values of (Vds(i)'/ Lch^(i))
        
        error_x's is a vector of the associated error in xvals
        
        yvals is a vector of values of 
                                    [(2 Vgs' Vds(i)' - (Vds^(i)')^2] / Lch^(i))
        
        error_y's is a vector of the associated error in yvals
        
        error_b's is identical to the input vector of error_b, included as an
        output for convenience.

    """
    xy_mat = []
    for j in range(len(Lch_list)):
        Lch = float(Lch_list[j])
        for i in range(len(Vgs_list)):
            Vds_target_list, Vch_list, Vgsp, error_b, mval = data_list[i]
            Vdsp = Vch_list[j]
            # next two lines are coefficients from Fig. 3c
            error_Vdsp = 1     
            error_Vgsp = 3/4     

            # the next several lines propagate error from the y intercept
            # uncertainty to associated uncertainty in the x and y values used
            # in the final fit. Note that these are merely coefficients for
            # the error; we later multiply them by random error sampled based
            # on error_b to estimate the true error.
            error_x = error_Vdsp / Lch
            A = 2*Vgsp*Vdsp
            B = Vdsp**2
            errorA = A*sqrt((error_Vgsp / Vgsp)**2 + (error_Vdsp / Vdsp)**2)
            errorB = B*sqrt(2*(error_Vdsp / Vdsp)**2)
            error_y = np.sqrt(errorA**2 + errorB**2)/Lch
            xval = Vdsp/Lch
            yval = (2*Vgsp*Vdsp - Vdsp**2)/Lch
            xy_mat.append([xval, error_x, yval, error_y, error_b])

    return np.array(xy_mat)

def MC_step(xy_mat, EOT, target_current):
    """
    Perform one step of the Monte Carlo algorithm. 

    Inputs:
        xy_mat; built and defined using the above function

        EOT; the equivalent oxide thickness in SI units, i.e., [m]

        target_current, the I_D^T used for the extraction

    Outputs the mu and VT for a single Monte Carlo step. We use an averaging 
    process described below to then estimate the actual mu and VT based on
    many small steps.

    """
    xval, error_x, yval, error_y, error_b =\
                                           xy_mat[:,0],\
                                           xy_mat[:,1],\
                                           xy_mat[:,2],\
                                           xy_mat[:,3],\
                                           xy_mat[:,4]
    x = []
    y = []
    x_error = []
    y_error = []

    for j in range(np.size(xval)):
        # draw a random number from a gaussian distribution with standard
        # deviation = error_b. We take this as an estimate for the true error
        # in error_b for one monte carlo step. Then, propagate this error to
        # x using the coefficient determined in build_data_matrix. Then, we
        # repeat this process for y. We take the true x and y for this Monte 
        # Carlo step to then be x + error_x and y + error_y. 
        error_in_b1 =  np.random.normal(0,error_b[j],1)[0]
        error_in_x = error_in_b1 * abs(error_x[j])
        error_in_b2 =  np.random.normal(0,error_b[j],1)[0]
        error_in_y = error_in_b2 * abs(error_y[j])
        new_xval = xval[j] + error_in_x
        new_yval = yval[j] + error_in_y
        x.append(new_xval)
        y.append(new_yval)
        
    # perform linear regression on the x and y values and extract mu and Vth
    # as described in the paper. 
    b, m, b_error, m_error = linear_regression(x,y) 
    Cox = 8.85e-12*3.9/(EOT) # Cox in SI units; note that EOT is in [m]
    A = (10**4/Cox * 2*target_current) # target current is the I_D^T we use
    mu = A/b # mobility in SI units
    Vth = m/2 # Vth in [V]
    return mu, Vth


###############################################################################
#
# Implement Monte Carlo method to propogate error for mu and VT extraction
#
###############################################################################

for i in range(N_montecarlo):
    """
    For each step of N_montecarlo, we:

    1. Propagate one round of error to x and y using the random sampling 
    technique described above

    2. Calculate mu and Vth for that Monte Carlo step with these x and y values 
    """
    mu_i,\
    Vth_i,\
    x_i,\
    y_i,\
    x_error_i,\
    y_error_i = MC_step(xy_mat, EOT, target_current)
    
    mu_list.append(mu_i)
    Vth_list.append(Vth_i)

mu_list = np.sort(mu_list)
Vth_list = np.sort(Vth_list)

# In normal gaussian statistics, we have 68% of the data within one standard
# deviation of the mean. Accordingly, to find the "true" mean and error, we
# assume a similar distribution. We first filter the data based on this 68%
# range, taking the mean value and 34% of the data above/below this median.
# We then take the center of this filtered data (i.e., the overall median) as 
# the nominal mu or VT, and we take the total range/2 as the estimated standard
# error.

p1 = int(50-34)
p2 = int(50+34)

mu_filtered = mu_list[
                      int(np.size(mu_list)* p1/100)
                      :
                      int(np.size(mu_list)* p2/100)
                     ]

Vth_filtered = Vth_list[
                      int(np.size(Vth_list)* p1/100)
                      :
                      int(np.size(Vth_list)* p2/100)
                     ]

mu = (mu_filtered[0] + mu_filtered[-1])/2
mu_error = (mu_filtered[-1] - mu_filtered[0])/2

Vth = (Vth_filtered[0] + Vth_filtered[-1])/2
Vth_error = (Vth_filtered[-1] - Vth_filtered[0])/2

