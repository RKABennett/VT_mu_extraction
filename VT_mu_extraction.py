###############################################################################
#
# MIT License
#
# Copyright (c) 2024 the Authors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
###############################################################################

###############################################################################
#
# Please cite the following reference:
# [ADD CITATION HERE]
#
###############################################################################

from math import sqrt
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import random
from pathlib import Path
import statsmodels.api as sm
# get the current working directory
dir_path = os.path.dirname(os.path.abspath(__file__))

def extraction(
               foldername, 
               Lchs, 
               Vgs_vals, 
               EOT, 
               IDT, 
               NMC,
               plot_Vdsi_extractions = False,
               plot_deltaVC_extractions = False,
               plot_histograms = False
               ):
    '''
    Applies the method in [CITATION] on Id vs. Vds data to extract the mobility
    and threshold voltage of contact-gated FETs. Mentions of equations,  
    figures, variables, etc can be found in the above reference.

    Keyword arguments:
    foldername          -- Directory where Id vs. Vgs sweeps are stored
    Lchs                -- Array-like object listing out channel lengths used
                           (units: um)
    Vgs_vals            -- Array-like object listing out Vgs values used
                            (units: V)
    EOT                 -- Equivalent oxide thickness (units: nm)
    IDT                 -- I_d^T value used for constant-current extraction
                           (units: uA or uA/um (should be the same as the units
                            for Id)
    NMC                 -- Number of Monte Carlo trials used for error prop
                           (Tip: make sure NMC is large enough by 
                           increasing it to ensure that doing so does not 
                           change the final extracted values)

    Returns:
    mu                  -- Channel mobility (units: cm^2 V^-1 s^-1)
    mu_err              -- Estimated standard error for mu
                           (units: cm^2 V^-1 s^-1)
    VT                  -- Channel inversion threshold voltage (units: V)
    VT_err              -- Estimated standard error for VT (units: V)
    '''
    
    if plot_Vdsi_extractions or plot_deltaVC_extractions or plot_histograms:
        Path(dir_path + '/plots').mkdir(parents=True, exist_ok=True)
        # plot settings
        mpl.rcParams['lines.linewidth'] = 1.5
        mpl.rcParams['axes.linewidth'] = 0.9
        mpl.rcParams['xtick.major.width'] = 0.6
        mpl.rcParams['ytick.major.width'] = 0.6
        mpl.rcParams['xtick.minor.width'] = 0.3
        mpl.rcParams['ytick.minor.width'] = 0.3
        mpl.rcParams['font.family'] = 'arial'
        plt.rcParams['xtick.minor.visible'] = True
        plt.rcParams['ytick.minor.visible'] = True
        plt.rcParams["xtick.top"] = True
        plt.rcParams["ytick.right"] = True
        mpl.rcParams['xtick.direction'] = 'in'
        mpl.rcParams['ytick.direction'] = 'in'
        plt.rcParams.update({'font.size': 16})

    ###########################################################################
    #
    # Compile a table of Vds' and Vgs' values similar to that of Fig. 2e in the
    # main text
    #
    ###########################################################################
    
    data_list = []

    # Extract Vds' and Vgs' values for every Vgs 
    for Vgs in Vgs_vals:
        Vdsi_list,\
        Vch_list,\
        Vgsp,\
        error_VC = extract_at_fixed_Vgs(
                                    foldername, 
                                    Lchs, 
                                    Vgs, 
                                    IDT,
                                    plot_deltaVC_extractions,
                                    plot_Vdsi_extractions
                                    )

        data_list.append([Vdsi_list, Vch_list, Vgsp, error_VC])

    # Now, we iterate through each channel length. We use our extracted Vgsp 
    # values in conjunction with the appropriate Vdsp values to come up with 
    # our x and y values, and their accompanying errors, for the final fit.
    # Here, x and y values are those plotted on the x- and y-axes in Fig. 2e.
    
    xy_mat = build_data_matrix(Lchs, Vgs_vals, data_list)
    
    ###########################################################################
    #
    # Perform multiple Monte Carlo iterations to simulate distributions of mu
    # and VT based on their estimated errors
    #
    ###########################################################################

    mu_list, VT_list, x, y, x_error, y_error = [], [], [], [], [], []

    for i in range(NMC):
        mu_i, VT_i = MC_step(xy_mat, EOT, IDT)
        
        mu_list.append(mu_i)
        VT_list.append(VT_i)

    ###########################################################################
    #
    # Extract the nominal mobility/VT and their accompanying standard errors
    #
    ###########################################################################

    mu_list = np.sort(mu_list)
    VT_list = np.sort(VT_list)
    x = np.array(x)
    y = np.array(y)

    # Here, we filter the histograms by taking the median value +- 34% (i.e., 
    # the central 68% of the histograms) to stay roughly in line with Gaussian 
    # statistics, as we discuss in the Supporting Information. 

    p1 = 16     # exclude the bottom 16% of the histogram
    p2 = 100-p1 # exclude the top 16% of the histrogram

    mu_filtered = mu_list[
                          int(np.size(mu_list)* p1/100)
                          :
                          int(np.size(mu_list)* p2/100)
                         ]

    VT_filtered = VT_list[
                          int(np.size(VT_list)* p1/100)
                          :
                          int(np.size(VT_list)* p2/100)
                         ]

    # Uncomment the following lines to plot histograms of mu and VT before 
    # and after filtering. (It's normal for unfiltered distributions to contain 
    # extreme outliers. This is one of the reasons why we filter.)     
    
    # Calculate and return the final mu and VT values
    mu = (mu_filtered[0] + mu_filtered[-1])/2 
    mu_error = (mu_filtered[-1] - mu_filtered[0])/2
    VT = (VT_filtered[0] + VT_filtered[-1])/2
    VT_error = (VT_filtered[-1] - VT_filtered[0])/2

    if plot_histograms:
        histogram_fit(mu_list, 2, foldername, 'mobility')
        histogram_fit(mu_filtered,2, foldername, 'mobility_filtered')
        histogram_fit(VT_list, 0.2, foldername, 'VT')
        histogram_fit(VT_filtered, 0.2, foldername, 'VT_filtered')

    return mu, mu_error, VT, VT_error

def extract_at_fixed_Vgs(foldername, Lchs, 
                         Vgs, IDT, plot_deltaVC_extractions, 
                         plot_Vdsi_extractions):
    '''
    Extracts the voltage drop across the channel, Vch, for an Id vs. Vds sweep 
    at a fixed target current = I_d^T.

    Keyword arguments:
    foldername          -- TBA
    
    Lchs                -- Array-like object listing out channel lengths used
                           (units: um)
    Vgs                 -- Vgs value at which Id vs. Vds sweep is performed
    IDT                 -- I_d^T value used for constant-current extraction
                           (units: uA or uA/um (should be the same as the units
                            for Id))


    Returns:
    Vdsi_list           -- List containing the V_ds^(i) value for each Lch
    Vdsip_list          -- List containing V_ds^(i)' values for each Lch
    Vgsp                -- Vgs'
    error_delta_VC      -- Estimated standard error in delta V_C
    '''

    Vdsi_list = find_Vds_list(foldername, Lchs, Vgs, IDT, plot_Vdsi_extractions)
    Lchs = np.array(Lchs, dtype = 'float64')

    ###########################################################################
    #
    # Statistics to find b, m, and the error for b
    #
    ###########################################################################
    
    x = Lchs
    x = sm.add_constant(x)
    y = Vdsi_list

    delta_VC, m, error_delta_VC, error_m = linear_regression(x,y)
    Vdsip_list = Vdsi_list - delta_VC   # List of V_ds^(i)'
    Vgsp = Vgs - 0.75*delta_VC          # Vgs'

    ###########################################################################
    #
    # Optional: uncomment this section to generate a TLM-like plot (same as 
    # in Fig. 2b) for the delta V_C extraction.
    #
    ###########################################################################

    if plot_deltaVC_extractions:
        Path(dir_path + '/plots/deltaVC_extractions').mkdir(
                                                      parents=True, 
                                                      exist_ok=True)
        plotname = '/deltaVC_extraction_Vgs={}.png'.format(Vgs)
        x = np.array([0, np.max(Lchs)])
        y = m*x + delta_VC

        fig, ax = plt.subplots(1,1)
        
        ax.plot(
                  Lchs, 
                  Vdsi_list*1000, 
                  marker = 'o', 
                  color = 'k', 
                  ls = 'None'
                  )
        ax.plot(
                  x, 
                  y*1000, 
                  color = 'r', 
                  ls = '--', 
                  marker = 'None'
                  )

        ax.set_xlim(0,)
        ax.set_xlabel('$L_\\mathrm{ch} (um)$')
        ax.set_ylabel('$V_{DS}$ (mV) at $I_d = I_\\mathrm{D}^\\mathrm{T}$')
        plt.tight_layout()
        plt.savefig(dir_path + '/{}/{}'.format(
                                               'plots/deltaVC_extractions', 
                                               plotname)
                                               )
        plt.close()

    return(Vdsi_list, Vdsip_list, Vgsp, error_delta_VC)

def find_Vds_list(foldername, Lchs, 
                  Vgs, IDT, plot_Vdsi_extraction):
    '''
    Finds the Vds values where Id = some target current for a family of IdVd 
    curves at different Lchs.

    Keyword arguments:
    foldername          -- TBA
    Lchs                -- Array-like object listing out channel lengths used
                           (units: um)
    Vgs                 -- Vgs value at which Id vs. Vds sweep is performed
    EOT                 -- Equivalent oxide thickness (units: nm)
    IDT                 -- I_d^T value used for constant-current extraction
                           (units: uA or uA/um (should be the same as the units
                           for Id))

    Returns:
    Vdsi_list           -- List of V_ds^(i) values for each Lch values
    '''

    basefilename = foldername + '/Lch={}/IdVd_Vgs={}.csv'
    if plot_Vdsi_extraction:
        Path(dir_path + '/plots/Vdsi_extractions').mkdir(parents=True, 
                                                         exist_ok=True)
        fig, ax = plt.subplots(1,1)

    for Lch in Lchs:
        data = np.loadtxt(
                          basefilename.format(Lch, Vgs), 
                          skiprows = 1, 
                          delimiter = ','
                          ).T
        Vd = data[0]
        Id = data[1]

    Vdsi_list = []

    for Lch in Lchs:
        data = np.loadtxt(
                          basefilename.format(Lch, Vgs), 
                          skiprows = 1, 
                          delimiter = ','
                          ).T
        Vd = data[0]
        Id = data[1]
        Vds_target = find_Vds_targ(Vd, Id, IDT)
        Vdsi_list.append(Vds_target)
       
        if plot_Vdsi_extraction:
            ax.plot(
                     Vd*10**3, 
                     Id*10**6, 
                     label = Lch
                     )
            ax.plot(
                     Vds_target*10**3, 
                     IDT*10**6, 
                     marker = 'o', 
                     color = 'k',
                     )
            ax.axhline(IDT*10**6, ls = '--', color = 'k')
    
    if plot_Vdsi_extraction:
        ax.set_ylim(0,2*IDT*10**6)
        ax.set_xlim(np.min(Vd)*10**3, 1.1*np.max(Vdsi_list)*10**3)
        ax.set_xlabel('$V_\\mathrm{ds}$ (mV)')
        ax.set_ylabel('$I_\\mathrm{d}$ (µA/µm)')
        ax.legend(loc = 'best', title = '$L_\\mathrm{ch}$')
        plt.title('$V_\\mathrm{gs}$ = ' + str(Vgs) + ' V')
        plt.tight_layout()
        plt.savefig(dir_path + '/{}/Vdsi_extraction_Vgs={}.png'.format(
                                                      'plots/Vdsi_extractions',
                                                       Vgs))
        plt.close()
    return np.array(Vdsi_list)

def find_Vds_targ(Vds, Id, IDT):
    '''
    Finds the Vds value of a single IdVd where Id = some target current.


    Keyword arguments:
    Vds and Id      -- Array-like objects of V_ds and I_d values used in 
                       Id vs. Ids sweep (units: V_ds in V; Id in uA or uA/um)
    EOT             -- Equivalent oxide thickness (units: nm)
    IDT             -- I_d^T value used for constant-current extraction
                       (units: uA or uA/um (should be the same as the units
                       for Id))

    Returns:
    Vds_target      -- Vds value at which Id = target current (same as V_ds^(i)
                       in the main text)

    Note that if Vds_target is between two values (which is usually will be), 
    we perform linear extrapolation using the two neighboring datapoints. If
    the Id vs. Vds sweep contains multiple Vds values corresponding to the
    chosen IDT (which can happen if you have multiple sweeps in 
    your curve or if you have noise), then we return only the first. 
    '''

    for i in range(np.size(Vds) - 1):
        if (
            (Id[i] < IDT and Id[i + 1] > IDT) 
            or Id[i] == IDT
            ):
            m = (Id[i+1] - Id[i]) / (Vds[i+1] - Vds[i])
            b = Id[i] - m*Vds[i]
            Vds_target = (IDT - b)/m
            return Vds_target
    
    raise Exception("Could not find Vdsi value. Was your IDT chosen properly?")

def MC_step(xy_mat, EOT, IDT):
    '''
    Implements a single step of the Monte Carlo method (see Section S2). We 
    call this function many times to build up distributions of mu and VT that
    we then process to find better estimates for their nominal values and 
    corresponding errors.


    xy_mat:         -- matrix generated using 'build_data_matrix' function  
    EOT             -- Equivalent oxide thickness (units: nm)
    IDT             -- I_d^T value used for constant-current extraction
                       (units: uA or uA/um (should be the same as the units
                        for Id))


    Returns:
    mu              -- mobility from a single Monte Carlo step (equivalent to
                       \tilde(mu) in Section S2 of the Supporting Information)
    VT              -- Threshold voltage from a single Monte Carlo step 
                       (equivalent to \tilde(VT) in Section S2 of the 
                       Supporting Information)
    '''
    # unpack the xy_mat for convenience
    xval        = xy_mat[:,0]
    error_x     = xy_mat[:,1]
    yval        = xy_mat[:,2]
    error_y     = xy_mat[:,3]
    error_b     = xy_mat[:,4]

    # lists that we will populate with x and y values (as in Fig. 2e) for our
    # eventual linear regression.
    x = []
    y = []

    # apply the method described in Section S2 of the supporting information to
    # generate a series of perturbed x and y data points
    for j in range(np.size(xval)):
        error_in_b1 =  np.random.normal(0,error_b[j],1)[0]
        error_in_x = error_in_b1 * abs(error_x[j])
        error_in_b2 =  np.random.normal(0,error_b[j],1)[0]
        error_in_y = error_in_b2 * abs(error_y[j])
        new_xval = xval[j] + error_in_x
        new_yval = yval[j] + error_in_y
        x.append(new_xval)
        y.append(new_yval)
        
    b, m, b_error, m_error = linear_regression(x,y) 
    
    Cox = 8.85e-12*3.9/(EOT*10**-9)
    A = (10**4/Cox*2*IDT)
    
    # extract the mobility and VT for the single Monte Carlo trial
    mu = A/b
    Vth = m/2
    return mu, Vth

def build_data_matrix(Lchs, Vgs_vals, data_list):
    '''
    Builds a matrix containing [xvals, xerrs, yvals, yerrs, error_VCs]
    where:
        xvals           -- nominal x values in Fig. 2b
        xerrs           -- error bars prefactor for x values in Fig. 2b
        yvals           -- nominal y values in Fig. 2b
        yerrs           -- error bars prefactor for y values in Fig. 2b
        error_VC        -- error in delta VC 

    Here, xerrs and yerrs are prefactors, i.e., they must be multiplied by
    error_VC to get the true error in the x and y values. Thus, xerr and yerr 
    are equivalent to thee quantities described in equations S3 and S4 divided
    by sigma_V_C.

    Keyword arguments:
    Lchs                -- Array-like object listing out channel lengths used
                           (units: um)
    Vgs_vals            -- Array-like object listing out Vgs values used
                            (units: V)
    data_list           -- Matrix-like object built using the 
                        'extract_at_fixed_Vgs' function

    Returns:
    xy_mat              -- The matrix described above

    '''
    xy_mat = []
    for j in range(len(Lchs)):
        Lch = float(Lchs[j])
        for i in range(len(Vgs_vals)):
            Vdsi_list, Vch_list, Vgsp, error_b = data_list[i]
            Vdsp = Vch_list[j]

            A = 2*Vgsp*Vdsp
            B = Vdsp**2
        
            errorA = A*sqrt((3/4 / Vgsp)**2 + (1 / Vdsp)**2)
            errorB = B*sqrt(2*(1 / Vdsp)**2)        

            error_x = 1 / Lch
            error_y = np.sqrt(errorA**2 + errorB**2)/Lch

            xval = Vdsp/Lch
            yval = (2*Vgsp*Vdsp - Vdsp**2)/Lch

            xy_mat.append([xval, error_x, yval, error_y, error_b])

    return np.array(xy_mat)

def histogram_fit(vals, step, 
                  savefoldername, hist_name):
    '''
    Generates and saves a histogram to visualize mu or VT distributions 
    obtained from the Monte Carlo procedure.

    Keyword arguments:
    vals        -- quantities of interest (mu or VT values) whose distribution 
                   you want to see
    step        -- step size for bins
    foldername  -- directory to which you wish to save the histogram
    hist_name   -- file name for saving

    Returns:
    N/A
    '''
    Path(dir_path + '/plots/histograms').mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1,1)
    bins = np.arange(np.min(vals), np.max(vals) + 2*step, step)
    ax.hist(vals, edgecolor = 'k', color = 'gray', bins = bins)
    ax.set_xlim(np.min(vals), np.max(vals))
    ax.set_title('Min = {}, max = {}, range = {}'.format(
                       round(vals[0], 2), 
                       round(vals[-1],2), 
                       round(vals[-1] - vals[0], 2)
                       )
                       )
    ax.set_xlabel('Value')
    ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(dir_path\
                + '/plots/histograms/histogram_{}.png'.format(hist_name))
    plt.close()

def linear_regression(x, y):
    '''
    Implements linear regression for a collection of x,y data. 
    
    Keyword arguments:
    x           -- Array-like object of independent variable
    y           -- Array-like object of dependent variable

    Returns:
    b           -- y-intercept for line of best fit
    m           -- Slope of line of best fit
    b_error     -- Standard error for y-intercept
    m_error     -- Standard error for slope
    '''
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    results_summary = model.summary()
    b = model.params[0] 
    m = model.params[1] 
    b_error = model.bse[0] 
    m_error = model.bse[1] 

    return b, m, b_error, m_error
