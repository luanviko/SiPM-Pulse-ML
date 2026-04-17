from array import array 
import ROOT 
import numpy as np

def fit_data(data, pe_value=1.0, title="Global", xrange=[], initial_guesses=[], ranges=[]):
    
    # Numer of Gaussians to fit
    n_gauss = len(initial_guesses)
    
    # Create histogram and fill it with data
    canvas = ROOT.TCanvas("canvas", "canvas", 600, 600)
    hist1  = ROOT.TH1F("timing1", title, 167, xrange[0], xrange[1])
    
    for val in data:
        hist1.Fill(val / pe_value)
    
    # Construct the total fit formula: "gaus(0)+gaus(3)+..."
    formula = "+".join([f"gaus({i*3})" for i in range(n_gauss)])
    total = ROOT.TF1("mstotal", formula, xrange[0], xrange[1])
    
    individual_fits = []
    all_params = []

    # Fit individual Gaussians first to get better seeds
    for i in range(n_gauss):
        fit_name = f"fit{i}"
        low, high = ranges[i][0] / pe_value, ranges[i][1] / pe_value
        
        f_ind = ROOT.TF1(fit_name, "gaus", low, high)
        f_ind.SetParameters(initial_guesses[i][0], initial_guesses[i][1] / pe_value, initial_guesses[i][2])
        
        hist1.Fit(f_ind, "R+N") # N to not draw immediately
        individual_fits.append(f_ind)
        
        # Collect parameters for the total fit
        params = f_ind.GetParameters()
        all_params.extend([params[0], params[1], params[2]])

    # Set parameters for the total combined fit
    total_params_array = array('d', all_params)
    total.SetParameters(total_params_array)
    
    hist1.Draw("HIST")
    hist1.Fit(total, "R+")
    total.Draw("SAME")
    
    # Extract final results
    final_params = total.GetParameters()
    amps   = [final_params[i*3] for i in range(n_gauss)]
    means  = [final_params[i*3+1] for i in range(n_gauss)]
    sigmas = [final_params[i*3+2] for i in range(n_gauss)]
    
    canvas.SaveAs("./pe_fitting.png")
    return amps, means, sigmas

# Load data
amplitude = np.load("./data/01_processed_data_validated.npz")['amplitude']

# Visually determine the initial guesses for 
# the fit parameters (amplitude, mean, sigma) for each peak
guesses = [
    [4000, 8, 1], [2000, 18, 1], [1800, 26, 1], [1000, 34, 1], [500, 44, 1]
]

# Visually determine the fitting ranges for each peak to improve the fit quality
fit_ranges = [
    [0, 14], [14, 24], [24, 31], [31, 40], [40, 50]
]

# Perform the fitting and extract the parameters
amps, means, sigmas = fit_data(
    amplitude, 
    xrange=[0, 80], 
    initial_guesses=guesses, 
    ranges=fit_ranges
)

# I take the average distance between the means of the peaks to estimate the PE value
pe_value = (means[-1] - means[0]) / 4.0
print(f"PE Value: {pe_value:.2f} mV")