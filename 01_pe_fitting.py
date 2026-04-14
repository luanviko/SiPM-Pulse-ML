from array import array 
import ROOT 
import numpy as np


def fit_data(data, pe_value = 1., title:str = "Global", xrange=[], fitrange = []):
    
    parameters = array('d', 15*[0.])
    canvas = ROOT.TCanvas("canvas", "canvas", 600, 600)
    hist1  = ROOT.TH1F("timing1", title, 167, xrange[0], xrange[1])
    
    fit1 = ROOT.TF1("fit1", "gaus", 0/pe_value,  14/pe_value)    
    fit2 = ROOT.TF1("fit2", "gaus", 14/pe_value, 24/pe_value)    
    fit3 = ROOT.TF1("fit3", "gaus", 24/pe_value, 31/pe_value)   
    fit4 = ROOT.TF1("fit4", "gaus", 31/pe_value, 40/pe_value) 
    fit5 = ROOT.TF1("fit5", "gaus", 40/pe_value, 50/pe_value) 
    
    total = ROOT.TF1("mstotal","gaus(0)+gaus(3)+gaus(6)+gaus(9)+gaus(12)",0,140);
    
    fit1.SetParameters(4000, 8/pe_value,  1)
    fit2.SetParameters(2000, 18/pe_value, 1)
    fit3.SetParameters(1800, 26/pe_value, 1)
    fit4.SetParameters(1000, 34/pe_value, 1)
    fit5.SetParameters(500,  44/pe_value, 1)
        
    for i in range(0, len(data)):
        hist1.Fill(data[i]/pe_value)
    canvas.cd(0)    
    hist1.Draw("HIST")
    
    hist1.Fit("fit1", "R+")
    hist1.Fit("fit2", "R+")
    hist1.Fit("fit3", "R+")
    hist1.Fit("fit4", "R+")
    hist1.Fit("fit5", "R+")
    
    fit1.Draw("SAME")
    fit2.Draw("SAME")
    fit3.Draw("SAME")
    fit4.Draw("SAME")
    fit5.Draw("SAME")

    
    all_parameters = []
    for fit in [fit1, fit2, fit3, fit4, fit5]:
        all_parameters.append(fit.GetParameters()[0])
        all_parameters.append(fit.GetParameters()[1])
        all_parameters.append(fit.GetParameters()[2])
    
    i = 0
    for param in all_parameters:
        parameters[i] = all_parameters[i]
        i = i+1
    
    param0 = fit1.GetParameters()[0]
    param1 = fit1.GetParameters()[1]
    param2 = fit1.GetParameters()[2]
    param3 = fit2.GetParameters()[0]
    param4 = fit2.GetParameters()[1]
    param5 = fit2.GetParameters()[2]
    param6 = fit3.GetParameters()[0]
    param7 = fit3.GetParameters()[1]
    param8 = fit3.GetParameters()[2]
    param9 = fit4.GetParameters()[0]
    param10 = fit4.GetParameters()[1]
    param11 = fit4.GetParameters()[2]
    param12 = fit5.GetParameters()[0]
    param13 = fit5.GetParameters()[1]
    param14 = fit5.GetParameters()[2]
        
    # total.SetParameters(param0, param1, param2, param3, param4, param5, param6, param7, param8, param9, param10, param11, param12, param13, param14)
    total.SetParameters(parameters)    
    hist1.Fit(total, "R+")
    total.Draw("SAME")
        
    parameters.append(param0)
    parameters.append(param1)
    parameters.append(param2)
    
    
    canvas.SaveAs("./pe_fitting.png")

    amps    = [param0, param3, param6, param9, param12]
    means   = [param1, param4, param7, param10, param13]
    sigmas  = [param2, param5, param8, param11, param14]

    return amps, means, sigmas

amplitude = np.load("./data/processed_data_validated.npz")['amplitude']
amps, means, sigmas = fit_data(amplitude, xrange = [0,80])

pe_value = (means[-1]-means[0])/4.

error_m1 = sigmas[0] / np.sqrt(amps[0])
error_m5 = sigmas[-1] / np.sqrt(amps[-1])
pe_error = np.sqrt(error_m1**2 + error_m5**2)/4.

print(f'{np.average(np.diff(means)):.2f} mV/PE')

print(f"PE Value: {pe_value:.2f} mV ± {pe_error:.2f} mV")