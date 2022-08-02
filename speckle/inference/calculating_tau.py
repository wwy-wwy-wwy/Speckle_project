import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
import math

def objective(x, a, b):
	return a * x + b

def avg_correlation_single_px(corr_maps,px,loadtRange,load_t_Range, laglist_fx,laglist_global): #return seconds
    px_corrmap=corr_maps.GetCorrTimetrace(px,zRange=loadtRange)
    average_tau=[]
    for tau in laglist_fx:
        tauidx=laglist_global.index(tau)
        sum=0
        for t in range(len(load_t_Range)):
            sum=sum+px_corrmap[tauidx,t]
        px_tau=sum/len(load_t_Range)
        average_tau.append(px_tau)
    return average_tau

def plot_correlation_vs_lagtime(px,average_tau,lagtime_fx):
    plt.figure(figsize=(8,5))
    plt.plot(lagtime_fx,average_tau,'.',label=str(px)+' correlation vs time')
    plt.legend(fontsize=18)
    plt.xlabel("Lagtime [s]",fontsize=18)
    plt.ylabel("Correlation",fontsize=18)

def fit_exponential_decay(plateau,average_tau,lagtime_fx,startidx,endidx, plotBoolean):
    load_correlation=average_tau
    substracted_correlation=average_tau-plateau
    log_correlation=np.zeros(len(lagtime_fx))
    log_correlation_s=np.zeros(len(lagtime_fx))
    for i in range(len(lagtime_fx)):
        log_correlation[i]=math.log(abs(load_correlation[i]))
        log_correlation_s[i]=math.log(abs(substracted_correlation[i]))
    if plotBoolean==True:
        x, y = lagtime_fx, log_correlation_s
        # curve fit
        popt, _ = curve_fit(objective, x[startidx:endidx], y[startidx:endidx])
        # summarize the parameter values
        a, b = popt
        print("tau is:",-1/a)
        print('y = %.5f * x + %.5f' % (a, b))
        # plot input vs output
        plt.scatter(x, y)
        
        # define a sequence of inputs between the smallest and largest known inputs
        x_line = np.arange(min(x), max(x), 1)
        # calculate the output for the range
        y_line = objective(x_line, a, b)
        # create a line plot for the mapping function
        plt.plot(x_line, y_line, '--', color='red')
        plt.show()
        return -1/a,a,b
    else:
        x, y = lagtime_fx, log_correlation_s
        # curve fit
        popt, _ = curve_fit(objective, x[startidx:endidx], y[startidx:endidx])
        # summarize the parameter values
        a, b = popt
        print(-1/a)
        return -1/a
    
def fit_exponential_decay_rsq(plateau,average_tau,lagtime_fx,startidx,endidx, plotBoolean):
    load_correlation=average_tau
    substracted_correlation=average_tau-plateau
    log_correlation=np.zeros(len(lagtime_fx))
    log_correlation_s=np.zeros(len(lagtime_fx))
    for i in range(len(lagtime_fx)):
        log_correlation[i]=math.log(abs(load_correlation[i]))
        log_correlation_s[i]=math.log(abs(substracted_correlation[i]))
    if plotBoolean==True:
        x, y = lagtime_fx, log_correlation_s
        # curve fit
        popt, _ = curve_fit(objective, x[startidx:endidx], y[startidx:endidx])
        # summarize the parameter values
        a, b = popt
        print("tau is:",-1/a)
        print('y = %.5f * x + %.5f' % (a, b))
        # plot input vs output
        plt.scatter(x, y)
        
        # define a sequence of inputs between the smallest and largest known inputs
        x_line = np.arange(min(x), max(x), 1)
        # calculate the output for the range
        y_line = objective(x_line, a, b)
        # create a line plot for the mapping function
        plt.plot(x_line, y_line, '--', color='red')
        plt.show()
        
        fit=np.zeros(endidx-startidx)
        for i in range(endidx-startidx):
            fit[i]=a*x[i+startidx]+b
        corr_matrix = np.corrcoef(y[startidx:endidx],fit)
        corr = corr_matrix[0,1]
        R_sq = corr**2
        print('R square is '+str(R_sq))
        return -1/a,a,b
    else:
        x, y = lagtime_fx, log_correlation_s
        # curve fit
        popt, _ = curve_fit(objective, x[startidx:endidx], y[startidx:endidx])
        # summarize the parameter values
        a, b = popt
        print(-1/a)
        return -1/a

def reconstruct(lagtime_fx,plateau,average_value,a,b):
    y=np.zeros((len(lagtime_fx)))
    for i in range(len(lagtime_fx)):
        y[i]=np.exp(b+a*lagtime_fx[i])+plateau
    plt.figure(figsize=(8,5))
    plt.plot(np.log(lagtime_fx),y,'*',label="fit line")
    plt.plot(np.log(lagtime_fx),average_value,'.',label="data")
    plt.legend(['Correlation vs Log Lagtime'], fontsize=18)
    plt.xlabel("Lagtime [s]",fontsize=18)
    plt.ylabel("Correlation",fontsize=18)
    plt.legend(fontsize=18)
    plt.tick_params(direction='in')
    
def avg_correlation(corr_maps, px_list,loadtRange,load_t_Range,laglist_fx,laglist_global):
    average_tau=[]
    correlationmaps=[]
    for pixel in px_list:
        print(pixel)
        correlationmaps.append(corr_maps.GetCorrTimetrace(pixel,zRange=loadtRange))

    average_correlation=[]
    for tau in laglist_fx:
        sum_c=0
        tauidx=laglist_global.index(tau)+1
        for i in range(len(correlationmaps)):
            sum_c=sum_c+calculate_correlation_pixel(tauidx,i,correlationmaps,load_t_Range)
        average_correlation_tau=sum_c/len(correlationmaps)
        average_tau.append(average_correlation_tau)

    return average_tau

def calculate_correlation_pixel(tauidx,corridx,correlationmaps,load_t_Range):
    sum=0
    for t in range(len(load_t_Range)):
        sum=sum+correlationmaps[corridx][tauidx,t]
    c_pixel_t=sum/len(load_t_Range)
    return c_pixel_t

