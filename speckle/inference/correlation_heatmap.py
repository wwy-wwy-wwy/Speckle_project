import seaborn as sns
import numpy as np
import pickle
import matplotlib.pyplot as plt

def get_correlation(tauidx,loadtRange,t_range, px, corr_maps): #return seconds
    px_corrmap=corr_maps.GetCorrTimetrace(px,zRange=loadtRange)
    sum_correlation=0
    for t in range(0,len(t_range),1):
        sum_correlation=sum_correlation+px_corrmap[tauidx,t]
    average_correlation=sum_correlation/len(t_range)
    return average_correlation

def dump_heatmap(lag,laglist,loadtRange,t_range, corr_maps, heatmap_size, numberT_average, foldername):
    index=laglist.index(lag)+1
    heatmap = np.zeros((heatmap_size[0],heatmap_size[1]))
    for y in range(heatmap_size[1]):
        for x in range(heatmap_size[0]):
            print([y,x])
            heatmap[y,x]=get_correlation(index,loadtRange,t_range, [y,x], corr_maps)
    pickle.dump(heatmap, open(foldername+'_heatmap_decay_'+str(lag)+'_lagtime_'+str(numberT_average)+'avg.p',"wb"))
    
def display_correlation_heatmaps(lag, numberT_average, minV, maxV, foldername):
    heatmap_lag=pickle.load(open(foldername+'_heatmap_decay_'+str(lag)+'_lagtime_'+str(numberT_average)+'avg.p', "rb"))
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(heatmap_lag,vmin=minV, vmax=maxV)
    plt.show()
    
def mask_heatmap(center,radius_out,heatmap, scale_bar_position, scale_ratio):
    ##scale_bar_position:[x,y]
    x_center=center[1]
    y_center=center[0]
    x = np.arange(heatmap.shape[0])
    y = np.arange(heatmap.shape[1])
    xv, yv = np.meshgrid(x,y)
    xv -= x_center
    yv -= y_center
    radius = np.sqrt(xv**2 + yv**2)
    mask=radius<radius_out
    heatmap_masked=heatmap*mask
    f, ax = plt.subplots(figsize=(10, 8))
    ax = sns.heatmap(heatmap_masked,vmin=0, vmax=0.7)
    ## scale bar
    ax.plot([scale_bar_position[0],scale_bar_position[0]+5*scale_ratio],[scale_bar_position[1],scale_bar_position[1]],'w-')
    ax.text(scale_bar_position[0]+20,scale_bar_position[1]+10,r'$5\mu m$',color='w')
    plt.show()
    plt.show()
    return heatmap_masked

def plot_radius_distribution(heatmap,radius_out,scale_ratio):
    x = np.arange(heatmap.shape[0])
    y = np.arange(heatmap.shape[1])
    xv, yv = np.meshgrid(x,y)
    xv -= x_center
    yv -= y_center
    radius = np.sqrt(xv**2 + yv**2)
    step=1
    lower_bounds=np.arange(0,radius_out,step)
    micron_list=(lower_bounds+step/2)/scale_ratio ## mark with middle position, so step/2
    intensity_r=[]
    for bound in lower_bounds:
        bound_up=bound+step
        mask = np.logical_and(radius >= bound, radius <= bound_up)
        avg_intensity=np.sum(heatmap*mask)/np.sum(mask)
        intensity_r.append(avg_intensity)
    intensity_r=np.array(intensity_r)
    print(intensity_r)
    plt.plot(micron_list,intensity_r,'.')
    #plt.ylim(0,0.3)