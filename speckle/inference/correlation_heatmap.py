import seaborn as sns
import numpy as np
import pickle
import matplotlib.pyplot as plt

def get_correlation(tauidx,loadtRange,t_range, px, corr_maps): #return seconds
    px_corrmap=corr_maps.GetCorrTimetrace(px,zRange=loadtRange)
    sum_correlation=0
    for t in range(0,len(t_range),1):
        sum_correlation=sum_correlation+px_corrmap[tauidx,t]
        print(px_corrmap[tauidx,t])
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
    
#def plot_radius_distribution()