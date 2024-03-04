import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

colors_ = ['#00BFFF', '#DC143C', '#3CB371', '#008B8B', 
           '#8B0000', '#FFD700', '#FF6347', '#2E8B57',
           '#EEE8AA', '#F0E68C']




def plotting_violins(config, geo, scores, save_path):
    scores_df = pd.read_csv(scores)
    scores_df.columns=["geography", "lin", "lstm", "xg", "rf", "arima", "mean","ens"]
    scores_df = scores_df.loc[scores_df.geography==geo,]
    save_name = save_path+"violinplot_"+geo+ ".png"
    ax = sns.violinplot(data=scores_df.iloc[:,1:])
    plot_title="Violin plot of model mapes with {} days prediction in {}".format(config.size_PW,geo)
    ax.set(title=plot_title,ylabel="mape")
    plt.savefig(save_name)
    plt.clf()
    return(None)

def plotting_boxplots(config,geo,scores, save_path, ens=False):
    scores_df = pd.read_csv(scores)

    if ens:
        scores_df.columns=["geography", "lin", "lstm", "xg", "rf", "arima","mean","ens","selection","stacking"]
    else: 
        scores_df.columns=["geography", "lin", "lstm", "xg", "rf", "arima","mean","ens"]
    scores_df = scores_df.loc[scores_df.geography==geo,]
    save_name = save_path+ "boxplot_"+geo+".png"
    ax = sns.boxplot(data=scores_df.iloc[:,1:])
    plot_title="Boxplot of model mapes with {} days prediction in {}".format(config.size_PW,geo)
    ax.set(title=plot_title,ylabel="mape")
    plt.savefig(save_name)
    plt.clf()
    return(None)

def plotting_barplots(selection, save_path):
    
    selection.columns=["LR", "LSTM", "XG", "RF", "ARIMA"]
    
    colors = sns.color_palette("Set1", n_colors=len(selection.columns))
    sns.set_theme(style="whitegrid")
    ax = selection.plot(kind='bar', figsize=(10, 6), color=colors)

    plt.title('Number of model selections per region')
    plt.xlabel('Regions')
    plt.ylabel('Number of Selections')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Models')
    save_name=save_path+"barplots_selections.png"
    plt.savefig(save_name)
    plt.clf()  
    return(None)