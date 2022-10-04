#Import relevant libraries
import numpy as np
import pandas as pd
pd.set_option('max_columns', 150)

import gc
import os

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

#configure plots from matplotlib
matplotlib.rcParams['figure.dpi'] = 180 #resolution
matplotlib.rcParams['figure.figsize'] = (8,6) #figure size

#set seaborn style for plots
sns.set_style('darkgrid')

color = sns.color_palette()

def bar_visual(df, xlabel,ylabel, title):
    fig, ax = plt.subplots(figsize=(15,8))
    ax = sns.barplot(x = df.index, y = df.values, color = color[1])
    ax.set_xlabel(f'{xlabel}')
    ax.set_ylabel(f'{ylabel}')
    ax.xaxis.set_tick_params(rotation=90, labelsize=10)
    ax.set_title(f'{title}')
    fig.savefig(f'{title}.png')


def count_viz(df,xlabel,ylabel,title):
    fig, ax = plt.subplots(figsize = (10,5))
    ax = sns.countplot(df, color = color[1])
    ax.set_xlabel(f'{xlabel}', size = 10 )
    ax.set_ylabel(f'{ylabel}', size = 10)
    ax.tick_params(axis = 'both', labelsize = 8)
    ax.set_title(f'{title}')
    fig.savefig(f'{title}.png')
    plt.show()

def read(root,path):
    return pd.read_csv(root + path + "/" + path + ".csv")

    
