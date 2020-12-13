from Gill_Sim import Gill_Sim
import csv
from spread import draw_curve
import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import statistics
from generate_network import Dynamic_Social_Net


def test_average(start='2020-10-01', end='2020-11-01', N = 30, days = 30):
    s = Gill_Sim(Dynamic_Social_Net(n=1000))
    datacum = {'infected' : [[] for i in range(days)]}
    avgdict = {'infected' : [0 for i in range(days)]}
    stdict = {'infected' : [0 for i in range(days)]}
    for k in range(N):
        print('Start: ',k)
        s.run(30)
        print('Done: ', k)
        temp = s.return_data()['infected'][1:]
        for n in range(days):
            datacum['infected'][n].append(temp[n])
        s.reset() # ONLY EDGES CHANGE
            
    for n in range(days):
        avgdict['infected'][n] = statistics.mean(datacum['infected'][n])
        stdict['infected'][n] = statistics.stdev(datacum['infected'][n])
    c = []
    with open('data_2020_12.csv', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            c.append(row)
    inf = c[0].index('diff_totale_casi')
    s = [row[0] for row in c[1:]].index(start)
    e = [row[0] for row in c[1:]].index(end)
    real_data = {'infected':[int(i[inf]) for i in c[s:e+1]]}

    fig, ax = plt.subplots(2,1)
    x = [i+1 for i in range(len(real_data['infected']))]
    ax[0].plot(x, real_data['infected'], 'r', linewidth=1)
    x = [i+1 for i in range(len(avgdict['infected']))]
    ax[1].plot(x, avgdict['infected'], 'r', linewidth=1)
    ax[1].fill_between(x, [b-a for a,b in zip(stdict['infected'], avgdict['infected'])],
                    [b+a for a,b in zip(stdict['infected'], avgdict['infected'])],
                    color='r', alpha=0.1)
    plt.savefig('test_average', dpi=500)
    plt.close()


print('########### START #################')
test_average()









    
