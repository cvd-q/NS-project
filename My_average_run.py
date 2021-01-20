from Gill_Sim import Gill_Sim_lock
import csv
import numpy as np
import networkx as nx
from scipy import stats
import matplotlib.pyplot as plt
import statistics
from generate_network import Dynamic_Social_Net

class Gill_plus_average(Gill_Sim_lock):
    def __init__(self, d_G = Dynamic_Social_Net(n=1000), n_i = 3, default = True):
        super().__init__(d_G, n_i, default)
    def average_run(self, start='2020-10-01', end='2020-11-01', N = 10, days = 50):
        # START HERE; COMPLETE WITH LOCK DAYS !!!!!!!!!!!!!!
        # days = each simulation days, start and end = real data!
        self.start = start
        self.end = end
        self.days = days
        self.datacum = {'infected' : [[] for i in range(days)], 'susceptible' : [[] for i in range(days)],
                        'dead_cum' : [[] for i in range(days)], 'total_inf' : [[] for i in range(days)],
                        'r0' : [[] for i in range(days)], 'recovered_cum' : [[] for i in range(days)]}
        self.avgdict = {'infected' : [0 for i in range(days)], 'susceptible' : [0 for i in range(days)],
                        'dead_cum' : [0 for i in range(days)], 'total_inf' : [0 for i in range(days)],
                        'r0' : [0 for i in range(days)], 'recovered_cum' : [0 for i in range(days)]}
        self.stdict = {'infected' : [0 for i in range(days)], 'susceptible' : [0 for i in range(days)],
                       'dead_cum' : [0 for i in range(days)], 'total_inf' : [0 for i in range(days)],
                       'r0' : [0 for i in range(days)], 'recovered_cum' : [0 for i in range(days)]}
        for k in range(N):
            print('Start: ',k)
            self.run_lock(days)
            print('Done: ', k)
            data = self.return_data()
            for k in data.keys():
                for n in range(days):
                    self.datacum[k][n].append(data[k][n+1])
            self.reset() # ONLY EDGES CHANGE
        self.test_draw_curves()

    def test_draw_curves(self):
        for n in range(self.days):
            for k in self.datacum.keys():
                self.avgdict[k][n] = statistics.mean(self.datacum[k][n])
                self.stdict[k][n] = statistics.stdev(self.datacum[k][n])
        c = []
        with open('data_2020_12.csv', newline='') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                c.append(row)
        inf = c[0].index('diff_totale_casi')
        s = [row[0] for row in c[1:]].index(self.start)
        e = [row[0] for row in c[1:]].index(self.end)
        real_data = {'infected':[int(i[inf]) for i in c[s:e+1]]}

        fig, ax = plt.subplots(4,1)
        x = [i+1 for i in range(len(real_data['infected']))]
        ax[0].plot(x, real_data['infected'], 'r', linewidth=1)
        
        x = [i+1 for i in range(len(self.avgdict['infected']))]
        ax[1].plot(x, self.avgdict['infected'], 'r', linewidth=1)
        ax[1].fill_between(x, [b-a for a,b in zip(self.stdict['infected'], self.avgdict['infected'])],
                    [b+a for a,b in zip(self.stdict['infected'], self.avgdict['infected'])],
                    color='r', alpha=0.1)
        
        x = [i+1 for i in range(len(self.avgdict['infected']))]
        INCUB_DAYS = 7 
        l = [0]*INCUB_DAYS + self.avgdict['infected'][:len(self.avgdict['infected'])-INCUB_DAYS]
        ax[2].plot(x, l, 'r', linewidth=1) # translated curve; incubation period
        l1 = [0]*INCUB_DAYS + self.stdict['infected'][:len(self.stdict['infected'])-INCUB_DAYS]
        ax[2].fill_between(x, [b-a for a,b in zip(l1, l)],
                    [b+a for a,b in zip(l1, l)],
                    color='r', alpha=0.1)
        
        base_col = ['b','g','r','c','m','y','k']
        i = 0
        if len(self.avgdict.keys())<len(base_col):
            for k in self.avgdict.keys():
                ax[3].plot(x, self.avgdict[k], base_col[i], linewidth=1)
                ax[3].fill_between(x, [b-a for a,b in zip(self.stdict[k], self.avgdict[k])],
                            [b+a for a,b in zip(self.stdict[k], self.avgdict[k])],
                            color=base_col[i], alpha=0.1)
                i += 1
        else:
            print('PLOT ERROR: number of colors not enough!')
        plt.savefig('test_average', dpi=500)
        plt.close()


#only for initial correctness test
def test_average(start='2020-10-01', end='2020-11-01', N = 30, days = 30): 
    s = Gill_Sim(Dynamic_Social_Net(n=1000))
    datacum = {'infected' : [[] for i in range(days)]}
    avgdict = {'infected' : [0 for i in range(days)]}
    stdict = {'infected' : [0 for i in range(days)]}
    for k in range(N):
        print('Start: ',k)
        s.run(days)
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
Gill_plus_average().average_run()









    
