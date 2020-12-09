from scipy import stats
import numpy as np
import networkx as nx
from generate_network import Dynamic_Social_Net
from spread import draw_curve
import matplotlib.pyplot as plt
import csv

class Gill_Sim():
    def __init__(self, d_G = Dynamic_Social_Net(n=1500), n_i = 3, default = True):
        if default:
            self.d_G = d_G
            self.N_I = n_i
            self.n_i = self.N_I #CURRENT infected
            G = self.d_G.return_graph()
            for n in list(G.nodes()):
                G.nodes[n]['status'] = 'S'
            self.n_r = 0 #recovered
            self.n_d = 0 #death
            self.m_r = [] #recovered nodes list
            self.m_d = [] #death list
            self.m_i = [] #list of infected nodes
            self.m_i = np.random.choice(list(G.nodes), self.n_i)
            for n in self.m_i:
                G.nodes[n]['status'] = 'I'
            self.set_params()
            self.data = {'susceptible':[len(G.nodes())-self.n_i], 'infected':[None], 'recovered_cum':[None],
                         'dead_cum':[None], 'total_inf':[self.n_i]}
            # some data visualization
            #print(G.nodes.data())
        else:
            pass #complete!

    def reset(self):
        self.n_i = self.N_I
        G = self.d_G.return_graph()
        for n in list(G.nodes()):
            G.nodes[n]['status'] = 'S'
        self.n_r = 0 #recovered
        self.n_d = 0 #death
        self.m_r = [] #recovered nodes list
        self.m_d = [] #death list
        self.m_i = [] #list of infected nodes
        self.m_i = np.random.choice(list(G.nodes), self.n_i)
        for n in self.m_i:
            G.nodes[n]['status'] = 'I'
        self.data = {'susceptible':[len(G.nodes())-self.n_i], 'infected':[None], 'recovered_cum':[None],
                     'dead_cum':[None], 'total_inf':[self.n_i]}

    def set_params(self, mu = 0.011, beta = 0.011, gamma = 0.001): #mu=recovery rate, beta=infection rate
        # gamma = death rate
        self.mu = mu
        self.beta = beta
        self.gamma = gamma
        
    def run(self, T_sim = 30):
        time = 10 #before there's no recovery 
        Mu = self.mu #cumulative recovery rate
        Gamma = self.gamma #cumulative death rate
        tau = np.random.exponential(1) #normalized
        G = self.d_G.return_graph()
        for k in self.data.keys():
            self.data[k] += [0]*T_sim #for data visualization
        for t in range(T_sim):
            contactList = self.d_G.run_net(1)
            m_SI = [] #S nodes in contact with I nodes
            #only people was in contact with infected and who is not recovered and dead
            for (i, j) in contactList:
                if i in self.m_i and j not in self.m_i and j not in self.m_d and j not in self.m_r:
                    m_SI.append(j)
                elif j in self.m_i and i not in self.m_i and i not in self.m_d and i not in self.m_r:
                    m_SI.append(i)
            l_SI = len(m_SI)
            Beta = self.beta * l_SI #cumulative infection rate
            Gamma = self.gamma * l_SI #cumulative death rate
            Lambda = Mu + Beta +Gamma #cumulative transition rate      
            # check transition
            if Lambda<tau:  #no transition
                tau -= Lambda
            else: #transition occurred
                r = 1 #remaining fraction of time-step
                while r * Lambda >= tau:
                    z = np.random.choice(['I','R','D'], 1,
                                         p=[Beta/Lambda, Mu/Lambda, Gamma/Lambda])
                    #z = float(np.random.uniform(0,Lambda - np.finfo(float).eps))
                    #if z<Beta: #infection transition
                    if z == 'I' and len(m_SI):
                        m = int(np.random.choice(m_SI, 1))
                        G.nodes[m]['status'] = 'I' #node m infected
                        self.m_i = np.append(self.m_i, m)
                        self.n_i += 1
                        Mu += self.mu
                        Gamma += self.gamma
                        self.data['infected'][t+1] += 1
                    #else:   #recovery transition
                    elif z == 'R' and len(self.m_i) and t>time:
                        m = int(np.random.choice(self.m_i, 1))
                        G.nodes[m]['status'] = 'R' #node m recovered
                        self.m_i = np.delete(self.m_i, np.argwhere(self.m_i==m)) #remove m
                        self.n_i -=1
                        self.n_r +=1
                        Mu -= self.mu
                        Gamma -= self.gamma
                        self.m_r.append(m)
                    else: #death transition
                        if len(self.m_i):
                            m = int(np.random.choice(self.m_i, 1))
                            G.nodes[m]['status'] = 'D' #node m dies
                            self.m_i = np.delete(self.m_i, np.argwhere(self.m_i==m)) #remove m
                            self.n_i -=1
                            self.n_d +=1
                            #Mu -= self.mu #INVARIANT???
                            Gamma -= self.gamma
                            self.m_d.append(m)
                            
                    r -= tau/Lambda #remaining time fraction
                    ##### REDO, time is not expired yet #####
                    m_SI = [] #S nodes in contact with I nodes
                    for (i, j) in contactList:
                        if i in self.m_i and j not in self.m_i and j not in self.m_d and j not in self.m_r:
                            m_SI.append(j)
                        elif j in self.m_i and i not in self.m_i and i not in self.m_d and i not in self.m_r:
                            m_SI.append(i)
                    l_SI = len(m_SI)
                    Beta = self.beta * l_SI
                    Gamma = self.gamma * l_SI
                    Lambda = Mu + Beta + Gamma
                    tau = np.random.exponential(1)
                    ##### REDO, time is not expired yet #####
            self.data['susceptible'][t+1] = self.data['susceptible'][t] - self.data['infected'][t+1]
            self.data['recovered_cum'][t+1] = self.n_r
            self.data['dead_cum'][t+1] = self.n_d
            self.data['total_inf'][t+1] = self.data['total_inf'][t] + self.data['infected'][t+1]
            # some data visualization
##            print('Iteration: ', t)
##            #print('N_infected: ', self.n_i, 'Infected: ', list(self.m_i))
##            print('N_infected: ', self.data['infected'][t+1])
##            #print('N_recovered: ', self.n_r, 'Recovered: ', list(self.m_r))
##            if t == 0:
##                print('N_recovered: ', self.data['recovered_cum'][t+1], 'Increment: ',
##                        self.data['recovered_cum'][t+1])
##                print('N_death: ', self.data['dead_cum'][t+1], 'Increment: ',
##                        self.data['dead_cum'][t+1])
##                    
##            else:
##                print('N_recovered: ', self.data['recovered_cum'][t+1], 'Increment: ',
##                      self.data['recovered_cum'][t+1] - self.data['recovered_cum'][t])
##            #print('N_death: ', self.n_d, 'death: ', list(self.m_d))   
##                print('N_death: ', self.data['dead_cum'][t+1], 'Increment: ',
##                      self.data['dead_cum'][t+1] - self.data['dead_cum'][t])
##            print('Total_infected: ', self.data['total_inf'][t+1])
##            print('\n')
            
    def return_d_graph_families(self):
        return self.d_G.return_families()

    def return_d_graph(self):
        return self.d_G
    
    def return_data(self):
        return self.data
    
if __name__ == '__main__':
    s = Gill_Sim()
    #print('Families: ', s.return_d_graph_families())
    G = s.return_d_graph().return_graph()
    print('###########START############')
    s.run()
    data = s.return_data()
    draw_curve(data, len(G), len(data['infected']))

    ###### Test with real data ###########
    c = []
    with open('data_2020_12.csv', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            c.append(row)
    inf = c[0].index('diff_totale_casi')
    sept = [row[0] for row in c[1:]].index('2020-10-01')
    nov = [row[0] for row in c[1:]].index('2020-11-01')
    datadict = {'infected':[int(i[inf]) for i in c[sept:nov+1]]}

    fig, ax = plt.subplots(2,1)
    x = [i+1 for i in range(len(datadict['infected']))]
    ax[0].plot(x, datadict['infected'], 'r', linewidth=1)
    x = [i+1 for i in range(len(data['infected']))]
    ax[1].plot(x, data['infected'], 'r', linewidth=1)
    plt.savefig('Test_1', dpi=500)
    plt.close()
