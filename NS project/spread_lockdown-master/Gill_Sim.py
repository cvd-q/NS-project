from scipy import stats
import numpy as np
import networkx as nx
from generate_network import Dynamic_Social_Net
from spread import draw_curve
import matplotlib.pyplot as plt
import csv

class Gill_Sim():
    def __init__(self, d_G = Dynamic_Social_Net(n=1000), n_i = 3, default = True):
        if default:
            self.d_G = d_G
            self.N_I = n_i
            self.FAMILIES = self.d_G.return_families()
            self.n_i = self.N_I #CURRENT infected
            G = self.d_G.return_graph()
            for n in list(G.nodes()):
                G.nodes[n]['status'] = 'S'
                G.nodes[n]['inf_days'] = 0 ###
                G.nodes[n]['qua_days'] = 0 ###
            self.n_r = 0 #recovered
            self.n_d = 0 #death
            self.n_q = 0  #CURRENT quarantine ###
            self.m_r = [] #recovered nodes list
            self.m_d = [] #death list
            self.m_i = [] #list of infected nodes
            self.m_i = np.random.choice(list(G.nodes), self.n_i)
            self.m_q = []   #quarantine list ###
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
            G.nodes[n]['inf_days'] = 0 ###
            G.nodes[n]['qua_days'] = 0 ###
        self.n_r = 0 #recovered
        self.n_d = 0 #death
        self.n_q = 0  #CURRENT quarantine ###
        self.m_r = [] #recovered nodes list
        self.m_d = [] #death list
        self.m_i = [] #list of infected nodes
        self.m_i = np.random.choice(list(G.nodes), self.n_i)
        self.m_q = []   #quarantine list ###
        for n in self.m_i:
            G.nodes[n]['status'] = 'I'
        self.data = {'susceptible':[len(G.nodes())-self.n_i], 'infected':[None], 'recovered_cum':[None],
                     'dead_cum':[None], 'total_inf':[self.n_i]}

    def set_params(self, mu = 0.03, beta = 0.01, gamma = 0.005, qua = 0.03, BEFORE_QUA = 8, AFTER_QUA = 10): #mu=recovery rate, beta=infection rate
        # gamma = death rate, qua = quarantine rate, before_qua = n days before quarantine, after_qua = n days before exiting from quarantine
        # ATTENTION TO HOW THE PARAMS ARE USED IN RUN() FUNCTION! 
        self.mu = mu
        self.beta = beta
        self.gamma = gamma
        self.qua = qua ###
        self.BEFORE_QUA = BEFORE_QUA###
        self.AFTER_QUA = AFTER_QUA ###
        
    def run(self, T_sim = 30):
        #time = 10 ###
        Mu = self.mu #cumulative recovery rate
        Gamma = self.gamma #cumulative death rate
        Qua = self.qua #cumulative quarantine rate ###
        tau = np.random.exponential(1) #normalized
        G = self.d_G.return_graph()
        # when a member of one family is quarantined also other members are quarantined ###
        l_r = [] # from Q to R (one can be recovered when is in quarantine)###
        l_s = [] # from Q to S (family members quarantine because one among them is infected)###
        for k in self.data.keys():
            self.data[k] += [0]*T_sim #for data visualization
        for t in range(T_sim):
            ###contactList = self.d_G.run_net(1)
            self.d_G.run_net(1)
            m_SI = [] #S nodes in contact with I nodes
            #only people was in contact with infected and who is not recovered or dead or quarantined
            for i in self.m_i:
                for n in G.neighbors(i):
                    if G.nodes[n]['status'] == 'S': m_SI.append(n)
##            for (i, j) in contactList:
##                if i in self.m_i and i not in self.m_q and j not in self.m_i and j not in self.m_d and j not in self.m_r: ###
##                    m_SI.append(j)
##                elif j in self.m_i and j not in self.m_q and i not in self.m_i and i not in self.m_d and i not in self.m_r: ###
##                    m_SI.append(i)
            Beta = self.beta * len(m_SI) #cumulative infection rate
            Lambda = Mu + Beta + Gamma + Qua #cumulative transition rate ###      
            # check transition
            if Lambda<tau:  #no transition
                tau -= Lambda
            else: #transition occurred
                r = 1 #remaining fraction of time-step
                while r * Lambda >= tau:
                    z = np.random.choice(['I','R','D', 'Q'], 1,
                                         p=[Beta/Lambda, Mu/Lambda, Gamma/Lambda, Qua/Lambda]) ###
                    if z == 'I' and len(m_SI):
                        m = int(np.random.choice(m_SI, 1))
                        G.nodes[m]['status'] = 'I' #node m infected
                        self.m_i = np.append(self.m_i, m)
                        self.n_i += 1
                        G.nodes[m]['inf_days'] = 0
                        #Mu += self.mu ###
                        #Gamma += self.gamma ###  
                        self.data['infected'][t+1] += 1
                    #else:   #recovery transition
                    elif z == 'R': ### recovery transition
                        sp = False #
                        c = int(np.random.choice([0,1], 1))
                        if c: #0 = infected list, 1 = quarantined list
                            i_q = list(set(self.m_q) and set(self.m_i)) ### who are infected and quarantined
                            if len(i_q): 
                                m = int(np.random.choice(i_q, 1)) ###
                                l_r.append(m) ###the infected ready to exit from quarantine 
                            else:
                                sp = True #everyone in quarantine list is S
                        elif c==0 or sp: #SPONTANEOUS recovery
                            if len(self.m_i):
                                m = int(np.random.choice(list(self.m_i), 1))
                                G.nodes[m]['status'] = 'R' #node m recovered
                                self.m_i = np.delete(self.m_i, np.argwhere(self.m_i==m)) #remove m
                                self.n_i -= 1
                                self.m_r.append(m)
                                self.n_r +=1
                        if Mu>self.mu:  Mu -= self.mu
                        if Gamma>self.gamma:    Gamma -= self.gamma
                    elif z == 'D' and self.n_q: #death transition: one has to be quarantined (hospitalized) before death
                        ###if len(self.m_i):
                        i_q = list(set(self.m_q) and set(self.m_i)) ### who are infected and quarantined
                        if len(i_q): 
                            m = int(np.random.choice(i_q, 1)) ###
                            G.nodes[m]['status'] = 'D' #node m dies
                            self.m_q = np.delete(self.m_q, np.argwhere(self.m_q==m)) #remove m###
                            self.n_q -=1 ###
                            self.n_d +=1
                            #Mu -= self.mu 
                            if Gamma>self.gamma:  Gamma -= self.gamma
                            self.m_d.append(m)
                    else: #quarantin transition ###
                        if self.n_i:
                            m = int(np.random.choice(self.m_i, 1)) #node m quarantined
                            G.nodes[m]['status'] = 'Q'
                            G.nodes[m]['qua_days'] = 0
                            self.m_q = np.append(self.m_q, m)
                            self.n_q += 1
                            for f in self.FAMILIES: #quarantine family members (ONLY close contacts)
                                if m in f:
                                    i = f.index(m)
                                    for n in f:
                                        if n != i: #that is not the infected node
                                            if G.nodes[n]['status']!='D' and G.nodes[n]['status']!='Q':
                                                G.nodes[n]['status'] = 'Q' #member n is quarantined
                                                G.nodes[m]['qua_days'] = 0
                                                self.m_q = np.append(self.m_q, n)
                                                self.n_q += 1
                                                if G.nodes[n]['status'] == 'S': l_s += n 
                                                if G.nodes[n]['status'] == 'R': l_r += n
                                                
                            if Qua>self.qua:  Qua -= self.qua
                            Mu += self.mu
                            Gamma += self.gamma
 
                    r -= tau/Lambda #remaining time fraction
                    ##### REDO, time is not expired yet #####
                    m_SI = [] #S nodes in contact with I nodes
                    for i in self.m_i:
                        for n in G.neighbors(i):
                            if G.nodes[n]['status'] == 'S': m_SI.append(n)
##                    for (i, j) in contactList:
##                        if i in self.m_i and i not in self.m_q and j not in self.m_i and j not in self.m_d and j not in self.m_r: ###
##                            m_SI.append(j)
##                        elif j in self.m_i and j not in self.m_q and i not in self.m_i and i not in self.m_d and i not in self.m_r: ###
##                            m_SI.append(i)
                    l_SI = len(m_SI)
                    Beta = self.beta * l_SI
                    Lambda = Mu + Beta + Gamma + Qua
                    tau = np.random.exponential(1)
                    ##### REDO, time is not expired yet #####
            ###Day+1. Add probability to Qua only if nodes are infected since many days before.
            for n in self.m_i:
                G.nodes[n]['inf_days'] += 1
                if G.nodes[n]['inf_days']>self.BEFORE_QUA and G.nodes[n]['status'] == 'I':  Qua +=self.qua #G.nodes[n]['status'] != 'Q'
            for n in self.m_q:
                G.nodes[n]['qua_days'] += 1
                if G.nodes[n]['qua_days']>self.AFTER_QUA: #end quarantine
                    if n in l_s:
                        G.nodes[n]['status']='S'
                        l_s.remove(n)
                        self.m_q = np.delete(self.m_q, np.argwhere(self.m_q==n)) #remove n
                        self.n_q -= 1
                    if n in l_r:
                        G.nodes[n]['status']='R'
                        l_r.remove(n)
                        self.n_r +=1
                        self.m_r.append(m)
                        self.m_q = np.delete(self.m_q, np.argwhere(self.m_q==n)) #remove n
                        self.n_q -= 1
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
##            print('CURRENT N_quarante: ', self.n_q)
##            print('Total_infected: ', self.data['total_inf'][t+1])
##            print('\n')
##            
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
