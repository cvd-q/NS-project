from scipy import stats
import numpy as np
import networkx as nx
from generate_network import Dynamic_Social_Net
from spread import draw_curve
import matplotlib.pyplot as plt
import csv
import random

class Gill_Sim():
    def __init__(self, d_G = Dynamic_Social_Net(n=3000), n_i = 3, default = True):
        if default:
            self.d_G = d_G
            self.N_I = n_i
            self.FAMILIES = self.d_G.return_families()
            self.n_i = self.N_I #CURRENT infected
            G = self.d_G.return_graph()
            for n in list(G.nodes()):
                G.nodes[n]['status'] = 'S'
                G.nodes[n]['inf_days'] = 0 
                G.nodes[n]['qua_days'] = 0 
            self.n_r = 0 #recovered
            self.n_d = 0 #death
            self.n_q = 0  #CURRENT quarantine 
            self.m_r = [] #recovered nodes list
            self.m_d = [] #death list
            self.m_i = [] #list of infected nodes
            self.m_i = np.random.choice(list(G.nodes), self.n_i)
            self.m_q = []   #quarantine list 
            for n in self.m_i:
                G.nodes[n]['status'] = 'I'
            self.set_params()
            self.data = {'susceptible': [len(G.nodes())-self.n_i], 'infected':[None], 'recovered_cum':[None],
                         'dead_cum': [None], 'total_inf':[self.n_i], 'r0':[None]} ###
            # some data visualization
            #print(G.nodes.data())
        else:
            pass #complete!
    def reset(self):
        self.n_i = self.N_I
        G = self.d_G.return_graph()
        for n in list(G.nodes()):
            G.nodes[n]['status'] = 'S'
            G.nodes[n]['inf_days'] = 0 
            G.nodes[n]['qua_days'] = 0 
        self.n_r = 0 #recovered
        self.n_d = 0 #death
        self.n_q = 0  #CURRENT quarantine 
        self.m_r = [] #recovered nodes list
        self.m_d = [] #death list
        self.m_i = [] #list of infected nodes
        self.m_i = np.random.choice(list(G.nodes), self.n_i)
        self.m_q = []   #quarantine list 
        for n in self.m_i:
            G.nodes[n]['status'] = 'I'
        self.data = {'susceptible':[len(G.nodes())-self.n_i], 'infected':[None], 'recovered_cum':[None],
                     'dead_cum':[None], 'total_inf':[self.n_i], 'r0':[None]}###
        self.init_run()

    def set_params(self, mu = 0.003, beta = 0.012, gamma = 0.001, qua = 0.003, BEFORE_QUA = [8,14], AFTER_QUA = [10,20]): #mu=recovery rate, beta=infection rate
        # gamma = death rate, qua = quarantine rate, before_qua = n days before quarantine, after_qua = n days before exiting from quarantine
        # ATTENTION TO HOW THE PARAMS ARE USED IN RUN() FUNCTION! 
        self.mu = mu
        self.beta = beta
        self.gamma = gamma
        self.qua = qua 
        self.BEFORE_QUA = BEFORE_QUA
        self.AFTER_QUA = AFTER_QUA
        
    def init_run(self, restart=False, Mu=0, Gamma=0, Qua=0 , l_r=[], l_s=[]): #EXECUTE IT BEFORE RUN!
        if restart == False:
            self.Mu = self.mu #cumulative recovery rate
            self.Gamma = self.gamma #cumulative death rate
            self.Qua = self.qua #cumulative quarantine rate 
            # when a member of one family is quarantined also other members are quarantined 
            self.l_r = [] # from Q to R (one can be recovered when is in quarantine)
            self.l_s = [] # from Q to S (family members quarantine because one among them is infected)
            self.time = 0 # count days since first run
        else:
            self.Mu = Mu
            self.Gamma = Gamma
            self.Qua = Qua
            self.l_r = l_r
            self.l_s = l_s

    
    def run(self, T_sim = 30): 
        #self.init_run()
        for k in self.data.keys():
            self.data[k] += [0]*T_sim #for data visualization
        tau = np.random.exponential(1) #normalized
        for t in range(self.time, self.time + T_sim):
            self.d_G.run_net(1)
            G = self.d_G.return_graph()
            m_SI = [] #S nodes in contact with I nodes
            #only people was in contact with infected and who is not recovered or dead or quarantined
            for i in self.m_i:
                if G.nodes[i]['status'] == 'I': # if the node is not in quarantine
                    for n in G.neighbors(i):
                        if G.nodes[n]['status'] == 'S': m_SI.append(n)
            Beta = self.beta * len(m_SI) #cumulative infection rate
            Lambda = self.Mu + Beta + self.Gamma + self.Qua #cumulative transition rate
            l_r0 = len(self.m_i) #useful for computing the R0(t) (number of infected nodes at the beginning of the day)
            # check transition
            if Lambda<tau:  #no transition
                tau -= Lambda
            else: #transition occurred
                r = 1 #remaining fraction of time-step
                while r * Lambda >= tau:
                    z = np.random.choice(['I','R','D', 'Q'], 1,
                                         p=[Beta/Lambda, self.Mu/Lambda, self.Gamma/Lambda, self.Qua/Lambda]) 
                    if z == 'I' and len(m_SI):
                        m = int(np.random.choice(m_SI, 1))
                        G.nodes[m]['status'] = 'I' #node m infected
                        self.m_i = np.append(self.m_i, m)
                        self.n_i += 1
                        G.nodes[m]['inf_days'] = 0 
                        self.data['infected'][t+1] += 1
                    #else:   #recovery transition
                    elif z == 'R': ### recovery transition
                        sp = False #
                        c = int(np.random.choice([0,1], 1))
                        if c: #0 = infected list, 1 = quarantined list
                            i_q = list(set(self.m_q) and set(self.m_i) - set(self.l_r)) # who are infected and quarantined
                            if len(i_q): 
                                m = int(np.random.choice(i_q, 1))
                                self.l_r.append(m) #the infected ready to exit from quarantine
                                self.m_i = np.delete(self.m_i, np.argwhere(self.m_i == m)) #???????
                                self.n_i -= 1 #????????
                                self.m_r.append(m) #???????????
                                self.n_r += 1  #??????????????
                            else:
                                sp = True #everyone in quarantine list is S
                        elif c==0 or sp: #SPONTANEOUS recovery
                            if len(self.m_i):
                                # l_i = list(set(self.m_i)-set(self.m_q)) #infected node that is not in quarantine?????
                                if len(self.m_i):
                                    m = int(np.random.choice(self.m_i, 1))
                                    G.nodes[m]['status'] = 'R' #node m recovered
                                    self.m_i = np.delete(self.m_i, np.argwhere(self.m_i==m)) #remove m
                                    self.n_i -= 1
                                    self.m_r.append(m)
                                    self.n_r +=1
                        if self.Mu>self.mu:  self.Mu -= self.mu
                        if self.Gamma>self.gamma:    self.Gamma -= self.gamma
                    elif z == 'D' and self.n_q: #death transition: one has to be quarantined (hospitalized) before death
                        i_q = list(set(self.m_q) and set(self.m_i) )#- set(self.l_r)) # who are infected and quarantined??????
                        if len(i_q): 
                            m = int(np.random.choice(i_q, 1))
                            G.nodes[m]['status'] = 'D' #node m dies
                            self.m_q = np.delete(self.m_q, np.argwhere(self.m_q==m)) #remove m
                            self.m_i = np.delete(self.m_i, np.argwhere(self.m_i == m))  # remove m
                            self.n_q -=1
                            self.n_i -=1
                            self.n_d +=1
                            self.m_d.append(m)
                            if self.Gamma>self.gamma:  self.Gamma -= self.gamma

                    else: #quarantin transition
                        m_i = list(set(self.m_i) - set(self.m_q)) #node infected but no in quarantine
                        if len(m_i):
                            m = int(np.random.choice(m_i, 1)) #node m quarantined
                            G.nodes[m]['status'] = 'Q'
                            G.nodes[m]['qua_days'] = 0
                            self.m_q = np.append(self.m_q, m)
                            self.n_q += 1
                            for f in self.FAMILIES: #quarantine family members (ONLY close contacts)
                                if m in f:
                                    for n in f:
                                        if n != m: #that is not the infected node ###########################
                                            if G.nodes[n]['status']=='S' or G.nodes[n]['status']=='R':
                                                if G.nodes[n]['status'] == 'S':
                                                    self.l_s.append(n)
                                                elif G.nodes[n]['status'] == 'R':
                                                    self.l_r.append(n)
                                                G.nodes[n]['status'] = 'Q' #member n is quarantined
                                                G.nodes[n]['qua_days'] = 0
                                                self.m_q = np.append(self.m_q, n)
                                                self.n_q += 1
                                                
                            if self.Qua>self.qua:  self.Qua -= self.qua
                            self.Mu += self.mu
                            self.Gamma += self.gamma
 
                    r -= tau/Lambda #remaining time fraction
                    ##### REDO, time is not expired yet #####
                    m_SI = [] #S nodes in contact with I nodes
                    for i in self.m_i:
                        if G.nodes[i]['status'] == 'I':
                            for n in G.neighbors(i):
                                if G.nodes[n]['status'] == 'S': m_SI.append(n)
                    l_SI = len(m_SI)
                    Beta = self.beta * l_SI
                    Lambda = self.Mu + Beta + self.Gamma + self.Qua
                    tau = np.random.exponential(1)
                    ##### REDO, time is not expired yet #####
            ###Day+1. Add probability to Qua only if nodes are infected since many days before.
            for n in self.m_i:
                G.nodes[n]['inf_days'] += 1
                if G.nodes[n]['inf_days']>random.randint(self.BEFORE_QUA[0], self.BEFORE_QUA[1]) and G.nodes[n]['status'] == 'I':
                    self.Qua +=self.qua #G.nodes[n]['status'] != 'Q'
            m_q = self.m_q.copy()
            for n in m_q:
                G.nodes[n]['qua_days'] += 1
                if G.nodes[n]['qua_days']>random.randint(self.AFTER_QUA[0], self.AFTER_QUA[1]): #end quarantine
                    if n in self.l_s:
                        G.nodes[n]['status']='S'
                        self.l_s.remove(n)
                        self.m_q = np.delete(self.m_q, np.argwhere(self.m_q==n)) #remove n
                        self.n_q -= 1
                    elif n in self.l_r:
                        G.nodes[n]['status'] = 'R'
                        self.l_r.remove(n)
                        # self.n_r +=1 ???????
                        # self.m_r.append(n) ??????????
                        self.m_q = np.delete(self.m_q, np.argwhere(self.m_q==n)) #remove n
                        self.n_q -= 1
            self.data['susceptible'][t+1] = self.data['susceptible'][t] - self.data['infected'][t+1]
            if l_r0:
                self.data['r0'][t+1] = self.data['infected'][t+1] / l_r0
            else:
                self.data['r0'][t+1] = 0
            self.data['recovered_cum'][t+1] = self.n_r
            self.data['dead_cum'][t+1] = self.n_d
            self.data['total_inf'][t+1] = self.data['total_inf'][t] + self.data['infected'][t+1]
            # some data visualization
            if t % 15 == 0:
                print('Iteration: ', t)
                #print('N_infected: ', self.n_i, 'Infected: ', list(self.m_i))
                print('N_infected: ', self.data['infected'][t+1])
                #print('N_recovered: ', self.n_r, 'Recovered: ', list(self.m_r))
                if t == 0:
                    print('N_recovered: ', self.data['recovered_cum'][t+1], 'Increment: ',
                            self.data['recovered_cum'][t+1])
                    print('N_death: ', self.data['dead_cum'][t+1], 'Increment: ',
                            self.data['dead_cum'][t+1])
                        
                else:
                    print('N_recovered: ', self.data['recovered_cum'][t+1], 'Increment: ',
                          self.data['recovered_cum'][t+1] - self.data['recovered_cum'][t])   
                    print('N_death: ', self.data['dead_cum'][t+1], 'Increment: ',
                          self.data['dead_cum'][t+1] - self.data['dead_cum'][t])
                print('CURRENT N_quarantined: ', len(self.m_q), ' ', self.n_q)#self.n_q
                print('Total_infected: ', self.data['total_inf'][t+1])
                print('CURRENT R0: ', self.data['r0'][t+1])
                print('\n')
        self.time += T_sim

    def return_d_graph_families(self):
        return self.d_G.return_families()

    def return_d_graph(self):
        return self.d_G
    
    def return_data(self):
        return self.data

class Gill_Sim_lock(Gill_Sim):
    def __init__(self, d_G = Dynamic_Social_Net(n=1500), n_i = 3, default = True):
        super().__init__(d_G, n_i, default)
        
    def run_lock(self, T_sim = 50, s_lock = 10, e_lock = 20): #IMPROVE IT WITH LOCKDOWN LIST
        self.init_run()
        self.run(s_lock)
        print('####### START LOCK ##########')
        self.d_G.set_lock(lock=True)
        self.run(e_lock-s_lock)
        print('####### END LOCK ##########')
        self.d_G.set_lock(lock=False)
        self.run(T_sim-e_lock)

    def run_part_lock(self, T_sim=50, s_lock=10, e_lock=20):  # IMPROVE IT WITH LOCKDOWN LIST
        self.init_run()
        self.run(s_lock)
        print('####### START LOCK ##########')
        self.d_G.set_parameters_part_lock()
        self.run(e_lock - s_lock)
        print('####### END LOCK ##########')
        self.d_G.set_parameters()
        self.run(T_sim - e_lock)
        print(len(set(self.m_q)),' vs ',len(self.m_q))
        l=list(set(self.l_s) and set(self.l_r))
        print('l_s vs l_r: ', l)
        for i in l: print(G.nodes[i]['status'])


if __name__ == '__main__': #CHANGE TO Gill_Sim_lock()
    s = Gill_Sim_lock()
    #print('Families: ', s.return_d_graph_families())
    G = s.return_d_graph().return_graph()
    print('###########START############')
    s.run_part_lock(120, 16+7, 84+7)
    #s.run_lock(120, 16+7, 84+7) #DELAY 1 WEEK
    data = s.return_data()
    draw_curve(data, len(G), len(data['infected']))

    ###### Test with real data ###########
    c = []
    with open('data_2020_12.csv', newline='') as f:
        reader = csv.reader(f, delimiter=';')
        for row in reader:
            c.append(row)
    inf = c[0].index('diff_totale_casi')
    feb = [row[0] for row in c[1:]].index('2020-02-25')
    jun = [row[0] for row in c[1:]].index('2020-06-01')
    datadict = {'infected':[int(i[inf]) for i in c[feb:jun]]}

    fig, ax = plt.subplots(3,1)
    x = [i+1 for i in range(len(datadict['infected']))]
    ax[0].plot(x, datadict['infected'], 'r', linewidth=1)
    x = [i+1 for i in range(len(data['infected']))]
    ax[1].plot(x, data['infected'], 'r', linewidth=1)
    ax[1].vlines(x=16, ymin=0, ymax=max(data['infected'][1:]), colors='black', ls='--', lw=2)
    ax[1].vlines(x=84, ymin=0, ymax=max(data['infected'][1:]), colors='green', ls='--', lw=2)
    x = [i+1 for i in range(len(data['infected'])-1)]
    INCUB_DAYS = 14
    l = [0]*INCUB_DAYS + data['infected'][1:len(data['infected'])-INCUB_DAYS]
    ax[2].plot(x, l, 'r', linewidth=1) # translated curve; incubation period
    plt.savefig('Test_1', dpi=500)
    plt.close()
