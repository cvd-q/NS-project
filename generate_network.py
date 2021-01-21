import networkx as nx
import itertools as it
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
from random import shuffle

class Social_Net():

        def __init__(self, complete_net=False, n=1000):
                #initializes the graph and sets the parameters to default
                self.G = nx.Graph()
                self.set_parameters()
                if complete_net:
                        self.start_network(n)
                        #self.workplace_BA()
                        #self.workplace_random()
                        #self.interaction()
                        #self.social()
                        #print 'Classic network of size', n, 'created'
                else:
                        #print 'Network initialized. Please use function start_network to make the network'
                        pass

        def start_network(self, n):
                self.family_graph(n, self.G)
                self.workplace_BA()
                self.workplace_random()
                self.interaction() #lockdown interaction with essential workers
                self.social()
                #print 'Graph constructed'

        def return_graph(self):
                return self.G

        def family_graph(self, n, G):
                #print 'Size of the network is', n
                 fa = []     #list of families
                 for j, val in enumerate([int(round(i*n)) for i in self.family_sizes]):
                         for k in range(int(round(val/(j+1)))):
                                 fa.append(self.add_clique(j+1, G)) #denotes also workers
                 #print ('Family!')
                 #self.degree_histogram(G=self.G, hist_file='family_hist.png', loglog_file='family_log.png')
                 return fa

        def workplace_BA(self):
                #creates a scale-free network on non-essential workers
                self.n_work = 0
                self.working_nodes = []
                for n in self.G.nodes():
                        if self.G.nodes[n]['working'] and not self.G.nodes[n]['essential']:
                                self.n_work+=1
                                self.working_nodes.append(n)
                BAG = nx.barabasi_albert_graph(self.n_work, self.ba_degree)
                #print ('Workplace BA!')
                #self.degree_histogram(G=BAG, hist_file='workBA_hist.png', loglog_file='workBA_log.png')
                #map the BAG to the actual network edges
                for pair in list(BAG.edges()):
                        self.G.add_edge(self.working_nodes[pair[0]], self.working_nodes[pair[1]], lockdown=False)
                #print 'Printing edges of the workplace BA network'
                #print BAG.edges()
                return None

        def workplace_BA_trigger(self):
                BAG = nx.barabasi_albert_graph(self.n_work, self.ba_degree)
                for pair in list(BAG.edges()):
                        self.G.add_edge(self.working_nodes[pair[0]], self.working_nodes[pair[1]], lockdown=False)

        def workplace_random(self):
                #creates a random network on essential workers
                self.n_ess = 0
                self.essential_nodes = []
                for n in self.G.nodes():
                        if self.G.nodes[n]['essential']:
                                self.n_ess+=1
                                self.essential_nodes.append(n)
                #print 'Number of essential people is', n_ess
                #ER = nx.erdos_renyi_graph(n_ess, self.essential_connection)
                m = int(float(self.n_ess*self.rand_degree)/2)
                #print ('m is', m)
                ER = nx.gnm_random_graph(self.n_ess, m)
                #print ('Workplace Random!')
                #self.degree_histogram(G=ER, hist_file='workRand_hist.png', loglog_file='workRand_log.png')
                #map the ER to the actual network edges
                for pair in list(ER.edges()):
                        self.G.add_edge(self.essential_nodes[pair[0]], self.essential_nodes[pair[1]], lockdown=True)
                #print 'Printing edges of the workplace random network'
                #print ER.edges()

        def workplace_random_trigger(self):
                m = int(float(self.n_ess * self.rand_degree) / 2)
                ER = nx.gnm_random_graph(self.n_ess, m)
                for pair in list(ER.edges()):
                        self.G.add_edge(self.essential_nodes[pair[0]], self.essential_nodes[pair[1]], lockdown=True)

        def interaction(self):
                #add edges that represent interactions of everyone with essential workers
                for n in self.G.nodes():
                        connects = np.random.choice(np.arange(0,2), p=[1-self.interaction_prob, self.interaction_prob])
                        if connects:
                                m = random.choice(self.essential_nodes)
                                self.G.add_edge(n, m, lockdown=True)

                #print ('interaction!')
                #self.degree_histogram(G=newG, hist_file='inter_hist.png', loglog_file='inter_log.png')

        def social(self):
                allnodes = self.G.nodes()
                allpairs = list(it.combinations(allnodes, 2))
                select = random.sample(allpairs, k=int(self.social_prob*len(allpairs)))
                #print ('Social interaction will add', len(select), 'edges')
                for i,j in select:
                        self.G.add_edge(i, j, lockdown=False)
                #print ('Social!')
                #self.degree_histogram(G=socG, hist_file='soc_hist.png', loglog_file='soc_log.png')

        def set_parameters(self, family_sizes=[0.333, 0.271, 0.193, 0.151, 0.04, 0.013],
                           workrate=0.5,essential=0.2, ba_degree=4, essential_connection=0.6,
                           interaction_prob=0.20, social_prob=0.01, rand_degree=3):
                self.workrate = workrate
                self.essential = essential
                self.family_sizes = family_sizes
                self.ba_degree = ba_degree
                self.essential_connection = essential_connection
                self.interaction_prob = interaction_prob
                self.social_prob = social_prob
                self.rand_degree = rand_degree

        def set_parameters_part_lock(self, ba_degree=2, interaction_prob=0.08, social_prob=0.003, rand_degree=1):
                self.ba_degree = ba_degree
                self.interaction_prob = interaction_prob
                self.social_prob = social_prob
                self.rand_degree = rand_degree

        def return_parameters(self):
                pardict = {}
                pardict['workrate'] = self.workrate
                pardict['essential'] = self.essential
                pardict['family_sizes'] = self.family_sizes
                pardict['ba_degree'] = self.ba_degree
                pardict['essential_connection'] = self.essential_connection
                pardict['interaction_prob'] = self.interaction_prob
                pardict['social_prob'] = self.social_prob
                pardict['rand_degree'] = self.rand_degree
                return pardict

        def add_clique(self, clique_size, G):
                #adds a clique of size n to graph G
                if G.nodes():
                        l = max(G.nodes)+1
                else:
                        l = 0
                #print 'l is', l 
                works = []
                for k in range(clique_size):
                        works.append(np.random.choice(np.arange(0,2), p=[1-self.workrate, self.workrate]))
                nodes_to_add = [l+i for i in range(clique_size)]
                for i in range(clique_size):
                        if works[i]:
                                ess = np.random.choice(np.arange(0,2), p=[1-self.essential, self.essential])
                        else:
                                ess = 0
                        #print 'ess is', ess
                        G.add_node(l+i, working=works[i], essential=ess)
                #G.add_nodes_from(nodes_to_add)         #add nodes l, l+1, l+2, l+3 (for n=4) to the graph
                #add edges between all these pairs
                for i,j in list(it.combinations(nodes_to_add, 2)):
                        #print 'adding edge between', i, 'and', j
                        G.add_edge(i,j, lockdown=True)
                return nodes_to_add

        def degree_histogram(self, G=None, hist_file='temp_hist.png', loglog_file='temp_hist_log.png'):
                if not G:
                        G = self.G
                degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
                #print "Degree sequence", degree_sequence
                degreeCount = collections.Counter(degree_sequence)
                degreeCount = sorted(degreeCount.items())
                #print (type(degreeCount))
                deg, cnt = zip(*degreeCount)
                #print ('deg', deg)
                #print ('cnt', cnt)

                fig, ax = plt.subplots()
                plt.bar(deg, cnt)

                plt.title("Degree Histogram")
                plt.ylabel("Count")
                plt.xlabel("Degree")
                #ax.set_xticks([d + 0.4 for d in deg])
                #ax.set_xticklabels(deg)
                plt.savefig(hist_file, dpi=500)
                plt.close()

                if loglog_file:
                        fig, ax = plt.subplots()
                        #plt.loglog(deg, cnt)
                        plt.plot(deg, cnt, 'ko', markersize=2)
                        ax.set_xscale('log')
                        ax.set_yscale('log')
                        plt.title("Degree loglog plot")
                        plt.ylabel("P(k)")
                        plt.xlabel("Degree (k)")
                        plt.savefig(loglog_file, dpi=500)
                        plt.close()





class Dynamic_Social_Net(Social_Net):
        def __init__(self, n = 1000, default = True): #set times only for testing or generating nets
                if default:
                        self.G = nx.Graph()
                        #self.set_parameters(family_sizes=[0.333, 0.271, 0.193, 0.151, 0.04, 0.013], workrate=0.5,
                                            #essential=0.2, ba_degree=4, essential_connection=0.6,
                                            #interaction_prob=0.20, social_prob=0.01, rand_degree=3) #my choice
                        self.set_parameters()
                        self.families = self.family_graph(n, self.G)    #added egdes between members
                        self.workplace_BA() #no changes??????
                        self.random_enumerate_families()
                        self.lock = False
                else:
                        pass #complete?

        def random_enumerate_families(self):
                l = len(self.families)
                n_f = list(range(l+1))[1:]
                shuffle(n_f)
                i = 0
                for f in self.families:
                        for n in f:
                                self.G.nodes[n]['family'] = n_f[i]
                        i += 1

        def run_net(self, times = 10):
                # FUNCTION CALL ORDER IS IMPORTANT!
                self.G.clear_edges()
                for _, nodes_to_add in enumerate(self.families):
                        for i, j in list(it.combinations(nodes_to_add, 2)):
                                self.G.add_edge(i, j, lockdown=True)  # edges between family members
                # if self.lock == False:
                #         self.workplace_BA() it makes more sense?????
                self.workplace_random()
                self.interaction()  # lockdown interaction with essential workers
                if self.lock == False:
                        self.social()

                for t in range(times-1):
                        self.trigger()

        def trigger(self):
                # FUNCTION CALL ORDER IS IMPORTANT!
                self.G.clear_edges()
                for _, nodes_to_add in enumerate(self.families):
                        for i,j in list(it.combinations(nodes_to_add, 2)):
                                self.G.add_edge(i,j, lockdown=True)     #edges between family members
                # if self.lock == False:
                #         self.workplace_BA_trigger() it makes more sense?????
                self.workplace_random_trigger()
                self.interaction() #lockdown interaction with essential workers. =trigger
                if self.lock == False:
                        self.social()
                
        def return_families(self):
                return self.families

        def set_lock(self, lock):
                self.lock=lock


if __name__ == '__main__':
        My = Social_Net(complete_net=False)
        My.set_parameters()
        My.start_network(10000)
        #My.workplace_BA()
        #My.workplace_random()
        #My.interaction()
        #My.social()
        G = My.return_graph()
        print ('Overall')
        My.degree_histogram(loglog_file='temp_log.png')
        #print G.nodes(data=True)
        #print G.edges()
        #nx.draw(G, node_size=10)
        #plt.show()
        #nx.write_graphml(G, 'tenk_net.graphml')
