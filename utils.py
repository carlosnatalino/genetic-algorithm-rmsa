# utils file with functions

import copy
import functools
import itertools
import math

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def get_modulation_format(modulations, length):
    for i in range(len(modulations)-1):
        if length > modulations[i+1]['maximum_length'] and length <= modulations[i]['maximum_length']:
#             print(length, i, modulations[i]['modulation'])
            return i
    if length <= modulations[-1]['maximum_length']:
#         print(length, len(modulations) - 1, modulations[len(modulations) - 1]['modulation'])
        return len(modulations) - 1

def read_txt_file(file, name):
    graph = nx.Graph(name=name) # DiGraph because we have two fibers (one each way) between any pair of nodes
    nNodes = 0
    nLinks = 0
    idEdge = 0
    nodes = []
    edges = []
    with open(file, 'r') as nodes_lines:
        for idx, line in enumerate(nodes_lines):
            if idx > 2 and idx <= nNodes + 2: # skip title line
                info = line.replace("\n", "").replace(",", ".").split("\t")
                graph.add_node(info[0], name=info[1], pos=(float(info[2]), float(info[3])))
                nodes.append(info[0])
            elif idx > 2 + nNodes and idx <= 2 + nNodes + nLinks: # skip title line
                info = line.replace("\n", "").split("\t")
                graph.add_edge(info[1], info[2], id=idEdge, weight=int(info[3]))
                idEdge += 1
                edges.append((info[1], info[2]))
#                 graph.add_edge(info[2], info[1], id=idEdge, weight=int(info[3]))
#                 idEdge += 1
#                 edges.append((info[2], info[1]))
            elif idx == 1:
                nNodes = int(line)
            elif idx == 2:
                nLinks = int(line)
    graph.graph['nodes'] = nodes
    graph.graph['edges'] = edges
    return graph

def get_k_shortest_paths(G, source, target, k, weight='weight'):
    """
    Method from https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.simple_paths.shortest_simple_paths.html#networkx.algorithms.simple_paths.shortest_simple_paths
    """
    return list(itertools.islice(nx.shortest_simple_paths(G, source, target, weight=weight), k))


def get_path_weight(graph, path, weight='weight'):
    return np.sum([graph[path[i]][path[i+1]][weight] for i in range(len(path) - 1)])


class Path:
    """
    class to help with holding the information of the paths
    """
    def __init__(self, idp, k, node_list, length, modulation=None, best_modulation=None):
        self.id = idp
        self.k = k
        self.length = length
        self.modulation = modulation
        self.best_modulation = best_modulation
        self.set_node_list(node_list)
        
    def set_node_list(self, node_list):
        self.node_list = node_list
        self.hops = [[node_list[k], node_list[k + 1]] for k in range(len(node_list) - 1)]

    def __str__(self):
        return "node_list: {}, hops: {}, length: {}, modulation: {}".format(self.node_list, len(self.node_list) - 1, self.length, self.modulation['modulation'])

class Demand:

    def __init__(self, did, src, dst, bit_rate, shortest_length, best_modulation):
        self.demand_id = did
        self.source = src
        self.destination = dst
        self.bit_rate = bit_rate
        self.shortest_length = shortest_length
        self.best_modulation = best_modulation
        self.route = None
        self.initial_slot = None
        self.number_slots = None
        
    def set_route(self, route, modulations): # this method saves the selected route, and computes the modulation format and the number of slots required
        self.route = route
        self.number_slots = math.ceil(self.bit_rate / route.modulation['capacity'])
#         print(self.demand_id, self.bit_rate, self.modulation['modulation'], self.modulation['capacity'], self.number_slots)
        
    def __str__(self):
        return "demand: {}, source: {}, destination: {}".format(self.demand_id, self.source, self.destination)

class Solution:
    
    def __init__(self, demands, routes, first_slots, slots, slots_allocation):
        self.demands = demands
        self.routes = routes
        self.first_slots = first_slots
        self.slots = slots
        self.slots_allocation = slots_allocation
        
    def get_score(self):
        max_indices = np.argwhere(self.slots == 0)
        return np.max(max_indices[:, 1])
    
    def get_max_freq_slot(self):
        max_indices = np.argwhere(self.slots == 0)
        return np.max(max_indices[:, 1])


# procedure that returns the first available slot in the path selected
def get_available_first_slots(topology, slots, path, num_slots):
    available_slots = functools.reduce(np.multiply, slots[[topology[path.node_list[i]][path.node_list[i+1]]['id'] for i in range(len(path.node_list) - 1)], :])
    # indices = search_sequence_numpy(available_slots, np.ones(num_slots, dtype=int))
    indices = []
    for index in range(0, len(available_slots) - num_slots + 1):
        # print(index)
        if np.all(available_slots[index:index + num_slots] == 1):
            indices.append(index)
    # for ind in indices: # this code double-checks the assessment
    #     for i in range(len(path.node_list) - 1):
    #         if np.any(graph.graph['available_slots'][graph[path.node_list[i]][path.node_list[i + 1]]['index'], ind:ind+num_slots] == 0):
    #             allocation = graph.graph['available_slots'][graph[path.node_list[i]][path.node_list[i + 1]]['index'], :]
    #             logging.debug("checking service to link {}-{} with not sufficient resources on slots {}-{}".format(path.node_list[i], path.node_list[i + 1], ind, ind+num_slots))
    return indices


def spectrum_assignment(topology, demands, number_spectrum_units):
    routes = np.zeros((len(demands)), dtype=int) # this vector defines which route id is used for each demand... the route can be recoved from the all_paths vector
    first_slots = np.zeros((len(demands)), dtype=int) # this vector defines which is the first slot for each demand
    slots = np.ones((topology.number_of_edges(), number_spectrum_units), dtype=int) # this vector defines the spectrum usage
    slots_allocation = np.full((topology.number_of_edges(), number_spectrum_units), -1, dtype=int) # this vector contains which demand is using which slot
    for demand in sorted(demands, key=lambda x: len(x.route.node_list), reverse=True):
        routes[demand.demand_id] = demand.route.id
        first_slot = get_available_first_slots(topology, slots, demand.route, demand.number_slots)
        initial_slot = first_slot[0]
        for i in range(len(demand.route.node_list) - 1):
            slots[topology[demand.route.node_list[i]][demand.route.node_list[i + 1]]['id'], initial_slot:initial_slot + demand.number_slots] = 0
            slots_allocation[topology[demand.route.node_list[i]][demand.route.node_list[i + 1]]['id'], initial_slot:initial_slot + demand.number_slots] = demand.demand_id
        demand.initial_slot = initial_slot
    return routes, first_slots, slots, slots_allocation

def create_solution(topology, k_shortest_paths, original_demands, modulations, number_spectrum_units):
    demands = copy.deepcopy(original_demands) # makes a copy of the original vector -- remember that Python always passes reference
    routes = np.zeros((len(demands)), dtype=int) # this vector defines which route id is used for each demand
    first_slots = np.zeros((len(demands)), dtype=int) # this vector defines which is the first slot for each demand
    slots = np.ones((topology.number_of_edges(), number_spectrum_units), dtype=int) # this vector defines the spectrum usage

    # define the routes
    for demand in demands:
        selected_route = np.random.choice(k_shortest_paths[demand.source, demand.destination])
        demand.set_route(selected_route, modulations)
        routes[demand.demand_id] = selected_route.id
#         routes[demand.demand_id] = selected_route
        
    routes, first_slots, slots, slots_allocation = spectrum_assignment(topology, demands, number_spectrum_units)
    
        
#         print(demand.demand_id, len(demand.route.node_list), initial_slot, initial_slot + demand.number_slots)
        
    return Solution(demands, routes, first_slots, slots, slots_allocation)

# a population is a set of RMSA solutions
def create_initial_population(topology, k_shortest_paths, demands, modulations, population_size, number_spectrum_units):
    population = []
    for i in range(0, population_size):
        population.append(create_solution(topology, k_shortest_paths, demands, modulations, number_spectrum_units))
    return population


def plot_spectrum_assignment(vector, values=False, suffix=''):
    plt.figure(figsize=(15, 10))
    
    cmap = plt.cm.viridis
    cmap.set_under(color='white')
    
    cmap_reverse = plt.cm.viridis_r
    cmap_reverse.set_under(color='black')
    p = plt.pcolor(vector, cmap=cmap, vmin=-0.0001)
#     p.set_rasterized(False)

    if values:
        fmt = 'd'
        thresh = vector.max() / 2.
        for i, j in itertools.product(range(vector.shape[0]), range(vector.shape[1])):
            if vector[i, j] == -1:
                continue
            else:
                text = format(vector[i, j], fmt)
            color = cmap_reverse(vector[i, j] / vector.max())
#             print(vector.max(), vector[i, j], color)
            plt.text(j + 0.5, i + 0.5, text,
                     horizontalalignment="center", verticalalignment='center',
                     color=color)

    plt.xlabel('Frequency slot')
    plt.ylabel('Link Id (useful)')
    plt.colorbar()
#     plt.savefig('spectrum-allocation-' + suffix + '.pdf') # uncomment to save the figure
    plt.tight_layout()
    plt.show()

