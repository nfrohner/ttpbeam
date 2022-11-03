# calculate TTP lower bounds using Gurobi

import numpy as np
from gurobipy import *
import math
import functools
import networkx as nx

GUROBI_ENV=Env("")


def powerset(lst):
    return functools.reduce(lambda result, x: result + [subset + [x] for subset in result], lst, [[]])


def solve_cvrp_home_with_gurobi(d, pos, o, i, away_teams, u, v_min, v_max):
    away_teams = list(away_teams)

    if len(away_teams) == 0:
        if pos != i:
            return d[pos, i], [[pos, i], []]
        return 0, [[], []]

    if pos != i:
        start_away = True
    else:
        start_away = False

    m = Model("cvrp", env=GUROBI_ENV)

    m.setParam('OutputFlag', 0)

    edges_dict = dict()
    for away_team in away_teams:
        edges_dict[(i, away_team)] = d[i, away_team]
        edges_dict[(away_team, i)] = d[away_team, i]

    for away_team in away_teams:
        for other_away_team in away_teams:
            if away_team != other_away_team:
                edges_dict[(away_team, other_away_team)] = d[away_team, other_away_team]

    if start_away:
        edges_dict[(i, pos)] = 0
        edges_dict[(pos, i)] = d[pos, i]
        for away_team in away_teams:
            edges_dict[(pos, away_team)] = d[pos, away_team]

    edges, costs = multidict(edges_dict)

    if start_away:
        away_teams.append(pos)
    nodes = [i] + away_teams
    node_set = set(nodes)

    x = m.addVars(edges, name="edge", vtype=GRB.BINARY)
    m.addConstrs(x.sum('*', j) == 1 for j in away_teams)
    m.addConstrs(x.sum(j, '*') == 1 for j in away_teams)

    # min/max vehicles
    m.addConstr(x.sum(i, '*') >= v_min)
    m.addConstr(x.sum(i, '*') <= v_max)

    away_teams_set = set(away_teams)
    for subset_away_teams in powerset(away_teams_set):
    #for subset_away_teams in [list(away_teams_set)]:
        if len(subset_away_teams) > 0:
            subset_away_teams = set(subset_away_teams)
            complement = node_set - subset_away_teams
            if start_away:
                expr_1 = quicksum([x[k, l] for k in subset_away_teams for l in complement - set([pos])])
                expr_2 = quicksum([x[k, l] for k in complement for l in subset_away_teams - set([pos])])
            else:
                expr_1 = quicksum([x[k, l] for k in subset_away_teams for l in complement])
                expr_2 = quicksum([x[k, l] for k in complement for l in subset_away_teams])
            if start_away and pos in subset_away_teams:
                m.addConstr(expr_1 + expr_2 + x[i, pos] >= int(
                    2 * math.ceil((len(subset_away_teams) + o - 1) * 1.0 / u)))
            else:
                m.addConstr(expr_1 + expr_2 >= int(2*math.ceil(len(subset_away_teams)*1.0/u)))
    
        m.setObjective(x.prod(costs), GRB.MINIMIZE)
    #m.Params.lazyConstraints = 1
    m.setParam('Threads', 1)
    m.setParam('IntFeasTol', 1e-9)
    #m.optimize(mycallback)
    m.optimize()

    #sys.exit(1)
    #if m.status == GRB.Status.OPTIMAL:
    #    print("solved to optimality")
    #else:
    #    print("solved not to optimality")

    if m.status != GRB.Status.INFEASIBLE:
        #print("solution!")
        #return m.objVal
        # print(node.state.x[i], i, node.state.o[i])
        solution = m.getAttr('x', x)
        cvrp_graph = nx.DiGraph()
        cvrp_graph.add_nodes_from(nodes)
        for edge in solution.keys():
             if solution[edge] > 0.5:
                 cvrp_graph.add_edge(edge[0], edge[1], label=costs[edge])
        #vis = nx.nx_agraph.to_agraph(cvrp_graph)  # convert to a graphviz graph
        #vis.write("cvrp.dot")
        #sys.exit(1)
        away_cycle = []
        startable_cycles = []
        for cycle in list(nx.simple_cycles(cvrp_graph)):
            if start_away and pos in cycle and i in cycle:
                away_cycle = cycle
            else:
                startable_cycles.append(cycle)
        #print(m.objVal, [away_cycle, startable_cycles])
        return m.objVal, [away_cycle, startable_cycles]
    else:
        print("infeasible")
        return 0, None
