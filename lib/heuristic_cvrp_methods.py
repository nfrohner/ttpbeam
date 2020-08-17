# Python/Google OR-Tools heuristic CVRP(H) solution methods

from __future__ import print_function

import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def heuristic_cvrp_solution_for_team(d, u, i, away_teams, x, a_streak, debug=False):
    if len(away_teams) == 0:
        return d[x, i]

    if 0 < a_streak < u:
        indices = np.hstack(([i, x], away_teams))
    else:
        indices = np.hstack(([i], away_teams))
    new_d = d[indices, :][:, indices]

    data = dict()

    data['distance_matrix'] = new_d
    data['demands'] = [0]
    data['vehicle_capacities'] = []
    data['depot'] = 0
    data['starts'] = []
    data['ends'] = []

    if 0 < a_streak < u:
        data['demands'].append(a_streak)
        data['distance_matrix'][0][1] = 0
    for _ in away_teams:
        data['demands'].append(1)
    if 0 < a_streak < u:
        data['vehicle_capacities'].append(u)
        data['starts'].append(data['depot'])
    for _ in range(len(away_teams)):
        data['vehicle_capacities'].append(u)
        data['starts'].append(data['depot'])
    data['num_vehicles'] = len(data['vehicle_capacities'])
    for _ in range(data['num_vehicles']):
        data['ends'].append(data['depot'])

    #print(data)

    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['starts'], data['ends'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        """Returns the demand of the node."""
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # https://developers.google.com/optimization/routing/routing_options
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_ARC)
    # https://developers.google.com/optimization/routing/tsp#search_strategy
    #search_parameters.local_search_metaheuristic = (
    #    routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)

    #search_parameters.local_search_metaheuristic = (
    #            routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
    #search_parameters.solution_limit = 45
    #search_parameters.time_limit.FromMilliseconds(100)

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT)
    #search_parameters.local_search_metaheuristic = (
    #    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    assignment = routing.SolveWithParameters(search_parameters)

    if assignment:
        if debug:
            print_solution(data, manager, routing, assignment)
        return get_objective(data, manager, routing, assignment)
    else:
        return 0


def heuristic_cvrp_home_solution_for_team(d, u, i, away_teams, x, a_streak, v_min, v_max, debug=False):
    if len(away_teams) == 0:
        return d[x, i]

    if 0 < a_streak < u:
        indices = np.hstack(([i, x], away_teams))
    else:
        indices = np.hstack(([i], away_teams))
    new_d = d[indices, :][:, indices]

    data = dict()

    data['distance_matrix'] = new_d
    data['demands'] = [0]
    data['vehicle_capacities'] = []
    data['depot'] = 0
    data['starts'] = []
    data['ends'] = []

    if 0 < a_streak < u:
        data['demands'].append(a_streak)
        data['distance_matrix'][0][1] = 0
    for _ in away_teams:
        data['demands'].append(1)
    for _ in range(v_max):
        data['vehicle_capacities'].append(u)
        data['starts'].append(data['depot'])
    data['num_vehicles'] = len(data['vehicle_capacities'])
    for _ in range(data['num_vehicles']):
        data['ends'].append(data['depot'])

    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), data['num_vehicles'], data['starts'], data['ends'])
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        """Returns the demand of the node."""
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(
        demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity')

    # use at least v_min vehicles
    capacity_dimension = routing.GetDimensionOrDie('Capacity')
    for v in range(v_min):
        capacity_dimension.CumulVar(routing.End(v)).RemoveInterval(0, 0)

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # https://developers.google.com/optimization/routing/routing_options
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.FIRST_UNBOUND_MIN_VALUE)
    # https://developers.google.com/optimization/routing/tsp#search_strategy
    # search_parameters.local_search_metaheuristic = (
    #    routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC)

    #search_parameters.local_search_metaheuristic = (
    #    routing_enums_pb2.LocalSearchMetaheuristic.TABU_SEARCH)
    #search_parameters.solution_limit = 45
    #search_parameters.time_limit.FromMilliseconds(100)

    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GREEDY_DESCENT)
    #search_parameters.local_search_metaheuristic = (
    #    routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)

    assignment = routing.SolveWithParameters(search_parameters)

    if assignment:
        if debug:
            print_solution(data, manager, routing, assignment)
        return get_objective(data, manager, routing, assignment)
    else:
        return 0


def print_solution(data, manager, routing, assignment):
    """Prints assignment on console."""
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(
            manager.IndexToNode(index), route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        print(plan_output)
        total_distance += route_distance
        total_load += route_load
    print('Total distance of all routes: {}m'.format(total_distance))
    print('Total load of all routes: {}'.format(total_load))


def get_objective(data, manager, routing, assignment):
    """Prints assignment on console."""
    total_distance = 0
    total_load = 0
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = 'Route for vehicle {}:\n'.format(vehicle_id)
        route_distance = 0
        route_load = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index >= len(data['demands']):
                print(data)
                print(node_index)
            route_load += data['demands'][node_index]
            plan_output += ' {0} Load({1}) -> '.format(node_index, route_load)
            previous_index = index
            index = assignment.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id)
        plan_output += ' {0} Load({1})\n'.format(
            manager.IndexToNode(index), route_load)
        plan_output += 'Distance of the route: {}m\n'.format(route_distance)
        plan_output += 'Load of the route: {}\n'.format(route_load)
        total_distance += route_distance
        total_load += route_load

    return total_distance
