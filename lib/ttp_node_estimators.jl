# TTP node estimators based on cvrp and cvrph lower bounds

module TTPNodeEstimators
    include("ttp_instance.jl")
    include("ttp_util.jl")
    import .TTPInstance
    using .TTPUtil
    export heuristic_cvrp_solution, heuristic_cvrp_home_solution, heuristic_estimate

    using PyCall

    # solve cvrp instance with google or-tools via python interface
    function heuristic_cvrp_solution(ttp_instance::Main.TTPInstance.Instance, team::Int64, away_teams::Array{Int64}, position::Int64, streak::Int64)
        pushfirst!(PyVector(pyimport("sys")."path"), "lib")
        heuristic_cvrp_methods = pyimport("heuristic_cvrp_methods")
        return heuristic_cvrp_methods.heuristic_cvrp_solution_for_team(ttp_instance.d, ttp_instance.streak_limit, team-1, away_teams.-1, position-1, streak)
    end

    # solve cvrph instance with google or-tools via python interface
    function heuristic_cvrp_home_solution(ttp_instance::Main.TTPInstance.Instance, team::Int64, away_teams::Array{Int64}, position::Int64, streak::Int64, vehicles_min::Int64, vehicles_max::Int64)
        pushfirst!(PyVector(pyimport("sys")."path"), "lib")
        heuristic_cvrp_methods = pyimport("heuristic_cvrp_methods")
        #@printf("%d %s %d min vehicles %d max vehicles %d\n", team, away_teams, position, vehicles_min, vehicles_max)
        return heuristic_cvrp_methods.heuristic_cvrp_home_solution_for_team(ttp_instance.d, ttp_instance.streak_limit, team-1, away_teams.-1, position-1, streak, vehicles_min, vehicles_max)
    end

    # heuristic estimates based on precalculated exact cvrp bounds
    function heuristic_estimate(ttp_instance::Main.TTPInstance.Instance, team::Int64, teams_left::Array{Int64}, number_of_home_games_left::Int64, position::Int64, streak::Int64, bounds_by_state::Array{UInt16,4}, heuristic_estimates_cache::Nothing)
        if length(teams_left) == 0
            return ttp_instance.d[position, team]
        end

        if team == position
            streak = 0
        end
        if streak == ttp_instance.streak_limit
            ttp_instance.d[position, team] + bounds_by_state[team, mask_teams_left(team, teams_left), team, 1]
        else
            min(ttp_instance.d[position, team] + bounds_by_state[team, mask_teams_left(team, teams_left), team, 1], bounds_by_state[team, mask_teams_left(team, teams_left), position, streak+1])
        end
    end

    # heuristic estimates based on precalculated exact cvrph bounds
    function heuristic_estimate(ttp_instance::Main.TTPInstance.Instance, team::Int64, teams_left::Array{Int64}, number_of_home_games_left::Int64, position::Int64, streak::Int64, bounds_by_state::Array{UInt16,5}, heuristic_estimates_cache::Nothing)
        if length(teams_left) == 0
            return ttp_instance.d[position, team]
        end

        if team == position
            away_streak = 0
        else
            away_streak = streak
        end

        # we hit the away streak limit and have to return home, adding a detour
        if team != position && streak == ttp_instance.streak_limit
            detour = ttp_instance.d[position, team]
            away_streak = 0
            streak = 0
            position = team
        else
            detour = 0
        end

        minimum_home_stands_needed, maximum_home_stands_allowed = min_max_home_stands(ttp_instance, team, teams_left, number_of_home_games_left, position, streak)

        best_bound_direct = minimum(bounds_by_state[team, mask_teams_left(team, teams_left), position, away_streak+1, minimum_home_stands_needed:maximum_home_stands_allowed])
        if team != position
            minimum_home_stands_needed_detour, maximum_home_stands_allowed_detour = min_max_home_stands(ttp_instance, team, teams_left, number_of_home_games_left, team, 0)
            best_bound_via_home = ttp_instance.d[position, team] + minimum(bounds_by_state[team, mask_teams_left(team, teams_left), team, 1, minimum_home_stands_needed_detour:maximum_home_stands_allowed_detour])
        else
            best_bound_via_home = typemax(UInt16)
        end

        return detour + min(best_bound_direct, best_bound_via_home)
    end

    # heuristic estimate based on heuristic solution of cvrp instance via google or-tools
    function heuristic_estimate(ttp_instance::Main.TTPInstance.Instance, team::Int64, teams_left::Array{Int64}, number_of_home_games_left::Int64, position::Int64, streak::Int64, bounds_by_state::Nothing, heuristic_estimates_cache::Dict{NTuple{4,Int64},UInt16})
        if team == position
            streak = 0
        end
        if streak == ttp_instance.streak_limit
            detour = ttp_instance.d[position, team]
            cache_key = (team, mask_teams_left(team, teams_left), team, 0)
        else
            detour = 0
            cache_key = (team, mask_teams_left(team, teams_left), position, streak)
        end
        if !haskey(heuristic_estimates_cache, cache_key)
            heuristic_estimates_cache[cache_key] = heuristic_cvrp_solution(ttp_instance, team, teams_left, position, streak)
        end
        return detour + heuristic_estimates_cache[cache_key]
    end

    # the minimum number and maximum number of vehicles needed for solving the CVRPH problem for a team, inferred by the feasibilty and optimality considerations
    #     the current streak counts as a vehicle
    function min_max_vehicles(ttp_instance::Main.TTPInstance.Instance, team::Int64, teams_left::Array{Int64}, number_of_home_games_left::Int64, position::Int64, streak::Int64)
        minimum_vehicles_needed = max(min_vehicles_by_home_games(ttp_instance, team, teams_left, number_of_home_games_left, position, streak), min_vehicles_by_away_games(ttp_instance, team, teams_left, number_of_home_games_left, position, streak))
        maximum_vehicles_allowed = min(max_vehicles_by_home_games(ttp_instance, team, teams_left, number_of_home_games_left, position, streak), max_vehicles_by_away_games(ttp_instance, team, teams_left, number_of_home_games_left, position, streak))
        maximum_vehicles_allowed = max(maximum_vehicles_allowed, minimum_vehicles_needed)

        minimum_vehicles_needed, maximum_vehicles_allowed
    end

    # this is similar to min max vehicles, but for the precalculated lower bound values we do not count the vehicles (i.e. streaks) but the home stands, where the return home at the end also counts as home stand
    function min_max_home_stands(ttp_instance::Main.TTPInstance.Instance, team::Int64, teams_left::Array{Int64}, number_of_home_games_left::Int64, position::Int64, streak::Int64)
        minimum_vehicles_needed, maximum_vehicles_allowed = min_max_vehicles(ttp_instance, team, teams_left, number_of_home_games_left, position, streak)

        if team == position
            minimum_vehicles_needed + 1, maximum_vehicles_allowed + 1
        else
            minimum_vehicles_needed, maximum_vehicles_allowed
        end
    end

    # the remaining number of home games imply a minimum of vehicles (away streaks) for feasibility...
    function min_vehicles_by_home_games(ttp_instance::Main.TTPInstance.Instance, team::Int64, teams_left::Array{Int64}, number_of_home_games_left::Int64, position::Int64, streak::Int64)
        if team == position
            max(convert(Int64, ceil((streak + number_of_home_games_left)/ttp_instance.streak_limit))-1, 0)
        else
            convert(Int64, ceil(number_of_home_games_left/ttp_instance.streak_limit))
        end
    end

    # .. so do the remaining away games
    function min_vehicles_by_away_games(ttp_instance::Main.TTPInstance.Instance, team::Int64, teams_left::Array{Int64}, number_of_home_games_left::Int64, position::Int64, streak::Int64)
        if team == position
            convert(Int64, ceil(length(teams_left)/ttp_instance.streak_limit))
        else
            convert(Int64, ceil((streak + length(teams_left))/ttp_instance.streak_limit))
        end
    end

    # the number of home games left imply a maximum number of vehicles
    function max_vehicles_by_home_games(ttp_instance::Main.TTPInstance.Instance, team::Int64, teams_left::Array{Int64}, number_of_home_games_left::Int64, position::Int64, streak::Int64)
        1 + number_of_home_games_left
    end

    # if the streak limit > 1, it is never optimal to have more the one streak with length one, since they can be merged
    function max_vehicles_by_away_games(ttp_instance::Main.TTPInstance.Instance, team::Int64, teams_left::Array{Int64}, number_of_home_games_left::Int64, position::Int64, streak::Int64)
        if team == position
            convert(Int64, ceil(length(teams_left)/min(ttp_instance.streak_limit, 2)))
        else
            if streak == 1 && ttp_instance.streak_limit > 1
                1 + convert(Int64, ceil((length(teams_left)-1)/2))
            else
                1 + convert(Int64, ceil(length(teams_left)/min(ttp_instance.streak_limit, 2)))
            end
        end
    end

    # heuristic estimate based on heuristic solution of cvrph instance via google or-tools
    function heuristic_estimate(ttp_instance::Main.TTPInstance.Instance, team::Int64, teams_left::Array{Int64}, number_of_home_games_left::Int64, position::Int64, streak::Int64, bounds_by_state::Nothing, heuristic_estimates_cache::Dict{NTuple{6,Int64},UInt16})
        minimum_vehicles_needed, maximum_vehicles_allowed = min_max_vehicles(ttp_instance, team, teams_left, number_of_home_games_left, position, streak)

        if team == position
            streak = 0
        end

        if streak == ttp_instance.streak_limit
            detour = ttp_instance.d[position, team]
            minimum_vehicles_needed = max(minimum_vehicles_needed-1, 0)
            maximum_vehicles_allowed = max(maximum_vehicles_allowed-1, 0)
            cache_key = (team, mask_teams_left(team, teams_left), team, 0, minimum_vehicles_needed, maximum_vehicles_allowed)
        else
            detour = 0
            cache_key = (team, mask_teams_left(team, teams_left), position, streak, minimum_vehicles_needed, maximum_vehicles_allowed)
        end

        if !haskey(heuristic_estimates_cache, cache_key)
            heuristic_estimates_cache[cache_key] = heuristic_cvrp_home_solution(ttp_instance, team, teams_left, position, streak, minimum_vehicles_needed, maximum_vehicles_allowed)
        end

        return detour + heuristic_estimates_cache[cache_key]
    end
end
