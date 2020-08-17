# TTP beam search solver module

module TTPSolver
    include("ttp_instance.jl")
    include("ttp_states.jl")
    include("ttp_util.jl")
    include("ttp_node_estimators.jl")
    include("ttp_feasibility.jl")

    using DataStructures: PriorityQueue, enqueue!, dequeue!, peek
    using LinearAlgebra: diagind
    using Printf: @printf
    using Random

    import Base.length
    import .TTPInstance
    using .TTPStates
    using .TTPUtil
    using .TTPNodeEstimators
    using .TTPFeasibility

    # beam search statistics
    mutable struct Statistics
        feasibility_checks_failed::Int64
        optimality_checks_failed::Int64
        construction_time::Float64
        heuristic_estimates_cache_size::Int64
        Statistics() = new(0, 0, 0.0)
    end

    export construct, solution_to_rounds_matrix

    # special states are the root state..
    function root_state(ttp_instance::Main.TTPInstance.Instance)
        games_left = ones(Bool, (ttp_instance.n, ttp_instance.n))
        games_left[diagind(games_left)] .= false
        forbidden_opponents = -ones(Int64, ttp_instance.n)
        rounds = zeros(Int64, ttp_instance.n)
        positions = 1:ttp_instance.n
        possible_away_streaks = ttp_instance.streak_limit*ones(Int64, ttp_instance.n)
        possible_home_stands = ttp_instance.streak_limit*ones(Int64, ttp_instance.n)
        State(games_left, forbidden_opponents, rounds, positions, possible_away_streaks, possible_home_stands)
    end

    # ..and the terminal state, only paths that reach it, correspond to a feasible solution
    function terminal_state(ttp_instance::Main.TTPInstance.Instance)
        games_left = zeros(Bool, (ttp_instance.n, ttp_instance.n))
        forbidden_opponents = -ones(Int64, ttp_instance.n)
        rounds = ones(Int64, ttp_instance.n)*(2*ttp_instance.n-1)
        positions = 1:ttp_instance.n
        possible_away_streaks = zeros(Int64, ttp_instance.n)
        possible_home_stands = zeros(Int64, ttp_instance.n)
        State(games_left, forbidden_opponents, rounds, positions, possible_away_streaks, possible_home_stands)
    end

    # conditionally incorporate node into beam or update existing node with same state, if shortest path to it has been found
    function incorporate(parent::Node, new_node::Node, beam::PriorityQueue{State, Node}, beam_width::Int64)
        if haskey(beam, new_node.state)
            existing_node = beam[new_node.state]
            if new_node < existing_node
                beam[new_node.state] = new_node
            end
        else
            if length(beam) < beam_width
                enqueue!(beam, new_node.state, new_node)
            else
                worst_state, worst_node = peek(beam)
                if new_node < worst_node
                    beam[new_node.state] = new_node
                    dequeue!(beam)
                end
            end
        end
    end

    function calc_streak_limit_hits(ttp_instance::Main.TTPInstance.Instance, node::Node, state::State, old_node::Node, away_team::Int64, home_team::Int64)
        games_per_round = convert(Int64, ttp_instance.n/2)
        current_round = convert(Int64, ceil(node.layer / games_per_round))
        games_played_in_this_round = node.layer % games_per_round

        node.teams_home_stand_limit_hit_last_round = old_node.teams_home_stand_limit_hit_last_round
        node.teams_away_streak_limit_hit_current_round = old_node.teams_away_streak_limit_hit_current_round
        node.teams_away_streak_limit_hit_last_round = old_node.teams_away_streak_limit_hit_last_round
        node.teams_home_stand_limit_hit_current_round = old_node.teams_home_stand_limit_hit_current_round

        if games_played_in_this_round == 1
            node.teams_home_stand_limit_hit_last_round = node.teams_home_stand_limit_hit_current_round
            node.teams_away_streak_limit_hit_last_round = node.teams_away_streak_limit_hit_current_round
            node.teams_home_stand_limit_hit_current_round = 0
            node.teams_away_streak_limit_hit_current_round = 0
        end

        if old_node.state.possible_home_stands[away_team] == 0
            node.teams_home_stand_limit_hit_last_round -= 1
        end
        if node.state.possible_away_streaks[away_team] == 0
            node.teams_away_streak_limit_hit_current_round += 1
        end
        if node.state.possible_home_stands[away_team] == 0
            node.teams_home_stand_limit_hit_current_round += 1
        end
        if old_node.state.possible_away_streaks[home_team] == 0
            node.teams_away_streak_limit_hit_last_round -= 1
        end
        if node.state.possible_home_stands[home_team] == 0
            node.teams_home_stand_limit_hit_current_round += 1
        end
        if node.state.possible_away_streaks[home_team] == 0
            node.teams_away_streak_limit_hit_current_round += 1
        end

        @assert node.teams_home_stand_limit_hit_last_round >= 0
        @assert node.teams_away_streak_limit_hit_current_round >= 0
        @assert node.teams_away_streak_limit_hit_current_round >= 0
        @assert node.teams_home_stand_limit_hit_last_round >= 0
    end

    # perform a state transition by playing game (away_team, home_team)
    function play_game(ttp_instance::Main.TTPInstance.Instance, node::Node, away_team::Int64, home_team::Int64, bounds_by_state::Union{Array{UInt16,4},Array{UInt16,5},Nothing}, heuristic_estimates_cache::Union{Dict{NTuple{4,Int64},UInt16},Dict{NTuple{6,Int64},UInt16},Nothing}, noise::Float64)
        new_node = Node()
        new_node.layer = node.layer+1
        weight = ttp_instance.d[node.state.positions[away_team], home_team] + ttp_instance.d[node.state.positions[home_team], home_team]
        new_node.shortest_path_length = node.shortest_path_length + weight
        new_node.games_left = node.games_left - 1
        new_node.heuristic_estimates = copy(node.heuristic_estimates)
        new_node.solution = copy(node.solution)
        new_node.number_of_home_games_left = copy(node.number_of_home_games_left)
        new_node.number_of_away_games_left = copy(node.number_of_away_games_left)
        new_node.away_games_left_by_team = copy(node.away_games_left_by_team)
        new_node.home_games_left_by_team = copy(node.home_games_left_by_team)
        new_node.away_games_left_by_team[away_team] = copy(node.away_games_left_by_team[away_team])
        new_node.home_games_left_by_team[home_team] = copy(node.home_games_left_by_team[home_team])
        new_node.number_of_home_games_left[home_team] -= 1
        new_node.number_of_away_games_left[away_team] -= 1
        deleteat!(new_node.away_games_left_by_team[away_team], searchsortedfirst(new_node.away_games_left_by_team[away_team], home_team))
        deleteat!(new_node.home_games_left_by_team[home_team], searchsortedfirst(new_node.home_games_left_by_team[home_team], away_team))
        new_node.state = update_state(ttp_instance, node.state, away_team, home_team, new_node.number_of_away_games_left[home_team], new_node.number_of_home_games_left[away_team])

        calc_streak_limit_hits(ttp_instance, new_node, new_node.state, node, away_team, home_team)

        push!(new_node.solution, (away_team, home_team))

        new_node.heuristic_estimates[away_team] = heuristic_estimate(ttp_instance, away_team, new_node.away_games_left_by_team[away_team], new_node.number_of_home_games_left[away_team], new_node.state.positions[away_team], ttp_instance.streak_limit - new_node.state.possible_away_streaks[away_team], bounds_by_state, heuristic_estimates_cache)
        new_node.heuristic_estimates[home_team] = heuristic_estimate(ttp_instance, home_team, new_node.away_games_left_by_team[home_team], new_node.number_of_home_games_left[home_team], new_node.state.positions[home_team], ttp_instance.streak_limit - new_node.state.possible_home_stands[home_team], bounds_by_state, heuristic_estimates_cache)

        new_node.heuristic_estimate = sum(new_node.heuristic_estimates)
        new_node.noise = noise
        new_node, weight
    end

    # after last round every team has to go to its home venue, while half of them will be already there
    function go_home(ttp_instance::Main.TTPInstance.Instance, node::Node, terminal::Node)
        weight = sum(map(x -> ttp_instance.d[node.state.positions[x], x], 1:ttp_instance.n))
        if node.shortest_path_length + weight < terminal.shortest_path_length
            terminal.shortest_path_length = node.shortest_path_length + weight
            terminal.solution = copy(node.solution)
        end
        terminal
    end

    # in each layer, to break the symmetry within a round, one specific team is selected to play all its possible away and home games
    function next_team(ttp_instance::Main.TTPInstance.Instance, node::Node, teams_permutation::Array{Int64})
        argmin(map(x -> (node.state.rounds[x], teams_permutation[x]), 1:ttp_instance.n))
    end

    # incrementally checks whether playing (away_team, home_team) would result into a node that is admissible for the beam by its f-value, otherwise it is not expanded
    function delta_optimality_check(ttp_instance::Main.TTPInstance.Instance, node::Node, away_team::Int64, home_team::Int64, beam::PriorityQueue{State, Node}, beam_width::Int64, bounds_by_state::Union{Array{UInt16,4},Array{UInt16,5},Nothing}, heuristic_estimates_cache::Union{Dict{NTuple{4,Int64},UInt16},Dict{NTuple{6,Int64},UInt16},Nothing}, noise::Float64)
        weight = ttp_instance.d[node.state.positions[away_team], home_team] + ttp_instance.d[node.state.positions[home_team], home_team]
        away_team_away_teams_left = copy(node.away_games_left_by_team[away_team])
        deleteat!(away_team_away_teams_left, searchsortedfirst(away_team_away_teams_left, home_team))
        home_team_away_teams_left = node.away_games_left_by_team[home_team]
        away_team_number_of_home_games_left = node.number_of_home_games_left[away_team]
        home_team_number_of_home_games_left = node.number_of_home_games_left[home_team] - 1

        heuristic_estimate_delta = -(node.heuristic_estimates[away_team]+node.heuristic_estimates[home_team])
        heuristic_estimate_delta += heuristic_estimate(ttp_instance, away_team, away_team_away_teams_left, away_team_number_of_home_games_left, home_team, ttp_instance.streak_limit - (node.state.possible_away_streaks[away_team]-1), bounds_by_state, heuristic_estimates_cache)
        heuristic_estimate_delta += heuristic_estimate(ttp_instance, home_team, home_team_away_teams_left, home_team_number_of_home_games_left, home_team, ttp_instance.streak_limit - (node.state.possible_home_stands[home_team]-1), bounds_by_state, heuristic_estimates_cache)

        if length(beam) >= beam_width
            worst_state, worst_node = peek(beam)
            if worst_node.shortest_path_length + worst_node.heuristic_estimate + worst_node.noise <= node.shortest_path_length + weight + node.heuristic_estimate + heuristic_estimate_delta + noise
                return false
            end
        end

        return true
    end

    @enum GameResult begin
        played = 1
        disallowed = 2
        infeasible = 3
        suboptimal = 4
    end

    # conditionally play (away_team, home_team), if it is currently allowed and would not result into an infeasible state (according to our checks) and suboptimal node given our current beam
    function conditionally_play_game(ttp_instance::Main.TTPInstance.Instance, node::Node, away_team::Int64, home_team::Int64, beam::PriorityQueue{State, Node}, beam_width::Int64, bounds_by_state::Union{Array{UInt16,4},Array{UInt16,5},Nothing}, heuristic_estimates_cache::Union{Dict{NTuple{4,Int64},UInt16},Dict{NTuple{6,Int64},UInt16},Nothing}, dead_teams_check::Bool, noise::Float64)
        if game_allowed(ttp_instance, node.state, away_team, home_team)
            if delta_feasibility_check(ttp_instance, node, away_team, home_team, dead_teams_check)
                if delta_optimality_check(ttp_instance, node, away_team, home_team, beam, beam_width, bounds_by_state, heuristic_estimates_cache, noise)
                    new_node, weight = play_game(ttp_instance, node, away_team, home_team, bounds_by_state, heuristic_estimates_cache, noise)
                    incorporate(node, new_node, beam, beam_width)
                    return played
                else
                    return suboptimal
                end
            else
                return infeasible
            end
        else
            return disallowed
        end
    end

    function update(stats::Statistics, game_result::GameResult)
        if game_result == infeasible
            stats.feasibility_checks_failed += 1
        elseif game_result == suboptimal
            stats.optimality_checks_failed += 1
        end
    end

    # convert solution in form of growing vector to rounds matrix
    function solution_to_rounds_matrix(ttp_instance::Main.TTPInstance.Instance, solution::Array{Tuple{Int64,Int64}})
        copied_solution = copy(solution)
        rounds = 2*ttp_instance.n-2
        rounds_matrix = zeros(Int64, (rounds, ttp_instance.n))
        for round in 1:rounds
            for i in 1:ttp_instance.n/2
                game = pop!(solution)
                rounds_matrix[round,game[1]] = -game[2]
                rounds_matrix[round,game[2]] = game[1]
            end
        end
        rounds_matrix
    end

    function length(::Nothing)
        0
    end

    # optional Gaussian noise for f-values
    function noise_for_guidance_value(sigma::Float64, layer::Int64, first_k_layers_noisy::Int64)
        if first_k_layers_noisy == -1 || layer <= first_k_layers_noisy
            randn()*sigma
        else
            0.0
        end
    end

    # main function in this module, performs the actual beam search, the layerwise truncated BFS on the TTP state graph
    function construct(ttp_instance::Main.TTPInstance.Instance, bounds_by_state::Union{Array{UInt16,4},Array{UInt16,5},Nothing}, beam_width::Int64, dead_teams_check::Bool, randomized_team_order::Bool, sigma_rel::Float64, heuristic_estimates_cache::Union{Dict{NTuple{4,Int64},UInt16},Dict{NTuple{6,Int64},UInt16},Nothing}, first_k_layers_noisy::Int64 = -1)
        root = Node()
        root.shortest_path_length = 0
        root.games_left = ttp_instance.n*(ttp_instance.n-1)
        root.heuristic_estimates = zeros(ttp_instance.n)
        for team in 1:ttp_instance.n
            teams_left = convert(Vector{Int64}, 1:ttp_instance.n)
            deleteat!(teams_left, team)
            @time root.heuristic_estimates[team] = heuristic_estimate(ttp_instance, team, teams_left, ttp_instance.n-1, team, 0, bounds_by_state, heuristic_estimates_cache)
            push!(root.away_games_left_by_team, Vector{Int64}())
            push!(root.home_games_left_by_team, Vector{Int64}())
            root.away_games_left_by_team[team] = teams_left
            root.home_games_left_by_team[team] = teams_left
        end
        root.heuristic_estimate = sum(root.heuristic_estimates)
        root.state = root_state(ttp_instance)
        root.number_of_home_games_left = ones(ttp_instance.n)*(ttp_instance.n-1)
        root.number_of_away_games_left = ones(ttp_instance.n)*(ttp_instance.n-1)

        sigma = root.heuristic_estimate*sigma_rel

        last_layer = root.games_left

        terminal = Node()
        terminal.shortest_path_length = typemax(Int64)
        terminal.heuristic_estimates = zeros(ttp_instance.n)
        terminal.heuristic_estimate = 0
        terminal.state = terminal_state(ttp_instance)
        terminal.number_of_home_games_left = zeros(ttp_instance.n)
        terminal.number_of_home_games_left = zeros(ttp_instance.n)

        stats = Statistics()

        if randomized_team_order
            teams_permutation = convert(Array{Int64}, randperm(ttp_instance.n))
        else
            teams_permutation = convert(Array{Int64}, 1:ttp_instance.n)
        end

        @printf("root heuristic estimate: %d\n", root.heuristic_estimate)

        Q = [root]
        beam = PriorityQueue{State, Node}(Base.Order.Reverse)

        stats.construction_time = @elapsed for layer in 0:last_layer
            #best_estimate = minimum(map(x -> x.shortest_path_length + x.heuristic_estimate, Q))
            #@printf("layer %d size %d best estimate %d\n", layer, length(Q), best_estimate)
            @printf("layer %d size %d\n", layer, length(Q))

            for node in Q
                if node.games_left == 0
                    go_home(ttp_instance, node, terminal)
                else
                    team = next_team(ttp_instance, node, teams_permutation)

                    for opponent in node.away_games_left_by_team[team]
                        noise = noise_for_guidance_value(sigma, layer+1, first_k_layers_noisy)
                        game_result = conditionally_play_game(ttp_instance, node, team, opponent, beam, beam_width, bounds_by_state, heuristic_estimates_cache, dead_teams_check, noise)
                        update(stats, game_result)
                    end

                    for opponent in node.home_games_left_by_team[team]
                        noise = noise_for_guidance_value(sigma, layer+1, first_k_layers_noisy)
                        game_result = conditionally_play_game(ttp_instance, node, opponent, team, beam, beam_width, bounds_by_state, heuristic_estimates_cache, dead_teams_check, noise)
                        update(stats, game_result)
                    end
                end
            end

            Q = sort(map(x -> x, values(beam)))
            beam = PriorityQueue{State, Node}(Base.Order.Reverse)
        end

        stats.heuristic_estimates_cache_size = length(heuristic_estimates_cache)
        @printf("feasibility checks failed %d\n", stats.feasibility_checks_failed)
        @printf("optimality checks failed %d\n", stats.optimality_checks_failed)
        @printf("shortest path length %d\n", terminal.shortest_path_length)
        @printf("heuristic estimates cache size %d\n", stats.heuristic_estimates_cache_size)

        if length(terminal.solution) == 0
            println("no feasible solution found")
        else
            display("text/plain", solution_to_rounds_matrix(ttp_instance, terminal.solution))
            println()
        end
        terminal, stats
    end
end
