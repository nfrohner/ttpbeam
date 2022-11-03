# TTP beam search solver module

module TTPSolver
    include("ttp_instance.jl")
    include("ttp_states.jl")
    include("ttp_util.jl")
    include("ttp_node_estimators.jl")
    include("ttp_feasibility.jl")

    using DataStructures: PriorityQueue, enqueue!, dequeue!, peek, extract_all_rev!, extract_all!, BinaryMaxHeap, sizehint!
    using LinearAlgebra: diagind
    using Printf: @printf
    using Random

    import Base.length
    import .TTPInstance
    using .TTPStates
    using .TTPUtil
    using .TTPNodeEstimators

    # beam search statistics
    mutable struct Statistics
        node_expansions::Int64
        duplicate_states::Int64
        feasibility_checks_failed::Int64
        optimality_checks_failed::Int64
        reflective_symmetry_breaking_checks_failed::Int64
        construction_time::Float64
        incorporation_time::Float64
        feasibility_check_time::Float64
        reflective_symmetry_breaking_time::Float64
        next_team_determination::Float64
        transition_time::Float64
        optimality_check_time::Float64
        moving_to_next_layer_time::Float64
        heuristic_estimates_cache_size::Int64
        delta_optimality_checks_performed::Int64
        cache_misses::Int
        cache_hits::Int
        heuristics_calculation_time::Float64
        Statistics() = new(0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 0.0)
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
    function incorporate(parent::Node, new_node::Node, beam::PriorityQueue{State, Node}, beam_width::Int64, stats::Statistics)
        if haskey(beam, new_node.state)
            existing_node = beam[new_node.state]
            stats.duplicate_states += 1
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

    function incorporate(parent::Node, new_node::Node, beam::BinaryMaxHeap{Node}, beam_width::Int64, stats::Statistics)
        if length(beam) < beam_width
            push!(beam, new_node)
        else
            worst_node = first(beam)
            if new_node < worst_node
                pop!(beam)
                push!(beam, new_node)
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
    function play_game(ttp_instance::Main.TTPInstance.Instance, node::Node, away_team::Int64, home_team::Int64, bounds_by_state::Union{Array{UInt16,4},Array{UInt16,5},Nothing}, heuristic_estimates_cache::Union{Dict{NTuple{4,Int64},UInt64},Dict{NTuple{6,Int64},UInt64},Nothing}, noise::Float64, debug::Bool = false)
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

    function worst_node_from_beam(beam::PriorityQueue{State, Node})
        state, node = peek(beam)
        node
    end

    function worst_node_from_beam(beam::BinaryMaxHeap{Node})
        first(beam)
    end

    # incrementally checks whether playing (away_team, home_team) would result into a node that is admissible for the beam by its f-value, otherwise it is not expanded
    function delta_optimality_check(ttp_instance::Main.TTPInstance.Instance, node::Node, away_team::Int64, home_team::Int64, beam, beam_width::Int64, bounds_by_state::Union{Array{UInt16,4},Array{UInt16,5},Nothing}, heuristic_estimates_cache::Union{Dict{NTuple{4,Int64},UInt64},Dict{NTuple{6,Int64},UInt64},Nothing}, noise::Float64, stats::Statistics=nothing)
        weight = ttp_instance.d[node.state.positions[away_team], home_team] + ttp_instance.d[node.state.positions[home_team], home_team]
        away_team_away_teams_left = copy(node.away_games_left_by_team[away_team])
        deleteat!(away_team_away_teams_left, searchsortedfirst(away_team_away_teams_left, home_team))
        home_team_away_teams_left = node.away_games_left_by_team[home_team]
        away_team_number_of_home_games_left = node.number_of_home_games_left[away_team]
        home_team_number_of_home_games_left = node.number_of_home_games_left[home_team] - 1

        heuristic_estimate_delta = -(node.heuristic_estimates[away_team]+node.heuristic_estimates[home_team])
        heuristic_estimate_delta += heuristic_estimate(ttp_instance, away_team, away_team_away_teams_left, away_team_number_of_home_games_left, home_team, ttp_instance.streak_limit - (node.state.possible_away_streaks[away_team]-1), bounds_by_state, heuristic_estimates_cache, stats)
        heuristic_estimate_delta += heuristic_estimate(ttp_instance, home_team, home_team_away_teams_left, home_team_number_of_home_games_left, home_team, ttp_instance.streak_limit - (node.state.possible_home_stands[home_team]-1), bounds_by_state, heuristic_estimates_cache, stats)

        #@assert weight + heuristic_estimate_delta >= 0

        if length(beam) >= beam_width
            worst_node = worst_node_from_beam(beam)
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
        reflective = 5
    end

    @enum ReflectiveSymmetryBreaking begin
        none = 1
        home = 2
        away = 3
    end

    # conditionally play (away_team, home_team), if it is currently allowed and would not result into an infeasible state (according to our checks) and suboptimal node given our current beam
    # team is the one selected by next_team
    # pivot_team is the first according the given team ordering (permutation) and opens a round, might be used for reflective symmetry breaking 
    function conditionally_play_game(ttp_instance::Main.TTPInstance.Instance, node::Node, team::Int64, pivot_team::Int64, away_team::Int64, home_team::Int64, beam, beam_width::Int64, bounds_by_state::Union{Array{UInt16,4},Array{UInt16,5},Nothing}, heuristic_estimates_cache::Union{Dict{NTuple{4,Int64},UInt64},Dict{NTuple{6,Int64},UInt64},Nothing}, dead_teams_check::Bool, noise::Float64, stats::Statistics, reflective_symmetry_breaking::ReflectiveSymmetryBreaking = none, debug::Bool = false)
        if game_allowed(ttp_instance, node.state, away_team, home_team)
            stats.reflective_symmetry_breaking_time += @elapsed reflective_symmetry_breaking_check_result = game_allowed_by_reflective_symmetry_breaking(ttp_instance, node, team, pivot_team, away_team, home_team, reflective_symmetry_breaking)
            if reflective_symmetry_breaking_check_result    
                stats.feasibility_check_time += @elapsed feasibility_check_result = delta_feasibility_check(ttp_instance, node, away_team, home_team, dead_teams_check)
                if feasibility_check_result
                    stats.optimality_check_time += @elapsed optimality_check_result = delta_optimality_check(ttp_instance, node, away_team, home_team, beam, beam_width, bounds_by_state, heuristic_estimates_cache, noise, stats)
                    stats.delta_optimality_checks_performed += 1
                    if optimality_check_result
                        stats.transition_time += @elapsed new_node, weight = play_game(ttp_instance, node, away_team, home_team, bounds_by_state, heuristic_estimates_cache, noise, debug)
                        stats.incorporation_time += @elapsed incorporate(node, new_node, beam, beam_width, stats)
                        return played
                    else
                        return suboptimal
                    end
                else
                    return infeasible
                end
            else
                return reflective
            end    
        else
            return disallowed
        end
    end

    function game_allowed_by_reflective_symmetry_breaking(ttp_instance::Main.TTPInstance.Instance, node::Node, team::Int64, pivot_team::Int64, away_team::Int64, home_team::Int64, reflective_symmetry_breaking::ReflectiveSymmetryBreaking)
        if reflective_symmetry_breaking == none || team != pivot_team || node.state.rounds[pivot_team] >= ttp_instance.n - 1 || (reflective_symmetry_breaking == home && team == away_team) || (reflective_symmetry_breaking == away && team == home_team)
            return true
        else
            pivots_round = node.state.rounds[team] + 1
            rounds_in_first_half_left = ttp_instance.n - 1 - pivots_round
            if reflective_symmetry_breaking == home
                relevant_games_played = convert(Int64, floor(ttp_instance.n - 1 - node.number_of_away_games_left[team]))
            else
                relevant_games_played = convert(Int64, floor(ttp_instance.n - 1 - node.number_of_home_games_left[team]))
            end
            relevant_games_required = convert(Int64, floor(ttp_instance.n/2 - relevant_games_played))
            relevant_streaks_required = convert(Int64, floor(rounds_in_first_half_left / (ttp_instance.streak_limit + 1)))
            relevant_games_possible = rounds_in_first_half_left - relevant_streaks_required
            
            @assert relevant_games_possible >= 0
            
            # assertion before last game of first half is played for pivot
            if reflective_symmetry_breaking == home && relevant_games_possible >= relevant_games_required && node.state.rounds[team] == ttp_instance.n - 2
                @assert node.number_of_away_games_left[team] < node.number_of_home_games_left[team]
            end
            
            if reflective_symmetry_breaking == away && relevant_games_possible >= relevant_games_required && node.state.rounds[team] == ttp_instance.n - 2
                @assert node.number_of_away_games_left[team] > node.number_of_home_games_left[team]
            end
            
            return relevant_games_possible >= relevant_games_required
        end
    end

    function update(stats::Statistics, game_result::GameResult)
        if game_result == infeasible
            stats.feasibility_checks_failed += 1
        elseif game_result == suboptimal
            stats.optimality_checks_failed += 1
        elseif game_result == reflective
            stats.reflective_symmetry_breaking_checks_failed += 1
        end
    end

    # convert solution in form of growing vector to rounds matrix
    function solution_to_rounds_matrix(ttp_instance::Main.TTPInstance.Instance, solution::Array{Tuple{Int64,Int64}})
        copied_solution = reverse!(copy(solution))
        rounds = 2*ttp_instance.n-2
        rounds_matrix = zeros(Int64, (rounds, ttp_instance.n))
        for round in 1:rounds
            for i in 1:ttp_instance.n/2
                game = pop!(copied_solution)
                rounds_matrix[round,game[1]] = -game[2]
                rounds_matrix[round,game[2]] = game[1]
            end
        end
        rounds_matrix
    end

    function length(::Nothing)
        0
    end

    # the first couple of layers might fit into the beam, depending on beam width, there noise does not make sense
    function start_noise_layer(ttp_instance::Main.TTPInstance.Instance, beam_width::Int64)
        layer = 0
        size_requirement_estimate = 1
        while size_requirement_estimate <= beam_width
            size_requirement_estimate = size_requirement_estimate*2*(ttp_instance.n-2*layer-1)
            layer += 1
        end
        return layer
    end

    function spread_noise(ttp_instance::Main.TTPInstance.Instance, sigma::Float64, layer::Int64, start_noise_layer::Int64, beam_width::Int64, first_k_layers_noisy::Int64)
        return sigma > 0.0 && (first_k_layers_noisy == -1 || start_noise_layer <= layer < start_noise_layer + first_k_layers_noisy)
    end

    # optional Gaussian noise for f-values
    function noise_for_guidance_value(ttp_instance::Main.TTPInstance.Instance, sigma::Float64, layer::Int64, start_noise_layer::Int64, first_k_layers_noisy::Int64, beam_width::Int64)
        if spread_noise(ttp_instance, sigma, layer, start_noise_layer, beam_width, first_k_layers_noisy)
            randn()*sigma
        else
            0.0
        end
    end

    function root_node(ttp_instance::Main.TTPInstance.Instance, bounds_by_state::Union{Array{UInt16,4},Array{UInt16,5},Nothing}, heuristic_estimates_cache::Union{Dict{NTuple{4,Int64},UInt64},Dict{NTuple{6,Int64},UInt64},Nothing})
        root = Node()
        root.shortest_path_length = 0
        root.games_left = ttp_instance.n*(ttp_instance.n-1)
        root.heuristic_estimates = zeros(ttp_instance.n)
        for team in 1:ttp_instance.n
            teams_left = convert(Vector{Int64}, 1:ttp_instance.n)
            deleteat!(teams_left, team)
            root.heuristic_estimates[team] = heuristic_estimate(ttp_instance, team, teams_left, ttp_instance.n-1, team, 0, bounds_by_state, heuristic_estimates_cache)
            push!(root.away_games_left_by_team, Vector{Int64}())
            push!(root.home_games_left_by_team, Vector{Int64}())
            root.away_games_left_by_team[team] = teams_left
            root.home_games_left_by_team[team] = teams_left
        end
        root.heuristic_estimate = sum(root.heuristic_estimates)
        root.state = root_state(ttp_instance)
        root.number_of_home_games_left = ones(ttp_instance.n)*(ttp_instance.n-1)
        root.number_of_away_games_left = ones(ttp_instance.n)*(ttp_instance.n-1)

        root
    end    

    @enum TeamOrdering begin
        lexicographic = 1
        random = 2
        highest_independent_lower_bound = 3
        lowest_independent_lower_bound = 4
        highest_total_distance = 5
        lowest_total_distance = 6
    end

    # main function in this module, performs the actual beam search, the layerwise truncated BFS on the TTP state graph
    function construct(ttp_instance::Main.TTPInstance.Instance, start_node::Node, bounds_by_state::Union{Array{UInt16,4},Array{UInt16,5},Nothing}, beam_width::Int64, dead_teams_check::Bool, team_ordering::TeamOrdering, reflective_symmetry_breaking::ReflectiveSymmetryBreaking, sigma_rel::Float64, heuristic_estimates_cache::Union{Dict{NTuple{4,Int64},UInt64},Dict{NTuple{6,Int64},UInt64},Nothing}, first_k_layers_noisy::Int64 = -1, use_binary_heap::Bool = true, debug::Bool = false)
        sigma = (start_node.shortest_path_length + start_node.heuristic_estimate)*sigma_rel

        last_layer = start_node.layer + start_node.games_left

        terminal = Node()
        terminal.shortest_path_length = typemax(Int64)
        terminal.heuristic_estimates = zeros(ttp_instance.n)
        terminal.heuristic_estimate = 0
        terminal.state = terminal_state(ttp_instance)
        terminal.number_of_home_games_left = zeros(ttp_instance.n)
        terminal.number_of_home_games_left = zeros(ttp_instance.n)

        stats = Statistics()

        #println(convert(Array{Int64, 2}, ttp_instance.d))
        #println(sum(convert(Array{Int64, 2}, ttp_instance.d), dims=2))

        if team_ordering == random
            teams_permutation = convert(Array{Int64}, randperm(ttp_instance.n))
        elseif team_ordering == lowest_independent_lower_bound
            teams_permutation = convert(Array{Int64}, sortperm(start_node.heuristic_estimates))
        elseif team_ordering == highest_independent_lower_bound
            teams_permutation = convert(Array{Int64}, sortperm(start_node.heuristic_estimates, rev=true))
        elseif team_ordering == lowest_total_distance
            teams_permutation = convert(Array{Int64}, sortperm(vec(sum(convert(Array{Int64, 2}, ttp_instance.d), dims=2))))
        elseif team_ordering == highest_total_distance
            teams_permutation = convert(Array{Int64}, sortperm(vec(sum(convert(Array{Int64, 2}, ttp_instance.d), dims=2)), rev=true))
        else
            teams_permutation = convert(Array{Int64}, 1:ttp_instance.n)
        end

        teams_permutation = sortperm(teams_permutation)

        pivot_team_for_reflective_symmetry_breaking = argmin(teams_permutation)
        @assert pivot_team_for_reflective_symmetry_breaking == next_team(ttp_instance, start_node, teams_permutation)

        if debug
            @printf("root heuristic estimates: %s\n", start_node.heuristic_estimates)
            @printf("team permutation: %s\n", teams_permutation)
            @printf("pivot team: %d\n", pivot_team_for_reflective_symmetry_breaking)
            @printf("start node heuristic estimate: %d\n", start_node.shortest_path_length + start_node.heuristic_estimate)
        end

        Q = [start_node]
        if use_binary_heap
            beam = BinaryMaxHeap{Node}()
            sizehint!(beam, beam_width)
        else
            beam = PriorityQueue{State, Node}(Base.Order.Reverse)
        end

        start_noise_layer_var = start_noise_layer(ttp_instance, beam_width)
        if debug
            @printf("start noise layer: %d\n", start_noise_layer_var)
        end

        stats.construction_time = @elapsed for layer in start_node.layer:last_layer
            #best_estimate = minimum(map(x -> x.shortest_path_length + x.heuristic_estimate, Q))
            #@printf("layer %d size %d best estimate %d\n", layer, length(Q), best_estimate)
            if debug
                @printf("layer %d size %d\n", layer, length(Q))
            end

            if spread_noise(ttp_instance, sigma, layer+1, start_noise_layer_var, beam_width, first_k_layers_noisy)
                if debug
                    @printf("spreading noise sigma=%f for %d\n", sigma, layer+1)
                end
            end

            seen_nodes = Set{State}()

            for node in Q
                stats.node_expansions += 1

                if use_binary_heap && node.state in seen_nodes
                    stats.duplicate_states += 1
                    continue
                else
                    push!(seen_nodes, node.state)
                end

                if node.games_left == 0
                    go_home(ttp_instance, node, terminal)
                else
                    stats.next_team_determination += @elapsed team = next_team(ttp_instance, node, teams_permutation)

                    for opponent in node.away_games_left_by_team[team]
                        noise = noise_for_guidance_value(ttp_instance, sigma, layer+1, start_noise_layer_var, first_k_layers_noisy, beam_width)
                        game_result = conditionally_play_game(ttp_instance, node, team, pivot_team_for_reflective_symmetry_breaking, team, opponent, beam, beam_width, bounds_by_state, heuristic_estimates_cache, dead_teams_check, noise, stats, reflective_symmetry_breaking, debug)
                        update(stats, game_result)
                    end

                    for opponent in node.home_games_left_by_team[team]
                        noise = noise_for_guidance_value(ttp_instance, sigma, layer+1, start_noise_layer_var, first_k_layers_noisy, beam_width)
                        game_result = conditionally_play_game(ttp_instance, node, team, pivot_team_for_reflective_symmetry_breaking, opponent, team, beam, beam_width, bounds_by_state, heuristic_estimates_cache, dead_teams_check, noise, stats, reflective_symmetry_breaking, debug)
                        update(stats, game_result)
                    end
                end
            end

            #stats.moving_to_next_layer_time += @elapsed Q = extract_all_rev!(beam)
            if use_binary_heap
                stats.moving_to_next_layer_time += @elapsed Q = extract_all_rev!(beam)
                sizehint!(beam, beam_width)
            else
                stats.moving_to_next_layer_time += @elapsed Q = sort(map(x -> x, values(beam)))
                beam = PriorityQueue{State, Node}(Base.Order.Reverse)
            end
        end

        if length(terminal.solution) == 0
            if debug
                println("no feasible solution found")
            end
        else
            if debug
                display("text/plain", solution_to_rounds_matrix(ttp_instance, terminal.solution))
                println()
            end
        end

        stats.heuristic_estimates_cache_size = length(heuristic_estimates_cache)
        if debug
            @printf("node expansions: %d\n", stats.node_expansions)
            @printf("duplicate states: %d\n", stats.duplicate_states)
            @printf("feasibility checks failed %d\n", stats.feasibility_checks_failed)
            @printf("reflective symmetry breaking checks failed %d\n", stats.reflective_symmetry_breaking_checks_failed)
            @printf("optimality checks failed %d\n", stats.optimality_checks_failed)
            @printf("shortest path length %d\n", terminal.shortest_path_length)
            @printf("heuristic estimates cache size %d\n", stats.heuristic_estimates_cache_size)
            @printf("heuristics calculation time: %.02f\n", stats.heuristics_calculation_time)
            @printf("heuristics calculations: %d\n", stats.cache_misses)
            @printf("cache efficiency: %.02f\n", stats.cache_hits/(stats.cache_hits + stats.cache_misses)*100)
            @printf("main loop time: %f\n", stats.construction_time)
            @printf("incorporation time: %f\n", stats.incorporation_time)
            @printf("moving to next layer time: %f\n", stats.moving_to_next_layer_time)
            @printf("feasibility check time: %f\n", stats.feasibility_check_time)
            @printf("reflective symmetry breaking time: %f\n", stats.reflective_symmetry_breaking_time)
            @printf("optimality check time: %f\n", stats.optimality_check_time)
            @printf("delta optimality checks performed: %d\n", stats.delta_optimality_checks_performed)
            @printf("transition time: %f\n", stats.transition_time)
            @printf("next team determination time: %f\n", stats.next_team_determination)
            @printf("expansion time per node: %f microseconds\n", stats.construction_time*1000*1000/stats.node_expansions)
        end

        terminal, stats
    end
end
