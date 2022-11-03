#!/usr/bin/env julia
# ttp_beam_search.jl
# performs beam search for a given TTP instance with either approximate CVRP or CVRPH bounds guidance solved by google OR-tools

include("lib/ttp_instance.jl")
include("lib/ttp_util.jl")
include("lib/ttp_solver.jl")
include("lib/ttp_simulated_annealing_neighborhoods.jl")

import .TTPInstance
using .TTPUtil
import .TTPSolver

using Printf: @printf

function main()
    if(length(ARGS) != 11)
        println("Usage: $PROGRAM_FILE <instance-file> <streak-limit> <no-repeat> <beam-width> <dead_teams_check> <team-ordering> <reflective-symmetry-breaking> <relative-sigma> <cvrp-heuristic-respect-home-games> <first-k-layers-noisy>")
        exit(1)
    end

    instance_file = ARGS[1]
    streak_limit = parse(Int64, ARGS[2])
    no_repeat = parse(Bool, ARGS[3])
    beam_width = parse(Int64, ARGS[4])
    dead_teams_check = parse(Bool, ARGS[5])
    team_ordering = getproperty(TTPSolver, Symbol(ARGS[6]))
    reflective_symmetry_breaking = getproperty(TTPSolver, Symbol(ARGS[7]))
    relative_sigma = parse(Float64, ARGS[8])
    cvrp_heuristic_respect_home_games = parse(Bool, ARGS[9])
    first_k_layers_noisy = parse(Int64, ARGS[10])
    run_final_ls = parse(Bool, ARGS[11])
    ttp_instance = TTPInstance.read(instance_file, streak_limit, no_repeat)

    @printf("Solving TTP instance %s with beam search and ortools, beam width %d, streak limit %d, no repeaters set to %s\n", instance_file, beam_width, streak_limit, no_repeat ? "true" : "false")
    if cvrp_heuristic_respect_home_games
        heuristic_estimates_cache = Dict{NTuple{6,Int64}, UInt64}()
    else
        heuristic_estimates_cache = Dict{NTuple{4,Int64}, UInt64}()
    end
    root_node = TTPSolver.root_node(ttp_instance, nothing, heuristic_estimates_cache)
    construction_time = @elapsed terminal, stats = TTPSolver.construct(ttp_instance, root_node, nothing, beam_width, dead_teams_check, team_ordering, reflective_symmetry_breaking, relative_sigma, heuristic_estimates_cache, first_k_layers_noisy)

    costs_after_local_search = -1 
    ls_improvements_found = -1
    ls_time = -1
    ls_improvements_found = -1
    ls_evaluated_schedules = -1
    if length(terminal.solution) > 0
        @printf("objective by beam search: %d\n", terminal.shortest_path_length)
        schedule = convert(Matrix{Int64}, transpose(TTPSolver.solution_to_rounds_matrix(ttp_instance, terminal.solution)))
        if run_final_ls
            ls_time = @elapsed schedule_after_local_search, costs_after_local_search, ls_improvements_found, ls_evaluated_schedules = feasible_local_search(ttp_instance, schedule, 2, false)
            @printf("time for local search: %f\n", ls_time)
            @printf("local search evaluated schedules: %d\n", ls_evaluated_schedules)

            @printf("costs after local search: %d\n", costs_after_local_search)
            if costs_after_local_search < terminal.shortest_path_length
                @printf("number of improvements found: %d\n", ls_improvements_found)
                println("improved schedule:")
                display("text/plain", transpose(schedule_after_local_search))
                println()
            end
        else
            display("text/plain", transpose(schedule))
            println()
        end
    else
        @printf("no feasible solution found\n")
    end

    @printf("[CSV] %s;%d;%d;%d;%.02f;%.02f;%d;%d;%s;%s;%f;%d;%d;%d;%f;%d;%d;%d\n", TTPUtil.basename(instance_file, ".txt"), ttp_instance.n, beam_width, terminal.shortest_path_length, construction_time, stats.optimality_check_time, cvrp_heuristic_respect_home_games, dead_teams_check, team_ordering, reflective_symmetry_breaking, relative_sigma, first_k_layers_noisy, length(heuristic_estimates_cache), stats.delta_optimality_checks_performed, ls_time, costs_after_local_search, ls_improvements_found, ls_evaluated_schedules)
end

@time main()
