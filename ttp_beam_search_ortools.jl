#!/usr/bin/env julia
# ttp_beam_search.jl
# performs beam search for a given TTP instance with either approximate CVRP or CVRPH bounds guidance solved by google OR-tools

include("lib/ttp_instance.jl")
include("lib/ttp_util.jl")
include("lib/ttp_solver.jl")

import .TTPInstance
using .TTPUtil
import .TTPSolver

using Printf: @printf

function main()
    if(length(ARGS) != 9)
        println("Usage: $PROGRAM_FILE <instance-file> <streak-limit> <no-repeat> <beam-width> <dead_teams_check> <randomized-team-order> <relative-sigma> <cvrp-heuristic-respect-home-games> <first-k-layers-noisy>")
        exit(1)
    end

    instance_file = ARGS[1]
    streak_limit = parse(Int64, ARGS[2])
    no_repeat = parse(Bool, ARGS[3])
    beam_width = parse(Int64, ARGS[4])
    dead_teams_check = parse(Bool, ARGS[5])
    randomized_team_order = parse(Bool, ARGS[6])
    relative_sigma = parse(Float64, ARGS[7])
    cvrp_heuristic_respect_home_games = parse(Bool, ARGS[8])
    first_k_layers_noisy = parse(Int64, ARGS[9])
    ttp_instance = TTPInstance.read(instance_file, streak_limit, no_repeat)

    @printf("Solving TTP instance %s with beam search and ortools, beam width %d, streak limit %d, no repeaters set to %s\n", instance_file, beam_width, streak_limit, no_repeat ? "true" : "false")
    if cvrp_heuristic_respect_home_games
        heuristic_estimates_cache = Dict{NTuple{6,Int64}, UInt16}()
    else
        heuristic_estimates_cache = Dict{NTuple{4,Int64}, UInt16}()
    end
    construction_time = @elapsed terminal, stats = TTPSolver.construct(ttp_instance, nothing, beam_width, dead_teams_check, randomized_team_order, relative_sigma, heuristic_estimates_cache, first_k_layers_noisy)

    #save_numpy_pickle("heuristic_estimates_cache.pkl.bz2", heuristic_estimates_cache)
    @printf("[CSV] %s;%d;%d;%d;%.02f;%d;%d;%f;%d\n", TTPUtil.basename(instance_file, ".txt"), ttp_instance.n, beam_width, terminal.shortest_path_length, construction_time, dead_teams_check, randomized_team_order, relative_sigma, first_k_layers_noisy)
end

@time main()
