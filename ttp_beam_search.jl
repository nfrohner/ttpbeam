#!/usr/bin/env julia
# ttp_beam_search.jl
# performs beam search for a given TTP instance with either exact CVRP or CVRPH bounds guidance
# bounds have to precalculated via ttp_bounds_precalculation.jl saved as pickled numpy array

include("lib/ttp_instance.jl")
include("lib/ttp_util.jl")
include("lib/ttp_solver.jl")

import .TTPInstance
using .TTPUtil
import .TTPSolver

using Printf: @printf

function main()
    if(length(ARGS) != 9)
        println("Usage: $PROGRAM_FILE <instance-file> <streak-limit> <no-repeat> <cvrp-bounds-file> <beam-width> <dead_teams_check> <randomized-team-order> <relative-sigma> <first-k-layers-noisy>")
        exit(1)
    end

    instance_file = ARGS[1]
    streak_limit = parse(UInt8, ARGS[2])
    no_repeat = parse(Bool, ARGS[3])
    cvrp_bounds_file = ARGS[4]
    beam_width = parse(Int64, ARGS[5])
    dead_teams_check = parse(Bool, ARGS[6])
    randomized_team_order = parse(Bool, ARGS[7])
    relative_sigma = parse(Float64, ARGS[8])
    first_k_layers_noisy = parse(Int64, ARGS[9])
    ttp_instance = TTPInstance.read(instance_file, streak_limit, no_repeat)

    @printf("Loading bounds file %s\n", cvrp_bounds_file)
    bounds_by_state = load_numpy_pickle(cvrp_bounds_file)

    @printf("Solving TTP instance %s with beam search, beam width %d, streak limit %d, no repeaters set to %s\n", instance_file, beam_width, streak_limit, no_repeat ? "true" : "false")
    construction_time = @elapsed terminal, stats = TTPSolver.construct(ttp_instance, bounds_by_state, beam_width, dead_teams_check, randomized_team_order, relative_sigma, nothing, first_k_layers_noisy)

    @printf("[CSV] %s;%d;%d;%d;%.02f;%d;%d;%f;%d\n", TTPUtil.basename(instance_file, ".txt"), ttp_instance.n, beam_width, terminal.shortest_path_length, construction_time, dead_teams_check, randomized_team_order, relative_sigma, first_k_layers_noisy)
end

@time main()
