#!/usr/bin/env julia

include("lib/ttp_instance.jl")
include("lib/ttp_util.jl")
include("lib/ttp_cvrp.jl")

import .TTPInstance
using .TTPUtil
import .TTPCVRP
using Printf: @printf
using PyCall

function main()
    if(length(ARGS) != 4)
        println("Usage: $PROGRAM_FILE <instance-file> <streak-limit> <numpy-pickle-file> <cvrp-h-bounds>")
        exit(1)
    end

    instance_file = ARGS[1]
    streak_limit = parse(UInt8, ARGS[2])
    numpy_pickle_file = ARGS[3]
    ttp_instance = TTPInstance.read(instance_file, streak_limit, true)
    cvrp_h_bounds = parse(Bool, ARGS[4])

    if cvrp_h_bounds
        bounds_by_state = zeros(UInt16, (ttp_instance.n, 2^(ttp_instance.n-1), ttp_instance.n, streak_limit, ttp_instance.n))
    else
        bounds_by_state = zeros(UInt16, (ttp_instance.n, 2^(ttp_instance.n-1), ttp_instance.n, streak_limit))
    end
    root_bound_sum = TTPCVRP.calculate_bounds_for_teams(ttp_instance, bounds_by_state)

    @printf("bounds sum %d\n", sum(bounds_by_state))
    @printf("root bound %d\n", root_bound_sum)

    root_bound = 0
    for team in 1:ttp_instance.n
        if cvrp_h_bounds
            team_root_bound = minimum(@view bounds_by_state[team, 2^(ttp_instance.n-1), team, 1, :])
        else
            team_root_bound = bounds_by_state[team, 2^(ttp_instance.n-1), team, 1]
        end
        root_bound += team_root_bound
        @printf("team %d bound %d\n", team, team_root_bound)
    end

    @printf("root bound: %d\n", root_bound)

    save_numpy_pickle(numpy_pickle_file, bounds_by_state)
end

@time main()
