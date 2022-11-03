#!/usr/bin/env julia
# ttp_beam_search.jl
# performs iterative randomized beam search for a given TTP instance with either approximate CVRP or CVRPH bounds guidance solved by google OR-tools

include("lib/ttp_instance.jl")
include("lib/ttp_util.jl")
include("lib/ttp_solver.jl")

import .TTPInstance
using .TTPUtil
import .TTPSolver

using Printf: @printf
using TableLogger

function main()
    start_time = time()

    if(length(ARGS) != 14)
        println("Usage: $PROGRAM_FILE <instance-file> <streak-limit> <no-repeat> <time-limit> <start-beam-width> <beam-width-limit> <beam-width-increase-factor> <dead_teams_check> <team-ordering> <reflective-symmetry-breaking> <relative-sigma> <runs-per-beam-width> <cvrp-heuristic-respect-home-games> <first-k-layers-noisy>")
        exit(1)
    end

    instance_file = ARGS[1]
    streak_limit = parse(Int64, ARGS[2])
    no_repeat = parse(Bool, ARGS[3])
    time_limit = parse(Int, ARGS[4])
    start_beam_width = parse(Int, ARGS[5])
    beam_width_limit = parse(Int, ARGS[6])
    beam_width_increase_factor = parse(Float32, ARGS[7])
    dead_teams_check = parse(Bool, ARGS[8])
    team_ordering = getproperty(TTPSolver, Symbol(ARGS[9]))
    reflective_symmetry_breaking = getproperty(TTPSolver, Symbol(ARGS[10]))
    relative_sigma = parse(Float64, ARGS[11])
    number_of_runs = parse(Int, ARGS[12])
    cvrp_heuristic_respect_home_games = parse(Bool, ARGS[13])
    first_k_layers_noisy = parse(Int64, ARGS[14])
    ttp_instance = TTPInstance.read(instance_file, streak_limit, no_repeat)

    #@printf("Solving TTP instance %s with beam search and ortools, beam width %d, streak limit %d, no repeaters set to %s\n", instance_file, beam_width, streak_limit, no_repeat ? "true" : "false")
    if cvrp_heuristic_respect_home_games
        heuristic_estimates_cache = Dict{NTuple{6,Int64}, UInt64}()
    else
        heuristic_estimates_cache = Dict{NTuple{4,Int64}, UInt64}()
    end

    log_table = init_log_table(
        (id=:iteration, name="it."),
        (id=:time, name="time [s]"),
        (id=:beam_width, name="beta", width=8),
        (id=:dur, name="dur [s]"),
        (id=:obj, name="obj."),
        (id=:best, name="best"),
        (id=:exp, name="#exp", width=13),
        (id=:thrp, name="[ks^-1]", width=10),
        (id=:lce, name="lce [log]"), # log cache efficiency, log10 of miss rate
        (id=:tph, name="tph [ms]"), # time per heuristic calculation, i.e., average for cache miss
        (id=:flag, name="I", width=5);
        width = 12,
        alignment = :right
    )

    beam_width = start_beam_width
    best_objective = typemax(Int)
    best_terminal = nothing
    root_node = TTPSolver.root_node(ttp_instance, nothing, heuristic_estimates_cache)
    iteration_number = 0
    print_header(log_table)
    time_limit_hit = false
    while true
        if beam_width > beam_width_limit || time_limit_hit
            break
        end
        for i in 1:number_of_runs
            iteration_number += 1

            construction_time = @elapsed terminal, run_stats = TTPSolver.construct(ttp_instance, root_node, nothing, beam_width, dead_teams_check, team_ordering, reflective_symmetry_breaking, relative_sigma, heuristic_estimates_cache, first_k_layers_noisy)
            # dont accept solution after time limit, TODO: already respect time limit in construct method
            current_time = time() - start_time
            if current_time > time_limit
                time_limit_hit = true
                break
            end

            if terminal.shortest_path_length < best_objective
                flag = "I"
                best_objective = terminal.shortest_path_length
                best_terminal = terminal
            else
                flag = "-"
            end

            set_value!(log_table, :iteration, iteration_number)
            set_value!(log_table, :beam_width, beam_width)
            if terminal.shortest_path_length == typemax(Int)
                set_value!(log_table, :obj, "-")
            else
                set_value!(log_table, :obj, terminal.shortest_path_length)
            end
            if best_objective == typemax(Int)
                set_value!(log_table, :best, "-")
            else
                set_value!(log_table, :best, best_objective)
            end
            set_value!(log_table, :time, current_time)
            set_value!(log_table, :dur, construction_time)
            set_value!(log_table, :exp, run_stats.node_expansions)
            set_value!(log_table, :thrp, round(run_stats.node_expansions/(1000*construction_time), digits=2))
            lce = run_stats.cache_misses/(run_stats.cache_misses + run_stats.cache_hits)
            tph = run_stats.heuristics_calculation_time/run_stats.cache_misses
            if run_stats.cache_misses == 0
                set_value!(log_table, :lce, "-inf")
                set_value!(log_table, :tph, 0.0)
            else
                set_value!(log_table, :lce, round(log10(lce), digits=2))
                set_value!(log_table, :tph, round(tph*1000, digits=2))
            end
            set_value!(log_table, :flag, flag)

            print_line(log_table)

        end
        beam_width = Int(round(beam_width*beam_width_increase_factor))
    end

    if beam_width > beam_width_limit
        println("beam width limit hit")
    end
    if time_limit_hit
        println("time limit hit")
    end

    if best_terminal !== nothing && best_objective < typemax(Int)
        @printf("best objective found: %d\n", best_objective)
        schedule = convert(Matrix{Int64}, TTPSolver.solution_to_rounds_matrix(ttp_instance, best_terminal.solution))
        display("text/plain", schedule)
        println()
    else
        println("no feasible solution found")
    end
end

@time main()
