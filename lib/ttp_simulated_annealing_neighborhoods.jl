# implementation of TTSA neighborhood structures from Anagnostopolous et al.

using Base: swapcols!

##
# cost and constraint violation functions
##
function cost(d::Matrix{UInt16}, schedule::Matrix{Int64})
    result = 0
    for i in 1:size(schedule, 1)
        team_result = 0
        current_position = i
        for j in 1:size(schedule, 2)
            if schedule[i, j] < 0
                target_position = -schedule[i, j]
                delta = d[target_position, current_position]
                team_result += delta
                current_position = target_position
            else
                target_position = i
                delta = d[target_position, current_position]
                team_result += delta
                current_position = target_position
            end
        end
        team_result += d[current_position, i]
        result += team_result
    end
    return result
end

function nbv(instance, schedule, method)
    if method == "lower"
        return nbv_punishOnlyOnce(instance, schedule)
    elseif method == "higher"
        return nbv_punishLongerStreaks(instance, schedule)
    end
end

function nbv_punishOnlyOnce(instance, schedule)
    n = size(schedule,1)
    streak_violations = 0
    repeat_violations = 0

    for t in 1:n
        last_team = t
        away_streak = 0
        home_streak = 0
        for r in 1:size(schedule,2)
            if instance.no_repeat && last_team == -schedule[t,r]
                repeat_violations += 1
            end
            if schedule[t,r] < 0
                away_streak += 1
                home_streak = 0
            else
                home_streak += 1
                away_streak = 0
            end
            if home_streak == instance.streak_limit + 1 || away_streak == instance.streak_limit + 1
                streak_violations += 1
            end
            last_team = schedule[t,r]
        end
    end

    return streak_violations + convert(Int64, repeat_violations//2)
end

function nbv_punishLongerStreaks(instance, schedule)
    infeasibility_cnt = 0
    for t in 1:size(schedule,1)
        last_team = nothing
        away_streak = 0
        home_streak = 0
        for r in 1:size(schedule,2)
            if instance.no_repeat && last_team == -schedule[t,r]
                infeasibility_cnt += 1
            end
            if schedule[t,r] < 0
                away_streak += 1
                home_streak = 0
            else
                home_streak += 1
                away_streak = 0
            end
            if home_streak > instance.streak_limit || away_streak > instance.streak_limit
                infeasibility_cnt += 1
            end
            last_team = schedule[t,r]
        end
    end
    return infeasibility_cnt
end

##
# move definitions
##
struct MoveSwapHomes
    team_1::Int64
    team_2::Int64
end

struct MoveSwapRounds
    round_1::Int64
    round_2::Int64
end

struct MoveSwapTeams
    team_1::Int64
    team_2::Int64
end

struct MovePartialSwapRounds
    team::Int64
    round_1::Int64
    round_2::Int64
    touched_teams::Vector{Int64}
end

struct MovePartialSwapTeams
    team_1::Int64
    team_2::Int64
    round::Int64
    rounds_where_other_teams_meet_team_1::Array{Int64, 2}
end

###
# methods to undo moves
###
function apply_swapHomes!(schedule::Matrix{Int64}, t1::Int64, t2::Int64)
    for i in 1:size(schedule, 2)
        if abs(schedule[t1, i]) == t2
            schedule[t1, i] *= -1
            schedule[t2, i] *= -1
        end
    end
end

function apply_swapRounds!(schedule::Matrix{Int64}, r1::Int64, r2::Int64)
    swapcols!(schedule, r1, r2)
end

function apply_swapTeams!(schedule::Matrix{Int64}, t1::Int64, t2::Int64)
    for i in 1:size(schedule, 2)
        if abs(schedule[t1, i]) != t2
            schedule[t1, i], schedule[t2, i] = schedule[t2, i], schedule[t1, i]
            schedule[abs(schedule[t1, i]), i] = schedule[t1, i] > 0 ? -t1 : t1
            schedule[abs(schedule[t2, i]), i] = schedule[t2, i] > 0 ? -t2 : t2
        end
    end
end

function apply_partialSwapRounds!(schedule::Matrix{Int64}, t::Int64, r1::Int64, r2::Int64, d::Matrix{UInt16}, move::MovePartialSwapRounds)
    chain = move.touched_teams
    sizehint!(chain, size(schedule, 1))
    append!(chain, t)
    chaingoal = abs(schedule[t, r2])
    roundToLookAt = r1
    cnt = 0
    delta_costs = 0
    while last(chain) != chaingoal
        append!(chain, abs(schedule[last(chain), roundToLookAt]))
        roundToLookAt = roundToLookAt == r2 ? r1 : r2

        cnt += 1
    end
    for team in chain
        r1_entry = schedule[team, r1]
        r2_entry = schedule[team, r2]
        delta_costs += apply_and_calc_change_delta!(schedule, team, r1, schedule[team, r1], r2_entry, d)
        delta_costs += apply_and_calc_change_delta!(schedule, team, r2, schedule[team, r2], r1_entry, d)
    end

    return delta_costs
end

function apply_partialSwapTeams!(schedule::Matrix{Int64}, t1::Int64, t2::Int64, r::Int64, move::MovePartialSwapTeams)
    swappedAwayOfT1 = schedule[t1, r]
    if abs(swappedAwayOfT1) == t2
        return
    end
    
    roundsWhereT1MeetsOtherTeams = move.rounds_where_other_teams_meet_team_1
    schedule[t1, r], schedule[t2, r] = schedule[t2, r], schedule[t1, r]
    schedule[abs(schedule[t1, r]), r] = schedule[t1, r] < 0 ? t1 : -t1
    schedule[abs(schedule[t2, r]), r] = schedule[t2, r] < 0 ? t2 : -t2
    swappedToT1 = schedule[t1, r]
    while swappedToT1 != swappedAwayOfT1
        if swappedToT1 > 0
            idx = roundsWhereT1MeetsOtherTeams[1, swappedToT1]
        else
            idx = roundsWhereT1MeetsOtherTeams[2,-swappedToT1]
        end
        schedule[t1, idx], schedule[t2, idx] = schedule[t2, idx], schedule[t1, idx]
        schedule[abs(schedule[t1, idx]), idx] = schedule[t1, idx] < 0 ? t1 : -t1
        schedule[abs(schedule[t2, idx]), idx] = schedule[t2, idx] < 0 ? t2 : -t2
        swappedToT1 = schedule[t1, idx]
    end
end

function apply_and_calc_change_delta!(schedule::Matrix{Int64}, team::Int64, round::Int64, from::Int64, to::Int64, d::Matrix{UInt16})
    max_r = size(schedule, 2)

    if round == 1
        start_venue = team
    else
        start_venue = entry_to_venue(team, schedule[team, round-1])
    end

    old_entry_venue = entry_to_venue(team, from)
    new_entry_venue = entry_to_venue(team, to)

    if round == max_r
        end_venue = team
    else
        end_venue = entry_to_venue(team, schedule[team, round+1])
    end

    old_costs_by_entry = convert(Int64, d[start_venue, old_entry_venue]) + convert(Int64, d[old_entry_venue, end_venue])
    new_costs_by_entry = convert(Int64, d[start_venue, new_entry_venue]) + convert(Int64, d[new_entry_venue, end_venue])

    schedule[team, round] = to

    return new_costs_by_entry - old_costs_by_entry
end

function apply!(schedule::Matrix{Int64}, move)
    undo!(schedule, move)
end

function apply!(schedule::Matrix{Int64}, move::MovePartialSwapTeams)
    apply_partialSwapTeams!(schedule, move.team_1, move.team_2, move.round, move)
end

function undo!(schedule::Matrix{Int64}, move::MoveSwapHomes)
    apply_swapHomes!(schedule, move.team_1, move.team_2)
end

function undo!(schedule::Matrix{Int64}, move::MoveSwapRounds)
    apply_swapRounds!(schedule, move.round_1, move.round_2)
end

function undo!(schedule::Matrix{Int64}, move::MoveSwapTeams)
    apply_swapTeams!(schedule, move.team_1, move.team_2)
end

function undo!(schedule::Matrix{Int64}, move::MovePartialSwapTeams)
    apply_partialSwapTeams!(schedule, move.team_2, move.team_1, move.round, move)
end

function undo!(schedule::Matrix{Int64}, move::MovePartialSwapRounds)
    for team in move.touched_teams
        schedule[team, move.round_1], schedule[team, move.round_2] = schedule[team, move.round_2], schedule[team, move.round_1]
    end
end

##
# delta evaluators for neighborhood moves
##
function entry_to_venue(team::Int64, matrix_entry::Int64)
    if matrix_entry > 0
        return team
    else
        return -matrix_entry
    end
end

function delta_swapHomes(schedule::Matrix{Int64}, t1::Int64, t2::Int64, costs::Int64, infeasibilities::Int64, d::Matrix{UInt16})
    # collect delta costs
    delta_costs = 0
    for i in 1:size(schedule, 2)
        if abs(schedule[t1, i]) == t2
            delta_costs += apply_and_calc_change_delta!(schedule, t1, i, schedule[t1, i], -schedule[t1, i], d)
            delta_costs += apply_and_calc_change_delta!(schedule, t2, i, schedule[t2, i], -schedule[t2, i], d)
        end
    end

    return delta_costs, infeasibilities
end

function delta_swapRounds(schedule::Matrix{Int64}, r1::Int64, r2::Int64, costs::Int64, infeasibilities::Int64, d::Matrix{UInt16})
    delta_costs = 0
    for t in 1:size(schedule, 1)
        r1_entry = schedule[t, r1]
        r2_entry = schedule[t, r2]
        delta_costs += apply_and_calc_change_delta!(schedule, t, r1, r1_entry, r2_entry, d)
        delta_costs += apply_and_calc_change_delta!(schedule, t, r2, r2_entry, r1_entry, d)
    end    
    
    return delta_costs, infeasibilities
end

function delta_swapTeams(schedule::Matrix{Int64}, t1::Int64, t2::Int64, costs::Int64, infeasibilities::Int64, d::Matrix{UInt16})
    delta_costs = 0
    for i in 1:size(schedule, 2)
        if abs(schedule[t1, i]) != t2
            t1_entry = schedule[t1, i]
            t2_entry = schedule[t2, i]
            delta_costs += apply_and_calc_change_delta!(schedule, t1, i, t1_entry, t2_entry, d)
            delta_costs += apply_and_calc_change_delta!(schedule, t2, i, t2_entry, t1_entry, d)
            delta_costs += apply_and_calc_change_delta!(schedule, abs(schedule[t1, i]), i, schedule[abs(schedule[t1, i]), i], schedule[t1, i] > 0 ? -t1 : t1, d)
            delta_costs += apply_and_calc_change_delta!(schedule, abs(schedule[t2, i]), i, schedule[abs(schedule[t2, i]), i], schedule[t2, i] > 0 ? -t2 : t2, d)
        end
    end

    return delta_costs, infeasibilities
end

function delta_partialSwapTeams(schedule::Matrix{Int64}, t1::Int64, t2::Int64, r::Int64, costs::Int64, infeasibilities::Int64, d::Matrix{UInt16}, move::MovePartialSwapTeams)
    delta_costs = 0

    swappedAwayOfT1 = schedule[t1, r]
    if abs(swappedAwayOfT1) == t2
        return delta_costs, infeasibilities
    end

    roundsWhereT1MeetsOtherTeams = move.rounds_where_other_teams_meet_team_1
    for i in 1:1:size(schedule,2)
        if schedule[t1, i] > 0
            roundsWhereT1MeetsOtherTeams[1, schedule[t1,i]] = i
        else
            roundsWhereT1MeetsOtherTeams[2, -schedule[t1,i]] = i
        end
    end

    t1_entry = schedule[t1, r]
    t2_entry = schedule[t2, r]
    delta_costs += apply_and_calc_change_delta!(schedule, t1, r, schedule[t1, r], t2_entry, d)
    delta_costs += apply_and_calc_change_delta!(schedule, t2, r, schedule[t2, r], t1_entry, d)

    opponent1 = abs(schedule[t1, r])
    opponent1_entry = schedule[t1, r] < 0 ? t1 : -t1
    opponent2 = abs(schedule[t2, r])
    opponent2_entry = schedule[t2, r] < 0 ? t2 : -t2
    delta_costs += apply_and_calc_change_delta!(schedule, opponent1, r, schedule[opponent1, r], opponent1_entry, d)
    delta_costs += apply_and_calc_change_delta!(schedule, opponent2, r, schedule[opponent2, r], opponent2_entry, d)

    swappedToT1 = schedule[t1, r]
    while swappedToT1 != swappedAwayOfT1
        if swappedToT1 > 0
            idx = roundsWhereT1MeetsOtherTeams[1, swappedToT1]
        else
            idx = roundsWhereT1MeetsOtherTeams[2,-swappedToT1]
        end

        t1_entry = schedule[t1, idx]
        t2_entry = schedule[t2, idx]
        delta_costs += apply_and_calc_change_delta!(schedule, t1, idx, schedule[t1, idx], t2_entry, d)
        delta_costs += apply_and_calc_change_delta!(schedule, t2, idx, schedule[t2, idx], t1_entry, d)

        opponent1 = abs(schedule[t1, idx])
        opponent1_entry = schedule[t1, idx] < 0 ? t1 : -t1
        opponent2 = abs(schedule[t2, idx])
        opponent2_entry = schedule[t2, idx] < 0 ? t2 : -t2
        delta_costs += apply_and_calc_change_delta!(schedule, opponent1, idx, schedule[opponent1, idx], opponent1_entry, d)
        delta_costs += apply_and_calc_change_delta!(schedule, opponent2, idx, schedule[opponent2, idx], opponent2_entry, d)

        schedule[opponent1, idx] = opponent1_entry
        schedule[opponent2, idx] = opponent2_entry

        swappedToT1 = schedule[t1, idx]
    end

    return delta_costs, infeasibilities
end

function delta_partialSwapRounds(schedule::Matrix{Int64}, t::Int64, r1::Int64, r2::Int64, costs::Int64, infeasibilities::Int64, d::Matrix{UInt16}, move::MovePartialSwapRounds)
    delta_costs = apply_partialSwapRounds!(schedule, t, r1, r2, d, move)

    return delta_costs, infeasibilities
end

# feasible local search related functions
function feasible_local_search(instance, schedule::Matrix{Int64}, max_chain_length::Int, debug::Bool = true)
    current_costs = cost(instance.d, schedule)
    constraint_violations = nbv(instance, schedule, "lower")
    improving_moves_found = 0
    @assert constraint_violations == 0
    feasible_moves_with_delta_costs = []
    chain = []
    applied_moves = []
    seen_schedules = Set()
    evaluated_schedules = 0

    # swap homes
    while true
        chain_improvement = 0
        evaluated_schedules += evaluate_moves(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)

        # make improving move or return
        if length(feasible_moves_with_delta_costs) > 0
            sorted_moves_with_delta_costs = sort(feasible_moves_with_delta_costs, by = x -> x[2])
            #println(sorted_moves_with_delta_costs)
            best_move, delta_costs = sorted_moves_with_delta_costs[1]

            # only non improving moves
            if delta_costs >= 0
                break
            end

            #println(best_move, delta_costs)
            improving_moves_found += 1
            for move in best_move
                apply!(schedule, move)
                push!(applied_moves, move)
            end
            #apply!(schedule, best_move[1])
            #push!(applied_moves, best_move[1])
            #current_costs = cost(instance.d, schedule)
            current_costs += delta_costs
            
            if debug
                @assert current_costs == cost(instance.d, schedule)
                @assert nbv(instance, schedule, "lower") == 0
            end
        # only infeasible moves
        else
            break
        end

        #println(current_costs)
        empty!(feasible_moves_with_delta_costs)
    end
    
    @assert nbv(instance, schedule, "lower") == 0
    #println(applied_moves)
    schedule, current_costs, improving_moves_found, evaluated_schedules
end

function evaluate_moves(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)
    evaluated_schedules = 0
    if length(chain) >= max_chain_length
        return evaluated_schedules
    else
        evaluated_schedules += swap_homes_neighborhood(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)
        evaluated_schedules += swap_rounds_neighborhood(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)
        evaluated_schedules += swap_teams_neighborhood(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)
        if length(chain) == 0# || instance.n <= 16 #|| (!(chain[1] isa MovePartialSwapRounds) && !(chain[1] isa MovePartialSwapTeams))
        evaluated_schedules += partial_swap_rounds_neighborhood(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)
        evaluated_schedules += partial_swap_teams_neighborhood(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)
        end
        return evaluated_schedules
    end
end

function swap_homes_neighborhood(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)
    neighbors_checked = 0
    length_of_chain = length(chain)

    for team_1 in 1:(instance.n - 1)
        for team_2 in (team_1 + 1):instance.n
            move = MoveSwapHomes(team_1, team_2)
            delta_costs, _ = delta_swapHomes(schedule, team_1, team_2, current_costs, 0, instance.d)
            schedule_hash = hash((length_of_chain, schedule))
            if schedule_hash in seen_schedules
                undo!(schedule, move)
                continue
            else
                neighbors_checked += 1
                push!(seen_schedules, schedule_hash)
            end
            constraint_violations_after_move = nbv(instance, schedule, "lower")
            next_chain = cat(chain, move, dims=1)
            if constraint_violations_after_move == 0
                push!(feasible_moves_with_delta_costs, (next_chain, chain_improvement + delta_costs))
            end
            @assert length(next_chain) > length_of_chain
            @assert length(next_chain) <= max_chain_length
            neighbors_checked += evaluate_moves(instance, seen_schedules, schedule, current_costs + delta_costs, feasible_moves_with_delta_costs, next_chain, chain_improvement + delta_costs, max_chain_length, debug)
            undo!(schedule, move)
            
            if debug
                @assert current_costs == cost(instance.d, schedule)
                #@assert nbv(instance, schedule, "lower") == 0
            end
        end
    end

    neighbors_checked
end

function swap_rounds_neighborhood(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)
    neighbors_checked = 0
    rounds = size(schedule, 2)
    length_of_chain = length(chain)
    
    for round_1 in 1:(rounds - 1)
        for round_2 in (round_1 + 1):rounds
            neighbors_checked += 1
            move = MoveSwapRounds(round_1, round_2)
            delta_costs, _ = delta_swapRounds(schedule, round_1, round_2, current_costs, 0, instance.d)
            schedule_hash = hash((length_of_chain, schedule))
            if schedule_hash in seen_schedules
                undo!(schedule, move)
                continue
            else
                neighbors_checked += 1
                push!(seen_schedules, schedule_hash)
            end
            constraint_violations_after_move = nbv(instance, schedule, "lower")
            next_chain = cat(chain, move, dims=1)
            if constraint_violations_after_move == 0
                push!(feasible_moves_with_delta_costs, (next_chain, chain_improvement + delta_costs))
            end
            @assert length(next_chain) > length_of_chain
            @assert length(next_chain) <= max_chain_length
            neighbors_checked += evaluate_moves(instance, seen_schedules, schedule, current_costs + delta_costs, feasible_moves_with_delta_costs, next_chain, chain_improvement + delta_costs, max_chain_length, debug)
            undo!(schedule, move)
            
            if debug
                @assert current_costs == cost(instance.d, schedule)
                #@assert nbv(instance, schedule, "lower") == 0
            end
        end
    end

    neighbors_checked
end

function swap_teams_neighborhood(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)
    neighbors_checked = 0
    length_of_chain = length(chain)

    for team_1 in 1:(instance.n - 1)
        for team_2 in (team_1 + 1):instance.n
            neighbors_checked += 1
            move = MoveSwapTeams(team_1, team_2)
            delta_costs, _ = delta_swapTeams(schedule, team_1, team_2, current_costs, 0, instance.d)
            schedule_hash = hash((length_of_chain, schedule))
            if schedule_hash in seen_schedules
                undo!(schedule, move)
                continue
            else
                neighbors_checked += 1
                push!(seen_schedules, schedule_hash)
            end
            constraint_violations_after_move = nbv(instance, schedule, "lower")
            next_chain = cat(chain, move, dims=1)
            if constraint_violations_after_move == 0
                push!(feasible_moves_with_delta_costs, (next_chain, chain_improvement + delta_costs))
            end
            @assert length(next_chain) > length_of_chain
            @assert length(next_chain) <= max_chain_length
            neighbors_checked += evaluate_moves(instance, seen_schedules, schedule, current_costs + delta_costs, feasible_moves_with_delta_costs, next_chain, chain_improvement + delta_costs, max_chain_length, debug)
            undo!(schedule, move)
            
            if debug
                @assert current_costs == cost(instance.d, schedule)
                #@assert nbv(instance, schedule, "lower") == 0
            end
        end
    end

    neighbors_checked
end

function partial_swap_rounds_neighborhood(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)
    neighbors_checked = 0
    rounds = size(schedule, 2)
    length_of_chain = length(chain)

    for team in 1:instance.n
        for round_1 in 1:(rounds - 1)
            for round_2 in (round_1 + 1):rounds
                neighbors_checked += 1
                move = MovePartialSwapRounds(team, round_1, round_2, Vector{Int64}())
                delta_costs, _ = delta_partialSwapRounds(schedule, team, round_1, round_2, current_costs, 0, instance.d, move)
                schedule_hash = hash((length_of_chain, schedule))
                if schedule_hash in seen_schedules
                    undo!(schedule, move)
                    continue
                else
                    neighbors_checked += 1
                    push!(seen_schedules, schedule_hash)
                end
                constraint_violations_after_move = nbv(instance, schedule, "lower")
                next_chain = cat(chain, move, dims=1)
                if constraint_violations_after_move == 0
                    push!(feasible_moves_with_delta_costs, (next_chain, chain_improvement + delta_costs))
                end
                @assert length(next_chain) > length_of_chain
                @assert length(next_chain) <= max_chain_length
                neighbors_checked += evaluate_moves(instance, seen_schedules, schedule, current_costs + delta_costs, feasible_moves_with_delta_costs, next_chain, chain_improvement + delta_costs, max_chain_length, debug)
                undo!(schedule, move)
                
                if debug
                    @assert current_costs == cost(instance.d, schedule)
                    #@assert nbv(instance, schedule, "lower") == 0
                end
            end
        end
    end

    neighbors_checked
end

function partial_swap_teams_neighborhood(instance, seen_schedules, schedule, current_costs, feasible_moves_with_delta_costs, chain, chain_improvement, max_chain_length, debug)
    neighbors_checked = 0
    rounds = size(schedule, 2)
    length_of_chain = length(chain)

    for round in 1:rounds
        for team_1 in 1:instance.n
            for team_2 in 1:instance.n
                if team_1 == team_2
                    continue
                end

                neighbors_checked += 1
                move = MovePartialSwapTeams(team_1, team_2, round, zeros(2, size(schedule, 1)))
                delta_costs, _ = delta_partialSwapTeams(schedule, team_1, team_2, round, current_costs, 0, instance.d, move)
                schedule_hash = hash((length_of_chain, schedule))
                if schedule_hash in seen_schedules
                    undo!(schedule, move)
                    continue
                else
                    neighbors_checked += 1
                    push!(seen_schedules, schedule_hash)
                end
                constraint_violations_after_move = nbv(instance, schedule, "lower")
                next_chain = cat(chain, move, dims=1)
                if constraint_violations_after_move == 0
                    push!(feasible_moves_with_delta_costs, (next_chain, chain_improvement + delta_costs))
                end
                @assert length(next_chain) > length_of_chain
                @assert length(next_chain) <= max_chain_length
                neighbors_checked += evaluate_moves(instance, seen_schedules, schedule, current_costs + delta_costs, feasible_moves_with_delta_costs, next_chain, chain_improvement + delta_costs, max_chain_length, debug)
                undo!(schedule, move)

                if debug
                    @assert current_costs == cost(instance.d, schedule)
                    #@assert nbv(instance, schedule, "lower") == 0
                end
            end
        end
    end

    neighbors_checked
end