# TTP feasibility related checks for beam search nodes

module TTPFeasibility
    include("ttp_instance.jl")
    include("ttp_states.jl")

    import .TTPInstance
    import .TTPStates

    using Printf: @printf

    export delta_feasibility_check, game_allowed, delta_game_allowed, number_of_permitted_games_for_team_by_check

    # incrementally check whether playing (away_team, home_team) would result into a state without a feasible completion for certain according to our checks
    function delta_feasibility_check(ttp_instance::Main.TTPInstance.Instance, node::Main.TTPSolver.TTPStates.Node, away_team::Int64, home_team::Int64, dead_teams_check::Bool)
        away_team_home_games_left = node.number_of_home_games_left[away_team]
        away_team_away_games_left = node.number_of_away_games_left[away_team] - 1
        home_team_home_games_left = node.number_of_home_games_left[home_team] - 1
        home_team_away_games_left = node.number_of_away_games_left[home_team]

        if !delta_at_most_check(ttp_instance, node, away_team, home_team, away_team_home_games_left, away_team_away_games_left, home_team_home_games_left, home_team_away_games_left)
            return false
        end

        if ttp_instance.no_repeat && !delta_no_repeat_check(ttp_instance, node, away_team, home_team, away_team_home_games_left, away_team_away_games_left, home_team_home_games_left, home_team_away_games_left)
            return false
        end

        if dead_teams_check && !delta_check_dead_team(ttp_instance, node, away_team, home_team)
            return false
        end

        return true
    end

    # checks whether there are still enough home games left to accomodate for the remaining away games (and vice versa) for the away_team and home_team
    function delta_at_most_check(ttp_instance::Main.TTPInstance.Instance, node::Main.TTPSolver.TTPStates.Node, away_team::Int64, home_team::Int64, away_team_home_games_left::Int64, away_team_away_games_left::Int64, home_team_home_games_left::Int64, home_team_away_games_left::Int64)
        if home_team_home_games_left * ttp_instance.streak_limit + ttp_instance.streak_limit < home_team_away_games_left
            return false
        end

        if away_team_away_games_left * ttp_instance.streak_limit + ttp_instance.streak_limit < away_team_home_games_left
            return false
        end

        return true
    end

    # if after playing (away_team, home_team) only two games are left for one of the teams, that are against the same opponent, we would violate no-repeat for sure
    function delta_no_repeat_check(ttp_instance::Main.TTPInstance.Instance, node::Main.TTPSolver.TTPStates.Node, away_team::Int64, home_team::Int64, away_team_home_games_left::Int64, away_team_away_games_left::Int64, home_team_home_games_left::Int64, home_team_away_games_left::Int64)
        if (away_team_home_games_left != 1 || away_team_away_games_left != 1) && (home_team_home_games_left != 1 || home_team_away_games_left != 1)
            return true
        end

        if away_team_home_games_left == 1 && away_team_away_games_left == 1
            away_games = node.away_games_left_by_team[away_team]
            home_games = node.home_games_left_by_team[away_team]
            if away_games[1] == home_team
                remaining_away_opponent = away_games[2]
            else
                remaining_away_opponent = away_games[1]
            end
            remaining_home_opponent = home_games[1]
            if remaining_away_opponent == remaining_home_opponent
                return false
            end
        end

        if home_team_home_games_left == 1 && home_team_away_games_left == 1
            away_games = node.away_games_left_by_team[home_team]
            home_games = node.home_games_left_by_team[home_team]
            if home_games[1] == away_team
                remaining_home_opponent = home_games[2]
            else
                remaining_home_opponent = home_games[1]
            end
            remaining_away_opponent = away_games[1]
            if remaining_away_opponent == remaining_home_opponent
                return false
            end
        end

        return true
    end

    # infer whether there cannot be a dead team after game (away_team, home_team)
    function delta_infer_no_dead_team(ttp_instance::Main.TTPInstance.Instance, layer::Int64, state::Main.TTPSolver.TTPStates.State, team::Int64, team_home_games_left::Int64, team_away_games_left::Int64, teams_away_streak_limit_hit_last_round::Int64, teams_away_streak_limit_hit_current_round::Int64, teams_home_stand_limit_hit_last_round::Int64, teams_home_stand_limit_hit_current_round::Int64)
        worst_case_forbidden_games = 0

        games_per_round = convert(Int64, ttp_instance.n/2)
        current_round = convert(Int64, ceil(layer / games_per_round))
        games_played_in_this_round = layer % games_per_round

        if current_round > state.rounds[team]
            worst_case_forbidden_games += min(2*games_played_in_this_round, team_home_games_left)
            worst_case_forbidden_games += min(2*games_played_in_this_round, team_away_games_left)
        end

        if current_round == state.rounds[team]
            if state.positions[team] != team
                if state.possible_away_streaks[team] == 0
                    worst_case_forbidden_games += team_away_games_left
                    worst_case_forbidden_games += min(team_home_games_left, teams_away_streak_limit_hit_current_round-1)
                else
                    worst_case_forbidden_games += min(ttp_instance.n-1, teams_away_streak_limit_hit_current_round + teams_home_stand_limit_hit_current_round)
                end
            else
                if state.possible_home_stands[team] == 0
                    worst_case_forbidden_games += team_home_games_left
                    worst_case_forbidden_games += min(team_away_games_left, teams_home_stand_limit_hit_current_round-1)
                else
                    worst_case_forbidden_games += min(ttp_instance.n-1, teams_away_streak_limit_hit_current_round + teams_home_stand_limit_hit_current_round)
                end
            end
        else
            if state.positions[team] != team
                if state.possible_away_streaks[team] == 0
                    worst_case_forbidden_games += team_away_games_left
                    worst_case_forbidden_games += min(team_home_games_left, teams_away_streak_limit_hit_last_round-1)
                else
                    worst_case_forbidden_games += min(ttp_instance.n-1, teams_away_streak_limit_hit_last_round + teams_home_stand_limit_hit_last_round)
                end
            else
                if state.possible_home_stands[team] == 0
                    worst_case_forbidden_games += team_home_games_left
                    worst_case_forbidden_games += min(team_away_games_left, teams_home_stand_limit_hit_last_round-1)
                else
                    worst_case_forbidden_games += min(ttp_instance.n-1, teams_away_streak_limit_hit_last_round + teams_home_stand_limit_hit_last_round)
                end
            end
        end

        if state.forbidden_opponents[team] != -1
            worst_case_forbidden_games += 1
        end

        return worst_case_forbidden_games < team_away_games_left + team_home_games_left
    end

    # checks whether there would be a team left after game (away_team, home_team), that has no permitted games, a dead team, O(n^2)
    function delta_check_dead_team(ttp_instance::Main.TTPInstance.Instance, node::Main.TTPSolver.TTPStates.Node, away_team::Int64, home_team::Int64)
        state = Main.TTPSolver.TTPStates.update_state(ttp_instance, node.state, away_team, home_team, node.number_of_away_games_left[home_team], node.number_of_home_games_left[away_team])

        games_per_round = convert(Int64, ttp_instance.n/2)
        current_round = convert(Int64, ceil((node.layer+1) / games_per_round))
        games_played_in_this_round = (node.layer+1) % games_per_round

        teams_away_streak_limit_hit_last_round = node.teams_away_streak_limit_hit_last_round
        teams_away_streak_limit_hit_current_round = node.teams_away_streak_limit_hit_current_round
        teams_home_stand_limit_hit_last_round = node.teams_home_stand_limit_hit_last_round
        teams_home_stand_limit_hit_current_round = node.teams_home_stand_limit_hit_current_round
        teams_away_streak_limit_hit_last_round_2 = 0
        teams_away_streak_limit_hit_current_round_2 = 0
        teams_home_stand_limit_hit_last_round_2 = 0
        teams_home_stand_limit_hit_current_round_2 = 0

        if games_played_in_this_round == 1
            teams_home_stand_limit_hit_last_round = teams_home_stand_limit_hit_current_round
            teams_away_streak_limit_hit_last_round = teams_away_streak_limit_hit_current_round
            teams_home_stand_limit_hit_current_round = 0
            teams_away_streak_limit_hit_current_round = 0
        end

        if node.state.possible_home_stands[away_team] == 0
            teams_home_stand_limit_hit_last_round -= 1
        end
        if state.possible_away_streaks[away_team] == 0
            teams_away_streak_limit_hit_current_round += 1
        end
        if state.possible_home_stands[away_team] == 0
            teams_home_stand_limit_hit_current_round += 1
        end
        if node.state.possible_away_streaks[home_team] == 0
            teams_away_streak_limit_hit_last_round -= 1
        end
        if state.possible_home_stands[home_team] == 0
            teams_home_stand_limit_hit_current_round += 1
        end
        if state.possible_away_streaks[home_team] == 0
            teams_away_streak_limit_hit_current_round += 1
        end

        for team in 1:ttp_instance.n
            if team == away_team
                team_away_games_left = node.number_of_away_games_left[team] - 1
            else
                team_away_games_left = node.number_of_away_games_left[team]
            end

            if team == home_team
                team_home_games_left = node.number_of_home_games_left[team] - 1
            else
                team_home_games_left = node.number_of_home_games_left[team]
            end

            team_games_left = team_home_games_left + team_away_games_left

            if team_games_left == 0
                continue
            end

            if delta_infer_no_dead_team(ttp_instance, node.layer+1, state, team, team_home_games_left, team_away_games_left, teams_away_streak_limit_hit_last_round, teams_away_streak_limit_hit_current_round, teams_home_stand_limit_hit_last_round, teams_home_stand_limit_hit_current_round)
                 continue
            end

            witness_found = false
            for opponent in node.away_games_left_by_team[team]
                if state.games_left[team, opponent] && delta_game_allowed(ttp_instance, state, team, opponent, team, opponent)
                    witness_found = true
                    break
                end
            end
            if !witness_found
                for opponent in node.home_games_left_by_team[team]
                     if state.games_left[opponent, team] && delta_game_allowed(ttp_instance, state, team, opponent, opponent, team)
                         witness_found = true
                         break
                     end
                end
            end
            if !witness_found
                #@printf("no witness for %d\n", team)
                return false
            end
        end

        return true
    end

    # count number of permitted games via game_allowed check
    function number_of_permitted_games_for_team_by_check(ttp_instance::Main.TTPInstance.Instance, node::Main.TTPSolver.TTPStates.Node, team::Int64)
        length(filter(x -> future_game_allowed(ttp_instance, node.state, team, x, team, x), node.away_games_left_by_team[team])) + length(filter(x -> future_game_allowed(ttp_instance, node.state, team, x, x, team), node.home_games_left_by_team[team]))
    end

    # check whether game can be played for which both teams have to be in the same round and the at-most/no-repeat constraints must not be violated
    function game_allowed(ttp_instance::Main.TTPInstance.Instance, state::Main.TTPSolver.TTPStates.State, away_team::Int64, home_team::Int64)
        state.rounds[away_team] == state.rounds[home_team] && state.possible_away_streaks[away_team] > 0 && state.possible_home_stands[home_team] > 0 && (!ttp_instance.no_repeat || (state.forbidden_opponents[away_team] != home_team && state.forbidden_opponents[home_team] != away_team))
    end

    # check whether game can be played for which selected team has to be in at least the round of the opponent and the at-most/no-repeat constraints must not be violated
    function future_game_allowed(ttp_instance::Main.TTPInstance.Instance, state::Main.TTPSolver.TTPStates.State, team::Int64, opponent::Int64, away_team::Int64, home_team::Int64)
        state.rounds[team] >= state.rounds[opponent] && state.possible_away_streaks[away_team] > 0 && state.possible_home_stands[home_team] > 0 && (!ttp_instance.no_repeat || (state.forbidden_opponents[away_team] != home_team && state.forbidden_opponents[home_team] != away_team))
    end

    # the incremental variant of game_allowed, where we check whether playing the game (away_team, home_team) would be allowed
    function delta_game_allowed(ttp_instance::Main.TTPInstance.Instance, state::Main.TTPSolver.TTPStates.State, team::Int64, opponent::Int64, away_team::Int64, home_team::Int64)
        if state.rounds[team] < state.rounds[opponent]
            return false
        elseif state.rounds[team] == state.rounds[opponent]
            return game_allowed(ttp_instance, state, away_team, home_team)
        else
            if team == away_team
                return state.possible_away_streaks[team] > 0
            else
                return state.possible_home_stands[team] > 0
            end
        end
    end
end
