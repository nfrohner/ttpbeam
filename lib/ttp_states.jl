# TTP states related structs and basic functions

module TTPStates
    import Base.hash
    import Base.isequal
    import Base.isless

    export State, Node, update_state, hash, isequal

    # the state with corresponding hash and equality check functions, a state determines the set feasible completions for a corresponding partial schedule
    struct State
        games_left::Array{Bool, 2}
        forbidden_opponents::Array{Int64}
        rounds::Array{Int64}
        positions::Array{Int64}
        possible_away_streaks::Array{Int64}
        possible_home_stands::Array{Int64}
    end

    function hash(state::State, h::UInt)
        hash(state.games_left, hash(state.forbidden_opponents, hash(state.rounds, hash(state.positions, hash(state.possible_away_streaks, hash(state.possible_home_stands))))))
    end

    function isequal(a::State, b::State)
        isequal(a.games_left, b.games_left) && isequal(a.forbidden_opponents, b.forbidden_opponents) && isequal(a.rounds, b.rounds) && isequal(a.possible_away_streaks, b.possible_away_streaks) && isequal(a.positions, b.positions)  && isequal(a.possible_home_stands, b.possible_home_stands)
    end

    # the nodes in the beam search, hold a state, a partial solutions, and caching variables to quickly perform the feasibility/optimality check during construction
    mutable struct Node
        layer::Int64
        shortest_path_length::Int64
        heuristic_estimate::Int64
        games_left::Int64
        state::Union{State, Nothing}
        heuristic_estimates::Array{Int64}
        noise::Float64
        solution::Vector{Tuple{Int64,Int64}}
        number_of_away_games_left::Array{Int64}
        number_of_home_games_left::Array{Int64}
        away_games_left_by_team::Array{Array{Int64}}
        home_games_left_by_team::Array{Array{Int64}}
        teams_away_streak_limit_hit_last_round::Int64
        teams_away_streak_limit_hit_current_round::Int64
        teams_home_stand_limit_hit_last_round::Int64
        teams_home_stand_limit_hit_current_round::Int64
        Node() = new(0, 0, typemax(Int64), 0, nothing, [], 0.0, [], [], [], [], [], 0, 0, 0, 0)
    end

    function isequal(a::Node, b::Node)
        a_val = (a.shortest_path_length + a.heuristic_estimate + a.noise, a.shortest_path_length)
        b_val = (b.shortest_path_length + b.heuristic_estimate + b.noise, b.shortest_path_length)
        isequal(a_val, b_val)
    end

    function isless(a::Node, b::Node)
        a_val = (a.shortest_path_length + a.heuristic_estimate + a.noise, a.shortest_path_length)
        b_val = (b.shortest_path_length + b.heuristic_estimate + b.noise, b.shortest_path_length)
        isless(a_val, b_val)
    end

    # a state transitions copies the existing states and makes corresponding updates determined by the game being played
    function update_state(ttp_instance::Main.TTPInstance.Instance, state::State, away_team::Int64, home_team::Int64, home_team_number_of_away_games_left::Int64, away_team_number_of_home_games_left::Int64)
        games_left = copy(state.games_left)
        games_left[away_team, home_team] = false

        rounds = copy(state.rounds)
        rounds[away_team] += 1
        rounds[home_team] += 1

        positions = copy(state.positions)
        positions[away_team] = home_team
        positions[home_team] = home_team

        possible_away_streaks = copy(state.possible_away_streaks)
        possible_away_streaks[away_team] -= 1
        possible_away_streaks[home_team] = min(ttp_instance.streak_limit, home_team_number_of_away_games_left)

        possible_home_stands = copy(state.possible_home_stands)
        possible_home_stands[away_team] = min(ttp_instance.streak_limit, away_team_number_of_home_games_left)
        possible_home_stands[home_team] -= 1

        forbidden_opponents = copy(state.forbidden_opponents)
        if ttp_instance.no_repeat
            if games_left[home_team, away_team]
                forbidden_opponents[away_team] = home_team
                forbidden_opponents[home_team] = away_team
            else
                forbidden_opponents[away_team] = -1
                forbidden_opponents[home_team] = -1
            end

            for team in 1:ttp_instance.n
                if team != away_team && team != home_team && (forbidden_opponents[team] == away_team || forbidden_opponents[team] == home_team)
                    forbidden_opponents[team] = -1
                end
            end
        end

        State(games_left, forbidden_opponents, rounds, positions, possible_away_streaks, possible_home_stands)
    end
end
