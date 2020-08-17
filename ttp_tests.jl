#!/usr/bin/env julia
# testsets for different parts of the functionality

using Test

@testset "heuristic estimates" begin
    include("lib/ttp_instance.jl")
    include("lib/ttp_node_estimators.jl")

    import .TTPInstance
    import .TTPNodeEstimators

    ttp_instance = TTPInstance.read("insts/NL/nl8.txt", 3, true)

    @test TTPNodeEstimators.min_vehicles_by_home_games(ttp_instance, 1, [2,3,4,5,6,7,8], 7, 1, 0) == 2
    @test TTPNodeEstimators.min_vehicles_by_home_games(ttp_instance, 1, [2,3,4,5,6,7,8], 5, 1, 2) == 2
    @test TTPNodeEstimators.min_vehicles_by_home_games(ttp_instance, 1, [6,7,8], 2, 1, 1) == 0
    @test TTPNodeEstimators.min_vehicles_by_home_games(ttp_instance, 1, [6,7,8], 2, 1, 2) == 1
    @test TTPNodeEstimators.min_vehicles_by_home_games(ttp_instance, 1, [4,5,6,7,8], 7, 3, 2) == 3
    @test TTPNodeEstimators.min_vehicles_by_away_games(ttp_instance, 1, [2,3,4,5,6,7,8], 7, 1, 0) == 3
    @test TTPNodeEstimators.min_vehicles_by_away_games(ttp_instance, 1, [4,5,6,7,8], 7, 3, 2) == 3
    @test TTPNodeEstimators.min_vehicles_by_away_games(ttp_instance, 1, [5,6,7,8], 6, 3, 0) == 2
    @test TTPNodeEstimators.max_vehicles_by_away_games(ttp_instance, 1, [2,3,4,5,6,7,8], 7, 1, 0) == 4
    @test TTPNodeEstimators.max_vehicles_by_away_games(ttp_instance, 1, [3,4,5,6,7], 6, 2, 1) == 3
    @test TTPNodeEstimators.max_vehicles_by_away_games(ttp_instance, 1, [4,5,6,7], 6, 3, 2) == 3
    @test TTPNodeEstimators.max_vehicles_by_home_games(ttp_instance, 1, [2,3,4,5,6,7,8], 7, 1, 0) == 8
    @test TTPNodeEstimators.max_vehicles_by_home_games(ttp_instance, 1, [3,4,5,6,7], 6, 2, 1) == 7

    @test TTPNodeEstimators.min_max_vehicles(ttp_instance, 1, [2,3,4,5,6,7,8], 7, 1, 0) == (3, 4)
    @test TTPNodeEstimators.min_max_home_stands(ttp_instance, 1, [2,3,4,5,6,7,8], 7, 1, 0) == (4, 5)
    @test TTPNodeEstimators.min_max_home_stands(ttp_instance, 1, [2,3,4,5,6,7,8], 4, 1, 3) == (4, 5)
    @test TTPNodeEstimators.min_max_vehicles(ttp_instance, 1, [4,5,6,7], 6, 3, 2) == (2, 3)
    @test TTPNodeEstimators.min_max_vehicles(ttp_instance, 1, [4], 1, 3, 2) == (1, 2)
    @test TTPNodeEstimators.min_max_home_stands(ttp_instance, 1, [4], 1, 3, 2) == (1, 2)

    ttp_instance = TTPInstance.read("insts/NL/nl10.txt", 3, true)

    @test TTPNodeEstimators.min_vehicles_by_home_games(ttp_instance, 1, [6,7,8,9,10], 7, 5, 1) == 3
    @test TTPNodeEstimators.min_vehicles_by_away_games(ttp_instance, 1, [6,7,8,9,10], 7, 5, 1) == 2
    @test TTPNodeEstimators.min_max_vehicles(ttp_instance, 1, [6,7,8,9,10], 7, 5, 1) == (3, 3)
    @test TTPNodeEstimators.min_max_home_stands(ttp_instance, 1, [6,7,8,9,10], 7, 5, 1) == (3, 3)
end
