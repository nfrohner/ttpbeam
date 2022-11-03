# TTP CVRP(H) bounds precalculation module by recursive state space traversal and subsequent bottom up sweep

module TTPCVRP
    include("ttp_instance.jl")
    include("ttp_util.jl")

    using DataStructures: Queue, enqueue!, dequeue!, Deque
    import .TTPInstance
    using .TTPUtil
    import Base.hash
    import Base.isequal

    using Printf: @printf

    struct State
        teams_left::BitSet
        position::UInt8
        streak::UInt8
    end

    struct Arc{T}
        destination::T
        weight::UInt16
    end

    mutable struct Node
        layer::UInt16
        shortest_path_length::UInt16
        lower_bound::UInt16
        constrained_lower_bounds::Union{Vector{UInt16}, Nothing}
        parent::Union{Node, Nothing}
        forward_arcs::Vector{Arc{Node}}
        state::Union{State, Nothing}
        Node() = new(0, 0, typemax(UInt16), nothing, nothing, Vector{Arc{Node}}(), nothing)
    end

    function hash(state::State, h::UInt)
        hash(state.teams_left, hash(state.position, hash(state.streak)))
    end

    function isequal(a::State, b::State)
        isequal(a.teams_left, b.teams_left) && isequal(a.position, b.position) && isequal(a.streak, b.streak)
    end

    function move_to_team(ttp_instance::Main.TTPInstance.Instance, node::Node, to_team::UInt8)
        new_node = Node()
        weight = ttp_instance.d[node.state.position, to_team]
        new_node.shortest_path_length = node.shortest_path_length + weight
        new_node.state = State(delete!(copy(node.state.teams_left), to_team), to_team, node.state.streak+1)
        new_node, weight
    end

    function move_to_team_and_home(ttp_instance::Main.TTPInstance.Instance, node::Node, to_team::UInt8, home::UInt8)
        new_node = Node()
        weight = ttp_instance.d[node.state.position, to_team] + ttp_instance.d[to_team, home]
        new_node.shortest_path_length = node.shortest_path_length + weight
        new_node.state = State(delete!(copy(node.state.teams_left), to_team), home, 0)
        new_node, weight
    end

    function incorporate(parent::Node, new_node::Node, weight::UInt16, nodes::Dict{State, Node}, nodes_by_layers::Dict{UInt16, Deque{Node}}, Q::Queue{Node})
        new_node.layer = parent.layer+1
        if haskey(nodes, new_node.state)
            existing_node = nodes[new_node.state]
            if new_node.shortest_path_length < existing_node.shortest_path_length
                existing_node.shortest_path_length = new_node.shortest_path_length
                existing_node.parent = parent
            end
            push!(parent.forward_arcs, Arc{Node}(existing_node, weight))
        else
            nodes[new_node.state] = new_node
            push!(nodes_by_layers[new_node.layer], new_node)
            push!(parent.forward_arcs, Arc{Node}(new_node, weight))
            enqueue!(Q, new_node)
            #push!(Q, new_node)
        end
    end

    function construct(team::UInt8, ttp_instance::Main.TTPInstance.Instance, streak_limit)
        root = Node()
        root.shortest_path_length = 0
        root.state = State(delete!(BitSet(1:ttp_instance.n), team), team, 0)

        terminal = Node()
        terminal.shortest_path_length = typemax(UInt16)
        terminal.lower_bound = 0
        terminal.constrained_lower_bounds = ones(UInt16, ttp_instance.n)*typemax(UInt16)
        terminal.constrained_lower_bounds[1] = 0
        terminal.state = State(BitSet(), team, 0)

        Q = Queue{Node}()
        enqueue!(Q, root)
        #push!(Q, root)

        nodes = Dict{State, Node}()
        nodes[root.state] = root
        nodes[terminal.state] = terminal

        nodes_by_layers = Dict{UInt16, Deque{Node}}(i => Deque{Node}() for i = 0:ttp_instance.n-1)
        push!(nodes_by_layers[0], root)
        push!(nodes_by_layers[ttp_instance.n-1], terminal)

        transitions = 0

        while length(Q) > 0
            node = dequeue!(Q)
            #node = pop!(Q)
            for to_team::UInt8 in node.state.teams_left
                if length(node.state.teams_left) > 1 && node.state.streak < streak_limit - 1
                    new_node, weight = move_to_team(ttp_instance, node, to_team)
                    transitions += 1
                    incorporate(node, new_node, weight, nodes, nodes_by_layers, Q)
                end

                new_node, weight = move_to_team_and_home(ttp_instance, node, to_team, team)
                transitions += 1
                incorporate(node, new_node, weight, nodes, nodes_by_layers, Q)
            end
        end

        @printf("%d transitions\n", transitions)
        @printf("%d nodes\n", length(nodes))

        nodes, nodes_by_layers, terminal.shortest_path_length
    end

    function calculate_bounds_for_teams(ttp_instance::Main.TTPInstance.Instance, bounds_by_state::Array{UInt16,4})
        root_bound_sum = 0

        for team::UInt8 in 1:ttp_instance.n
            @printf("calculating team %d\n", team)
            @time nodes, nodes_by_layers, shortest_path = construct(team, ttp_instance, ttp_instance.streak_limit)

            root_bound_sum += shortest_path

            @time for i = ttp_instance.n-2:-1:0
                for node in nodes_by_layers[i]
                    node.lower_bound = minimum(map(x -> x.destination.lower_bound + x.weight, node.forward_arcs))
                end
            end

            @time for (node_state, node) in nodes
                bounds_by_state[team, mask_teams_left(team, node.state.teams_left), node.state.position, node.state.streak+1] = node.lower_bound
            end
        end

        root_bound_sum
    end

    function sum_with_potential_infinity(a::UInt16, b::UInt16)
        if a == typemax(UInt16) || b == typemax(UInt16)
            typemax(UInt16)
        else
            a + b
        end
    end

    function calculate_bounds_for_teams(ttp_instance::Main.TTPInstance.Instance, bounds_by_state::Array{UInt16,5})
        root_bound_sum = 0

        for team::UInt8 in 1:ttp_instance.n
            @printf("calculating team %d\n", team)
            @time nodes, nodes_by_layers, shortest_path = construct(team, ttp_instance, ttp_instance.streak_limit)

            root_bound_sum += shortest_path

            @time for i = ttp_instance.n-2:-1:0
                for node in nodes_by_layers[i]
                    node.constrained_lower_bounds = ones(UInt16, ttp_instance.n)*typemax(UInt16)
                    if node.state.position == team
                        node.constrained_lower_bounds[2:end] = min.(node.constrained_lower_bounds[2:end], map(x -> [sum_with_potential_infinity(x.destination.constrained_lower_bounds[i], x.weight) for i in 1:ttp_instance.n-1], node.forward_arcs)...)
                    else
                        node.constrained_lower_bounds = min.(node.constrained_lower_bounds, map(x -> [sum_with_potential_infinity(x.destination.constrained_lower_bounds[i], x.weight) for i in 1:ttp_instance.n], node.forward_arcs)...)
                    end
                end
            end

            @time for (node_state, node) in nodes
                bounds_by_state[team, mask_teams_left(team, node.state.teams_left), node.state.position, node.state.streak+1, :] = node.constrained_lower_bounds
            end

            flush(stdout)
        end

        root_bound_sum
    end
end
