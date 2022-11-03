# TTP instance module

module TTPInstance
    using DelimitedFiles

    struct Instance
        n::Int
        d::Array{UInt16, 2}
        streak_limit::Int
        no_repeat::Bool
    end

    function read(file_name::String, streak_limit::Integer, no_repeat::Bool)
        d = readdlm(file_name, UInt16)
        n = size(d)[1]
        Instance(n, d, streak_limit, no_repeat)
    end
end
