# TTP utility functions

module TTPUtil
    using PyCall

    export basename, mask_teams_left, save_numpy_pickle, load_numpy_pickle

    function basename(path::String, suffix::String)
        (n,e) = splitext(Base.basename(path))
        suffix==e ? n : path
    end

    function mask_teams_left(team, teams_left)
        set_mask = 0
        for away_team in teams_left
            if away_team < team
                set_mask += 2 ^ (away_team-1)
            else
                set_mask += 2 ^ (away_team-2)
            end
        end
        set_mask+1
    end

    function load_numpy_pickle(numpy_pickle_file)
        bz2 = pyimport("bz2")
        pickle = pyimport("pickle")
        numpy = pyimport("numpy")

        sfile = bz2.BZ2File(numpy_pickle_file, "r")
        bounds_by_state = pickle.load(sfile)
        sfile.close()

        return bounds_by_state
    end

    function save_numpy_pickle(numpy_pickle_file, bounds_by_state)
        bz2 = pyimport("bz2")
        pickle = pyimport("pickle")
        numpy = pyimport("numpy")

        sfile = bz2.BZ2File(numpy_pickle_file, "w")
        pickle.dump(numpy.array(bounds_by_state), sfile, protocol=4)
        sfile.close()
    end
end
