# A Randomized Beam Search Approach to the Traveling Tournament Problem

The traveling tournament problem is a difficult sports league scheduling problem introduced by

> Easton, K., Nemhauser, G., Trick, M.: The traveling tournament problem description and benchmarks.
> In: International Conference on Principles and Practice of Constraint Programming.
> LNCS, vol. 2239, pp. 580–584. Springer (2001)

The problem description, benchmark problem instances, and currently best known lower and upper bounds can be found on [Michael Trick's TTP page](http://https://mat.tepper.cmu.edu/TOURN/).

This repository contains a Julia implementation of a beam search approach as presented in

> Frohner, N., Neumann, B., and Raidl, G. R. (2020). A beam search approach to the traveling tournament problem.
> In Paquete, L. and Zarges, C., editors, Evolutionary Computation in Combinatorial Optimization – 20th European Conference,
> EvoCOP 2020, Held as Part of EvoStar 2020, volume 12102 of LNCS, pages 67–82, Sevilla, Spain. Springer.

The code has been tested with Julia 1.4.2 and DataStructures.jl 0.17.19. Google OR-Tools 7.5 with Python 3.7.6 are interfaced via PyCall.jl 1.91.4.

To precalculate the lower bounds for teams' states of an instances, to be saved into a pickled and bz2 compressed numpy array:

> ./ttp_bounds_precalculation.jl insts/circ/circ14.txt 3 data/circ14_cvrph.pkl.bz2 1

To subsequently call the randomized beam search approach with shuffled team ordering and relative noise of 0.001:

> ./ttp_beam_search.jl insts/circ/circ14.txt 3 1 data/circ14_cvrph.pkl.bz2 10000 1 1 0.001 -1

Alternatively, Google OR-Tools can be used to solve the arising capacitated vehicle routing problems on the fly used as guidance for the beam search:

> ./ttp_beam_search_ortools.jl insts/circ/circ14.txt 3 1 10000 1 1 0.001 1 -1