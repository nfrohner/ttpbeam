# A Randomized Beam Search Approach to the Traveling Tournament Problem

The traveling tournament problem is a difficult sports league scheduling problem introduced by

> Easton, K., Nemhauser, G., Trick, M.: The traveling tournament problem description and benchmarks.
> In: International Conference on Principles and Practice of Constraint Programming.
> LNCS, vol. 2239, pp. 580–584. Springer (2001)

The problem description, benchmark problem instances, and currently best known lower and upper bounds can be found on [Michael Trick's TTP page](https://mat.tepper.cmu.edu/TOURN/) and more recently at the [RobinX repository](https://www.sportscheduling.ugent.be/RobinX/).

This repository contains a Julia implementation of a beam search approach as presented in [1].

The code has been tested with Julia 1.7.2 and DataStructures.jl 0.18.3. Google OR-Tools 9.4 with Python 3.9 are interfaced via PyCall.jl.

To precalculate the lower bounds for teams' states of an instances (aka disjoint pattern database, similar as done by [2,3]), to be saved into a pickled and bz2 compressed numpy array:

> julia ttp_bounds_precalculation.jl insts/circ/circ14.txt 3 data/circ14_cvrph.pkl.bz2 true

To subsequently call the randomized beam search approach with shuffled team ordering and relative noise of 0.001:

> julia ttp_beam_search.jl insts/circ/circ14.txt 3 true data/circ14_cvrph.pkl.bz2 10000 true random 0.001 -1 false

A final feasible local search using the TTSA neighborhoods [3] can be activated by setting the last parameter to true.

There is also a [parallel beam search](https://github.com/nfrohner/parbeam) implementation for the TTP also faster in single threaded mode

Alternatively, Google OR-Tools can be used to solve the arising capacitated vehicle routing problems on the fly used as guidance for the beam search and keep already solved problems in a cache:

> julia ttp_beam_search_ortools.jl insts/circ/circ14.txt 3 true 16384 true lexicographic none 0.0 false -1

For the latter, there is also an iterative variant, which increases the beam width by a factor every number of runs until either a time or maximum beam width is hit:

> julia ttp_beam_search_ortools_iter.jl insts/NL/nl10.txt 3 true 3600 128 32768 2 true random none 0.001 2 true -1


## References

[1]
> Frohner, N., Neumann, B., and Raidl, G. R. (2020). A beam search approach to the traveling tournament problem.
> In Paquete, L. and Zarges, C., editors, Evolutionary Computation in Combinatorial Optimization – 20th European Conference,
> EvoCOP 2020, Held as Part of EvoStar 2020, volume 12102 of LNCS, pages 67–82, Sevilla, Spain. Springer.

[2]
> David C Uthus, Patricia J Riddle, and Hans W Guesgen.
> DFS* and the traveling tournament problem.
> In International Conference on AI and OR Techniques in Constraint Programming for Combinatorial Optimization Problems volume 5547 of LNCS, pages 279–293. Springer, 2009.

[3]
> David C Uthus, Patricia J Riddle, and Hans W Guesgen.
> Solving the traveling tournament problem with iterative-deepening A*.
> Journal of Scheduling, 15(5): 601–614, 2012.

[4]
> Aris Anagnostopoulos, Laurent Michel, Pascal Van Hentenryck, and Yannis Vergados.
> A simulated annealing approach to the traveling tournament problem.
> Journal of Scheduling, 9(2):177–193, 2006.