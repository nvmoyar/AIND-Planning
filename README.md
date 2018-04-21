# Artificial Inte# Artificial Intelligence Engineer Nanodegree

## Create a Domain Independent Planner (Implement a Planning Search agent)

[image1]: ./images/planning_screenshot.png "Planning Search agent"
[report]: ./heuristic_analysis.pdf "Heuristics Analysis"

### Project Overview and Discussion Report

The motivation of this project is to solve some deterministic logistic planning problems for an Air Cargo transport system using a planning search agent. With progression search algorithms like those in the navigation problem from lecture, optimal plans for each problem will be computed.  Unlike the navigation problem, there is no simple distance heuristic to aid the agent. 
Instead, we implement domain-independent heuristics.

In this Cargo problems, we assume that no action can satisfy more than one goal, and since goals are satisfied by different actions, so if there are three goal states, there will be necessary at least three actions to satisfy the goal state.  It is also assumed that all step nodes have the same cost. 

All these search algorithms will return a sequence of actions to complete the goal, not always an optimal one. From the AIMA book it is known that in order to evaluate performance these characteristics should be considered:  

* Completeness: Is the algorithm guaranteed to find a solution when there is one?
* Optimality: Does the strategy find the optimal solution? 
* Time complexity:  How long does it take to find a solution? 
* Space complexity: How much memory is needed to perform the search? The more search space, the more space complexity, therefore more memory to perform the operation. 

Our optimal is maybe the less number of operations to complete the goal, therefore ‘plan length’ parameter, but as not all the search functions used in this project are optimal, it has to be considered the nature itself of the algorithm. Node expansions and time elapsed gives an idea of the number of resources consumed.

By the end of this project, we will have the following algorithms: 

* breadth first search: uninformed
* breadth first tree search: uninformed
* depth first graph search: uninformed
* depth limited search: uninformed
* uniform cost search: uninformed
* recursive best first search: theoretically informed, formally uninformed
* greedy best first graph search with h_1: theoretically informed, formally uninformed
* A* search with h_1: theoretically informed, formally uninformed
* A* search with h ignore preconditions: informed (full use of heuristics)
* A* search h pg levelsum: informed (full use of heuristics)

### Comparison of the nature of the different algorigthms implemented

* **Uninformed search**

Among **Breadth-first** and **breadth tree algorithm** , the main difference is an explore set implementation to prevent expanding already explored paths.  Breadth-first tree search algorithm is able to find an optimal solution with more expansions, which is extremely inefficient, as probably some of these additional expansions are loopy paths. However, these both algorithms are optimal since they expand the shallowest nodes, therefore they both guarantee that no deeper nodes in the search space are going to be considered.

Both **Depth-first** and **Depth-limited** algorithms are aimed to find the longest path in a non-optimal way. The main difference is the branching factor due to the imposition of exploring the search space 50 times, producing several redundant paths on the search tree. Depth algorithms are not optimal as they might overlook the goal. Depth-limited has some advantages over Depth-first, since forcing the algorithm to search stop at some level, completeness can be imposed preventing those cases where the branching factor is infinite. However, for these problems complexity, the depth-limited search algorithm has a worse performance as more redundancy is being added.

Regarding the **Uniform cost search**, it expands nodes according to their optimal path cost. Compared to Breadth-first search, uniform-cost search will keep expanding nodes, trying to evaluate if there is any node goal-satisfying with lower cost meanwhile the breadth-first search algorithm will stop early once it finds a goal state.

* **Informed (Heuristic search)**

With the heuristics algorithms, we take advantage of knowing where we could find a goal state. An admissible heuristics is one that never overestimates the cost to reach the goal, however, all the nodes have the same cost in these problems. That makes that the apparently informed searches will act as uninformed as the advantage of knowing the possible goal state is not being used. However, strictly speaking, Recursive Best First search, Greedy Best First Graph search and A-star, are informed searches.  

The greedy algorithm is going to try to get as close as it can to the goal state, overlooking other paths with lesser cost. A* is also a path cost-based algorithm, likewise the uninformed-cost search algorithm, it estimates the cost of reaching a node, but combined to the cost from the node to the goal. In these three cargo problems, A* is used with different planning heuristics: 

* A* +  h_1 which means no heuristics and becomes A* to an uninformed search returning the same results as an Uninformed-cost search algorithm. 

* A* + Ignore-preconditions which drops all preconditions from actions and thus, every action is applicable in every state and the goal can be achieved in only one step. With this heuristics, we get the minimum number of actions that reach the goal state. 

* A* + level-sum is a planning graph heuristics that treat the problem as independent subgoals, returning the sum of the level costs of the goals

Recursive Best First is a memory-bounded search algorithm, similar to recursive depth-first search, but rather than continuing indefinitely down the current path, it uses a limit variable. If the current node exceeds this limit, the recursion unwinds back to the alternative path. This algorithm can provide a solution, in this case, using a memoization technique, in those cases when the search space is so complex, that A* runs out of memory.

### Observations and conclusions

In general, considering the search space for the either of the cargo problems and taking into account that all the paths have the same search cost, it seems reasonable to choose the breadth-first search, as it performs a reasonable number of actions needed with the minimum depth in the search space. It must be noticed that depth-first graph search is able to find a goal state on the way to find the deepest node with fewer node expansions and thus, expending less time. However, it cannot be guaranteed that always would work this way independently of the problem. Certainly, the metrics thrown by depth-first are fair enough for all these problems, though -please see the next section ‘Data’ in order to explore metrics-. Please see [Heuristics Analysis][report], last part 'Data' section. 

Uniform-cost algorithm for P2 and P3, has the better performance despite the number of nodes expanded, due to the implementation of the PriorityQueue, is more expensive to use, than the FIFO queue. Anyway, as the all the node steps have the same cost, in this case, it could be seen the Breadth-first search as the cheapest algorithm from the uninformed list. 

Regarding the ‘heuristics’ algorithms Recursive Best First Search, Greedy Best First Graph Search with h_1 and A* search with h_1, they are formally informed algorithms, but in this case, since this heuristics returns the same value for all nodes, they act like uniformed algorithms. 

Level-sum heuristics is only admissible if the problem is subgoal independent, so it can be considered for P1 and P2. However, for P3 as we have two planes and four cargos, the subgoals are not independent, therefore an optimal solution cannot be guaranteed by this heuristic. The best metrics are provided by the A* search with h_ignore preconditions, since the problem is relaxed, completely ignoring all preconditions whereas level-sum heuristics, incurs in more loops. 

Regarding which algorithm to use, it depends on the problem. Some non-optimal algorithms have a fairly good metrics due to the behavior of the data structures used in the implementation of this problem, producing some ‘lucky’ optimizations.  On the other hand turning back to the origin of this project, ‘Build a domain-independent planner’, the proposed algorithm should be able to solve the problem considering the logic nature of the same, therefore, BFS algorithm seems to be a good option for these problems as we are not discussing the cost of every node path. The best metrics are provided by A* with ignoring preconditions heuristics since the problem becomes relaxed and it can reach a goal state with fewer caveats. 

![Planning Search agent][image1]

### Modules available

Please visit [project's description on Udacity](https://github.com/udacity/AIND-Planning) for further details. 

`my_air_cargo_problems.py`  -> AirCargoProblem Class definition + problems air_cargo_p1, air_cargo_p2, and air_cargo_p3 definitions
`my_planning_graph.py` 		-> Domain independent heuristics
`lp_utils.py`				-> FluentState Class definition
`example_have_cake.py`		-> Contains the HaveCakeProblem Class definition implementation that is explained in the AIMA Book and helps a lot to visualize and go through this problem, really worthy to take a look before takling this problem. 		

### Install and Environment required

[Install](https://github.com/udacity/AIND-Planning)

* Download the aind-universal.yml (on this repo)
* conda env create -f aind-universal.yml to create the environment
* Activate the environment:
		* run source activate aind (OSX & Linux)
		* run activate aind (Windows)  

[Demo](https://github.com/udacity/AIND-Planning#examples-and-testing)
[Usage] python run_search.py -p 3 -s3

