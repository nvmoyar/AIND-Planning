from aimacode.logic import PropKB
from aimacode.planning import Action
from aimacode.search import (
    Node, Problem,
)
from aimacode.utils import expr
from lp_utils import (
    FluentState, encode_state, decode_state,
)
from my_planning_graph import PlanningGraph

from functools import lru_cache


class AirCargoProblem(Problem):
    def __init__(self, cargos, planes, airports, initial: FluentState, goal: list):
        """

        :param cargos: list of str
            cargos in the problem
        :param planes: list of str
            planes in the problem
        :param airports: list of str
            airports in the problem
        :param initial: FluentState object
            positive and negative literal fluents (as expr) describing initial state
        :param goal: list of expr
            literal fluents required for goal test
        """
        self.state_map = initial.pos + initial.neg
        self.initial_state_TF = encode_state(initial, self.state_map)
        Problem.__init__(self, self.initial_state_TF, goal=goal)
        self.cargos = cargos
        self.planes = planes
        self.airports = airports
        self.actions_list = self.get_actions() # load() and unload() subfunctions

    def get_actions(self):
        """
        This method creates concrete actions (no variables) for all actions in the problem
        domain action schema and turns them into complete Action objects as defined in the
        aimacode.planning module. It is computationally expensive to call this method directly;
        however, it is called in the constructor and the results cached in the `actions_list` property.

        Returns:
        ----------
        list<Action>
            list of Action objects
        """

        # TODO create concrete Action objects based on the domain action schema for: Load, Unload, and Fly
        # concrete actions definition: specific literal action that does not include variables as with the schema
        # for example, the action schema 'Load(c, p, a)' can represent the concrete actions 'Load(C1, P1, SFO)'
        # or 'Load(C2, P2, JFK)'.  The actions for the planning problem must be concrete because the problems in
        # forward search and Planning Graphs must use Propositional Logic

        def load_actions():
            """Create all concrete Load actions and return a list

            :return: list of Action objects
            """
            loads = []
            
            for airport in self.airports:
                for plane in self.planes: 
                    for cargo in self.cargos: 
                        # cargo is at airport, plane is at airport. We don't need to define airport, cargo and plane as there
                        # are declared on the class definition

                        precond_pos = [expr("At({}, {})".format(cargo, airport)), 
                                       expr("At({}, {})".format(plane, airport))]
                        precond_neg = []

                        effect_add = [expr("In({}, {})".format(cargo, plane))] # In(c, p))
                        effect_rem = [expr("At({}, {})".format(cargo, airport))] # ¬ At(c, a)

                        # Action(Load(c, p, a), [Precond], [Effect]:
                        load = Action(expr("Load({}, {}, {})".format(cargo, plane, airport)),
                            [precond_pos, precond_neg],
                            [effect_add, effect_rem])
                        
                        # Add every action to loads list
                        loads.append(load)

            return loads

        def unload_actions():
            """Create all concrete Unload actions and return a list

            :return: list of Action objects
            """
            unloads = []
            
            for airport in self.airports:
                for plane in self.planes:
                    for cargo in self.cargos:
                        precond_pos = [expr("In({}, {})".format(cargo, plane)), expr("At({}, {})".format(plane, airport))]
                        precond_neg = []
                        effect_add = [expr("At({}, {})".format(cargo, airport))]
                        effect_rem = [expr("In({}, {})".format(cargo, plane))]

                        unload = Action(expr("Unload({}, {}, {})".format(cargo, plane, airport)),
                            [precond_pos, precond_neg],
                            [effect_add, effect_rem])
                        unloads.append(unload)
            
            return unloads

        def fly_actions():
            """Create all concrete Fly actions and return a list

            :return: list of Action objects
            """
            flys = []
            for fr in self.airports:
                for to in self.airports:
                    if fr != to:
                        for p in self.planes:
                            precond_pos = [expr("At({}, {})".format(p, fr)),
                                           ]
                            precond_neg = []
                            effect_add = [expr("At({}, {})".format(p, to))]
                            effect_rem = [expr("At({}, {})".format(p, fr))]
                            fly = Action(expr("Fly({}, {}, {})".format(p, fr, to)),
                                         [precond_pos, precond_neg],
                                         [effect_add, effect_rem])
                            flys.append(fly)
            return flys

        return load_actions() + unload_actions() + fly_actions()

    def actions(self, state: str) -> list:
        """ Return the actions that can be executed in the given state.

        :param state: str
            state represented as T/F string of mapped fluents (state variables)
            e.g. 'FTTTFF'
        :return: list of Action objects
        """
        
        possible_actions = []
        kb = PropKB()  # problem definition using propositional logic. Knowledge-based agent declaration
        kb.tell(decode_state(state, self.state_map).pos_sentence()) # Add the sentence's clauses to the KB
        for action in self.actions_list: # for each action in action_list (get_actions()--> load(), unload())
            is_possible = True
            for clause in action.precond_pos:
                if clause not in kb.clauses:
                    is_possible = False
            for clause in action.precond_neg:
                if clause in kb.clauses:
                    is_possible = False
            if is_possible:
                possible_actions.append(action)
        
        return possible_actions #list based on the existing preconditions 

    def result(self, state: str, action: Action):
        """ Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state).

        :param state: state entering node
        :param action: Action applied
        :return: resulting state after action
        """
        
        # from AIMA 12.3.3 Physical objects can be viewed as generalized events -a chunk space-time-, not as the 
        # object itself, but the events that made the object possible throughtout space and time. We can describe the 
        # changing properties of the object using state fluents. Fluent, synonym of state vriable, to refer an aspecto of the 
        # world that changes. 

        new_state = FluentState([], [])
        old_state = decode_state(state, self.state_map)
        for fluent in old_state.pos:
            if fluent not in action.effect_rem:
                new_state.pos.append(fluent)
        for fluent in action.effect_add:
            if fluent not in new_state.pos:
                new_state.pos.append(fluent)
        for fluent in old_state.neg:
            if fluent not in action.effect_add:
                new_state.neg.append(fluent)
        for fluent in action.effect_rem:
            if fluent not in new_state.neg:
                new_state.neg.append(fluent)
        return encode_state(new_state, self.state_map)


    def goal_test(self, state: str) -> bool:
        """ Test the state to see if goal is reached

        :param state: str representing state
        :return: bool
        """
        kb = PropKB()
        kb.tell(decode_state(state, self.state_map).pos_sentence())
        for clause in self.goal:
            if clause not in kb.clauses:
                return False
        return True

    def h_1(self, node: Node):
        # note that this is not a true heuristic
        h_const = 1
        return h_const

    @lru_cache(maxsize=8192)
    def h_pg_levelsum(self, node: Node):
        """This heuristic uses a planning graph representation of the problem
        state space to estimate the sum of all actions that must be carried
        out from the current state in order to satisfy each individual goal
        condition.
        """
        # requires implemented PlanningGraph class
        pg = PlanningGraph(self, node.state)
        pg_levelsum = pg.h_levelsum()
        return pg_levelsum

    @lru_cache(maxsize=8192)
    def h_ignore_preconditions(self, node: Node):
        """This heuristic estimates the minimum number of actions that must be
        carried out from the current state in order to satisfy all of the goal
        conditions by ignoring the preconditions required for an action to be
        executed.
        """
        count = 0
        kb = PropKB()
        kb.tell(decode_state(node.state, self.state_map).pos_sentence())

        for clause in self.goal:
            if clause not in kb.clauses:
                count += 1
        return count

def air_cargo_p1() -> AirCargoProblem:
    """
    All problems are in the Air Cargo domain. 
    Initial states and goals for Air Cargo Problem 1
    PDDL description of this air cargo transportation planning problem:

    Init(At(C1, SFO) ∧ At(C2, JFK) 
    ∧ At(P1, SFO) ∧ At(P2, JFK) 

    ∧ Cargo(C1) ∧ Cargo(C2) 
    ∧ Plane(P1) ∧ Plane(P2)
    ∧ Airport(JFK) ∧ Airport(SFO))

    Goal(At(C1, JFK) ∧ At(C2, SFO))
    """
    # Objects definition: cargos, plane and airports, as lists
    cargos = ['C1', 'C2'] 
    planes = ['P1', 'P2']
    airports = ['JFK', 'SFO']

    # Positive preconditions (to satisfy)
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]

    # Negative preconditions:         
    neg = [expr('At(C2, SFO)'), # negative initally, but it satisfies the goal
           expr('In(C2, P1)'),  # C2 could be in P1, 
           expr('In(C2, P2)'),  # C2 could be in P2 ...the cargo is not in the plain, is at the airport 
           expr('At(C1, JFK)'), # initially cargo1 is not at JFK as is at SFO 
           expr('In(C1, P1)'),  # C1 could be in P1
           expr('In(C1, P2)'),  # or could be in P2 
           expr('At(P1, JFK)'), # initially P1 is not at JFK (as it's at SFO) 
           expr('At(P2, SFO)'), # initially P2 is not at SFO as it's at JFK
           ]
    init = FluentState(pos, neg)


    # the goal only indicates to switch the cargos btw airports, initally C1 is at SFO and the goal is C1 at JFK. 
    # Actually this cargo could be transported by any plane at the airport (precond: plane has to be at the airport)
    # However, the plane we have available is  P1, but the goal doesn't specify which plane has to carry the cargo. 
    
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p2() -> AirCargoProblem:
    """
    All problems are in the Air Cargo domain. 
    Initial states and goals for Air Cargo Problem 1
    PDDL description of this air cargo transportation planning problem:

    Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) 
    ∧ At(P1, SFO) ∧ At(P2, JFK) ∧ At(P3, ATL) 

    ∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3)
    ∧ Plane(P1) ∧ Plane(P2) ∧ Plane(P3)
    ∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL))

    Goal(At(C1, JFK) ∧ At(C2, SFO) ∧ At(C3, SFO))
    """
    # Objects definition: cargos, plane and airports, as lists
    cargos = ['C1', 'C2', 'C3'] 
    planes = ['P1', 'P2', 'P3']
    airports = ['JFK', 'SFO', 'ATL']

    # Positive preconditions (to satisfy)
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           expr('At(P3, ATL)'),
           ]


    # Negative preconditions - possible states 

    neg = [expr('At(C2, SFO)'), # goal
           expr('At(C2, ATL)'), 
           expr('In(C2, P1)'),  
           expr('In(C2, P2)'),  
           expr('In(C2, P3)'),
          
           expr('At(C1, JFK)'), # goal
           expr('At(C1, ATL)'),
           expr('In(C1, P1)'), 
           expr('In(C1, P2)'),
           expr('In(C1, P3)'),

           expr('At(C3, JFK)'),
           expr('At(C3, SFO)'), # goal
           expr('In(C3, P1)'), 
           expr('In(C3, P2)'),
           expr('In(C3, P3)'),
           
           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P3, SFO)'),
           expr('At(P3, JFK)'),
           ]       

    init = FluentState(pos, neg)       

    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


def air_cargo_p3() -> AirCargoProblem:
    """
    All problems are in the Air Cargo domain. 
    Initial states and goals for Air Cargo Problem 1
    PDDL description of this air cargo transportation planning problem:
    
    Init(At(C1, SFO) ∧ At(C2, JFK) ∧ At(C3, ATL) ∧ At(C4, ORD) 
    ∧ At(P1, SFO) ∧ At(P2, JFK) 

    ∧ Cargo(C1) ∧ Cargo(C2) ∧ Cargo(C3) ∧ Cargo(C4)
    ∧ Plane(P1) ∧ Plane(P2)
    ∧ Airport(JFK) ∧ Airport(SFO) ∧ Airport(ATL) ∧ Airport(ORD))
    
    Goal(At(C1, JFK) ∧ At(C3, JFK) ∧ At(C2, SFO) ∧ At(C4, SFO))
    """

    # Objects definition: cargos, plane and airports, as lists
    cargos = ['C1', 'C2', 'C3', 'C4'] 
    planes = ['P1', 'P2'] # two planes to carry/transport four cargos from/to four airports
    airports = ['JFK', 'SFO', 'ATL', 'ORD']

    # Positive preconditions (to satisfy)
    pos = [expr('At(C1, SFO)'),
           expr('At(C2, JFK)'),
           expr('At(C3, ATL)'),
           expr('At(C4, ORD)'),
           expr('At(P1, SFO)'),
           expr('At(P2, JFK)'),
           ]

    # Negative preconditions - possible states 

    neg = [expr('At(C2, SFO)'), # Goal
           expr('At(C2, ATL)'),
           expr('At(C2, ORD)'),  
           expr('In(C2, P1)'),  
           expr('In(C2, P2)'),  
    
          
           expr('At(C1, JFK)'), # Goal
           expr('At(C1, ATL)'),
           expr('At(C1, ORD)'),
           expr('In(C1, P1)'), 
           expr('In(C1, P2)'),
         

           expr('At(C3, JFK)'), # Goal
           expr('At(C3, SFO)'),
           expr('At(C3, ORD)'), 
           expr('In(C3, P1)'), 
           expr('In(C3, P2)'),
        
           expr('At(C4, JFK)'),
           expr('At(C4, SFO)'), # Goal
           expr('At(C4, ATL)'), 
           expr('In(C4, P1)'), 
           expr('In(C4, P2)'), 
           
           expr('At(P1, JFK)'),
           expr('At(P1, ATL)'),
           expr('At(P1, ORD)'),
           expr('At(P2, SFO)'),
           expr('At(P2, ATL)'),
           expr('At(P2, ORD)'),
           ]       

    init = FluentState(pos, neg)       


    # Goal(At(C1, JFK) ∧ At(C3, JFK) ∧ At(C2, SFO) ∧ At(C4, SFO))
    goal = [expr('At(C1, JFK)'),
            expr('At(C2, SFO)'),
            expr('At(C3, JFK)'),
            expr('At(C4, SFO)'),
            ]
    return AirCargoProblem(cargos, planes, airports, init, goal)


