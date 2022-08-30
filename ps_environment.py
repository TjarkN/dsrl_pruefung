from plantsim.plantsim import Plantsim
from problem import Problem
from random import seed
import itertools

class PlantSimulationProblem(Problem):
    """
    Problem for the agent
    """
    def __init__(self, plantsim: Plantsim, states=None, actions=None, id=None, evaluation=0, goal_state=False):
        """
        init method
        """
        self.plantsim = plantsim
        if actions is not None:
            self.actions = actions
        else:
            self.actions = self.plantsim.get_object("Actions").get_columns_by_header("name")
        if states is not None:
            self.states = states
        else:
            self.states = {}
            states = self.plantsim.get_object("States")
            for header in states.header:
                if header != "Index":
                    self.states[header] = states.get_columns_by_header(header)
                    # removing empty cells
                    self.states[header] = list(filter(None, self.states[header]))
        self.old_state = None
        self.state = None
        self.count_exit = 0
        self.old_count_exit = 0
        self.id = id
        self.evaluation = evaluation
        self.goal_state = goal_state
        self.next_event = True

    def copy(self):
        """
        Returns a copy of itself. This is done by creating a new PlantSimulationProblem-object and returning it
        """
        ps_copy = PlantSimulationProblem(self.plantsim, self.state.copy(), self.actions[:], self.id, self.evaluation,
                                         self.goal_state)
        return ps_copy

    def act(self, action):
        """
        Writes the action in the ActionControl table of the plantsim-model.
        After that it restarts the agent and the simulation.
        """
        self.plantsim.set_value("ActionControl[\"id\",1]", self.id)
        self.plantsim.set_value("ActionControl[\"action\",1]", action)

        self.plantsim.set_value("sync[\"isPythonActive\",1]", False)
        self.plantsim.execute_simtalk('restartAgent')
        self.next_event = True

    def to_state(self):
        return tuple(self.state)

    def is_goal_state(self, state):
        """
        returns if the given state is a goal state
        """
        return state.goal_state

    def get_current_state(self):
        """
        returns the current state of the plantsim-model
        """
        if self.next_event:
            self.old_state = self.state
            self.state = []
            #states = self.plantsim.get_next_message()
            current_state = self.plantsim.get_current_state()
            for key, value in current_state.items():
                if key == "id":
                    self.id = value
                elif key == "count_exit":
                    self.old_count_exit = self.count_exit
                    self.count_exit = value
                elif key == "goal_state":
                    self.goal_state = value
                elif key == "simulation_time":
                    self.evaluation = value
                else:
                    self.state.append(value)
            self.next_event = False
        return self

    def eval(self, state):
        """
        returns costs of 1 if the exit count did not change, otherwise it returns a positive reward of 11
        """
        costs = 1
        if state.old_count_exit != state.count_exit: # todo get this info from content storage table
            # if something
            costs -= 19 # todo set back to 12 maybe
        elif state.old_state != None:
            # after the first step there is an old state
            # if nothing was put into the exit
            # compare content of the storages
            if state.old_state[1] != state.state[1] or state.old_state[2] != state.state[2]:
                costs += 2
        return costs

    def get_all_actions(self):
        return self.actions

    def get_all_states(self):
        all_states = list(itertools.product(*list(self.states.values())))
        all_states = [tuple(x) for x in all_states]
        return all_states

    def get_reward(self, state):
        reward = -self.eval(state)
        return reward

    def reset(self):
        self.state = None
        self.id = None
        self.count_exit = 0
        self.goal_state = False
        self.next_event = True


class Environment:
    """
    Environment for the agent
    """
    def __init__(self, plantsim: Plantsim, seed_value=1):
        if seed_value is not None:
            seed(seed_value)
        plantsim.reset_simulation()
        self.problem = PlantSimulationProblem(plantsim)
        plantsim.start_simulation()

    def reset(self):
        self.problem.plantsim.execute_simtalk("reset")
        self.problem.plantsim.reset_simulation()
        self.problem.reset()
        self.problem.plantsim.start_simulation()