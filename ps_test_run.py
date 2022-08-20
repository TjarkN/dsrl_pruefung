import time

from ps_environment import Environment
from agents.q_learning_agent_mas import QLearningAgentMAS
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim

# doubleclick object in PlantSim and lookup the path_context
# socket is the name of the socket object in PlantSim or None if not used
# model = 'D:\Tjark\Dokumente\FH Bielefeld\Sommersemester 2022\Diskrete Simulation und Reinforceent Learning\Github Repo\plantsim_working\MiniFlow_BE_based_MAS.spp'
model = r'C:\Users\dlina\DSRL\DSRL_Pruefung.spp'
plantsim = Plantsim(version='16.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=False)

# do this after successful training (ps_training.py)

# test_agent#
env = Environment(plantsim)
agent = QLearningAgentMAS(env.problem)
q_table = agent.load_q_table("agents/q_table.npy")
agent.q_table = q_table # todo not sure if this is working
performance_test = []
number_of_tests = 20
it = 0
while it < number_of_tests:
    print(it)
    it += 1
    t = time.time()
    while not env.problem.is_goal_state(env.problem):
        action = agent.act()
        if action is not None:
            env.problem.act(action)
    run_time = time.time() - t
    print(run_time)
    performance_test.append(run_time)
    env.reset()

N = int(number_of_tests/10)
x = np.array(performance_test)
moving_average = np.convolve(x, np.ones(N)/N, mode='valid')
plt.plot(performance_test)
plt.plot(moving_average)
plt.show()

plantsim.quit()
