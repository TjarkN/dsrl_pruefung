import time

from ps_environment import Environment
from agents.q_learning_agent_mas import QLearningAgentMAS
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim

# doubleclick object in PlantSim and lookup the path_context
# socket is the name of the socket object in PlantSim or None if not used
model = "F:\Tjark\Dokumente\FH Bielefeld\Master\SoSe2022\Diskrete Simulation und Reinforcement Learning\Pruefung\dsrl_git\DSRL_Pruefung_1000.spp"
#model = 'D:\Tjark\Dokumente\FH Bielefeld\Sommersemester 2022\Diskrete Simulation und Reinforceent Learning\Pruefung\pruefung_git\DSRL_Pruefung_1000.spp'
#model = r'C:\Users\dlina\DSRL\DSRL_Pruefung_1000.spp'
plantsim = Plantsim(version='16.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=True)

# do this after successful training (ps_training.py)

# test_agent#
env = Environment(plantsim)
agent = QLearningAgentMAS(env.problem)
agent.load_q_table("agents/q_table_2608_1704_100_reward12.npy")
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
