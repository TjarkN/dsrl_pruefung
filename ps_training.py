import time
from time import sleep

from ps_environment import Environment
from agents.q_learning_agent_mas import QLearningAgentMAS
from agents.deep_q_learning_agent import DeepQLearningAgent
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim

# doubleclick object in PlantSim and lookup the path_context
# socket is the name of the socket object in PlantSim or None if not used
#model = 'D:\Tjark\Dokumente\FH Bielefeld\Sommersemester 2022\Diskrete Simulation und Reinforceent Learning\Pruefung\pruefung_git\DSRL_Pruefung.spp'
model = r'C:\Users\dlina\DSRL\DSRL_Pruefung.spp'
plantsim = Plantsim(version='16.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=True)

if not plantsim.plantsim.IsSimulationRunning():
    plantsim.start_simulation()

# todo info exit count aus anderer tabelle holen
# todo ausgangsmethode in der quelle, damit nicht zu viele spiele im umlauf sind

# set max number of iterations

max_iterations = 100
it = 0
env = Environment(plantsim)
agent = QLearningAgentMAS(env.problem, max_N_exploration=0.1) # Environment -> plantsimproblem -> plantsim
#agent.load_q_table("agents/q_table_2208_1700_500_reward100.npy")
#agent = DeepQLearningAgent(env.problem) # todo change current state into nummeric / integer values
performance_train = []
q_table = None # todo qtable ggf. weitertrainieren mit load q table
# training
while it < max_iterations:
    print("=== " + str(it) + " ===")
    it += 1

    # pause python until plantsim reached an actual state (game at entry of agent pick and place)
    active = plantsim.get_value("sync[\"isPythonActive\",1]")
    while not active:
        sleep(0.01)
        active = plantsim.get_value("sync[\"isPythonActive\",1]")

    t = time.time()
    q_table, N_sa = agent.train() # todo print reward eval
    run_time = time.time() - t
    if it % 100 == 0:
        print(f"Q-Table:\n{q_table}")
    print(f"Runtime Python: {run_time}")
    evaluation = env.problem.evaluation
    performance_train.append(evaluation) # evaluation)
    env.reset()

# save q_table
agent.save_q_table("agents/q_table_2508_1100_100_newreward50.npy")

# plot results
x = np.array(performance_train)
N = int(max_iterations/10)
moving_average = np.convolve(x, np.ones(N)/N, mode='valid')
plt.plot(performance_train)
plt.plot(moving_average)
plt.show()

# # test
# if not plantsim.plantsim.IsSimulationRunning():
#     plantsim.start_simulation()
#
# performance_test = []
# number_of_tests = 10
# it = 0
# agent.load_q_table("agents/q_table.npy")
# while it < number_of_tests:
#     print(it)
#     it += 1
#
#     active = plantsim.get_value("sync[\"isPythonActive\",1]")
#     while not active:
#         sleep(0.01)
#         active = plantsim.get_value("sync[\"isPythonActive\",1]")
#
#     t = time.time()
#     while not env.problem.is_goal_state(env.problem):
#         action = agent.act()
#         #if action is not None:
#         env.problem.act(action)
#
#     run_time = time.time() - t
#     print(run_time)
#     performance_test.append(run_time)
#     env.reset()
#
# N = int(number_of_tests/10)
# x = np.array(performance_test)
# moving_average = np.convolve(x, np.ones(N)/N, mode='valid')
# plt.plot(performance_test)
# plt.plot(moving_average)
# plt.show()

plantsim.quit()
