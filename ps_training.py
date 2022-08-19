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

# set max number of iterations

max_iterations = 500
it = 0
env = Environment(plantsim)
agent = QLearningAgentMAS(env.problem) # Environment -> plantsimproblem -> plantsim
performance_train = []
q_table = None
# training
while it < max_iterations:
    print(it)
    it += 1
    t = time.time()
    q_table, N_sa = agent.train()
    run_time = time.time() - t
    print(run_time)
    # evaluation = env.problem.evaluation
    performance_train.append(run_time) # evaluation)
    env.reset()

# plot results
x = np.array(performance_train)
N = int(max_iterations/10)
moving_average = np.convolve(x, np.ones(N)/N, mode='valid')
plt.plot(performance_train)
plt.plot(moving_average)
plt.show()

# save q_table
agent.save_q_table("agents/q_table.npy")

plantsim.quit()
