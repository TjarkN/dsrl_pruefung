from ps_environment import Environment
#from agents.q_learning_agent_mas import QLearningAgentMAS
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim

# doubleclick object in PlantSim and lookup the path_context
# socket is the name of the socket object in PlantSim or None if not used
model = 'D:\Tjark\Dokumente\FH Bielefeld\Sommersemester 2022\Diskrete Simulation und Reinforceent Learning\Github Repo\plantsim_working\MiniFlow_BE_based_MAS.spp'
plantsim = Plantsim(version='16.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=False)

plantsim.quit()
