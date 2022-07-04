from ps_environment import Environment
#from agents.q_learning_agent_mas import QLearningAgentMAS
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim

model = 'C:\\0 Eigene Dateien\9 Lehre\PlantSimulationProjects\MiniFlow_BE_based_MAS.spp'
plantsim = Plantsim(version='16.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=True)
