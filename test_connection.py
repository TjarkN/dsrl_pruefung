from time import sleep

from ps_environment import Environment
#from agents.q_learning_agent_mas import QLearningAgentMAS
import matplotlib.pyplot as plt
import numpy as np
from plantsim.plantsim import Plantsim


def test_run(iterations, plantsim):
    i = 0
    while i <= iterations:
        i+= 1
        action = np.random.choice(["move_to_lager1","move_to_lager2","move_to_ruecklauf"])
        plantsim.set_value("ActionControl[\"id\",1]", i)
        plantsim.set_value("ActionControl[\"action\",1]", action)
        value_py = plantsim.get_value("sync[\"python\",1]")
        value_ps = plantsim.get_value("sync[\"plantsim\",1]")
        while value_py >= value_ps:
            sleep(0.01)
        plantsim.set_value("sync[\"python\",1]", i)
        sleep(0.5)


# doubleclick object in PlantSim and lookup the path_context
# socket is the name of the socket object in PlantSim or None if not used
model = r'D:\Tjark\Dokumente\FH Bielefeld\Sommersemester 2022\Diskrete Simulation und Reinforceent Learning\Pruefung\pruefung_git\DSRL_Pruefung.spp'
plantsim = Plantsim(version='16.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=True)
id = 1
action = "move_to_ruecklauf"
iteration = 1



if not plantsim.plantsim.IsSimulationRunning():
    plantsim.start_simulation()

sleep(10)
#plantsim.quit()

test_run(100, plantsim)