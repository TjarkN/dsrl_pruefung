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
        active = plantsim.get_value("sync[\"isPythonActive\",1]")
        while active == "false":
            sleep(0.01)
            active = plantsim.get_value("sync[\"isPythonActive\",1]")

        gercount = 0
        language = plantsim.get_value("CurrentState[\"entry_language\",1]")
        if language=="GameGer":
            action = "move_to_lager1"
            gercount += 1
        else:
            action = np.random.choice(["move_to_lager2","move_to_ruecklauf"]) #"move_to_lager1",
            if gercount == 9:
                action = "move_to_lager1"
                gercount = 0

        plantsim.set_value("ActionControl[\"id\",1]", i)
        plantsim.set_value("ActionControl[\"action\",1]", action)
        print(action)

        plantsim.set_value("sync[\"isPythonActive\",1]", False)
        plantsim.execute_simtalk('restartAgent')
        print(i)


# doubleclick object in PlantSim and lookup the path_context
# socket is the name of the socket object in PlantSim or None if not used
model = r'D:\Tjark\Dokumente\FH Bielefeld\Sommersemester 2022\Diskrete Simulation und Reinforceent Learning\Pruefung\pruefung_git\DSRL_Pruefung.spp'
#model = r'C:\Users\dlina\DSRL\DSRL_Pruefung.spp'
plantsim = Plantsim(version='16.1', license_type='Educational', path_context='.Modelle.Modell', model=model,
                    socket=None, visible=True)

if not plantsim.plantsim.IsSimulationRunning():
    plantsim.start_simulation()



#sleep(10)
#plantsim.quit()

test_run(500, plantsim)

