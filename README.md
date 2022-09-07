# DSRL Pruefung Sommersemester 2022
## Tjark Nitsche & Lina Doering
### Prüfungsabgabe in Diskrete Simulation und Reinforcement Learning

Abgaberelevante Dateien:
- ps_training.py
- ps_environment.py
- ps_test_run.py
- DSRL_Pruefung_100.spp (Modell mit 120 Input + goal state = 100)
- DSRL_Pruefung_100.spp.bak
- Abgabe-q_table-100_nitsche_doering.npy
- DSRL_Pruefung_1000.spp (Modell mit Input 1100 + goal state = 1000)
- DSRL_Pruefung_1000.spp.bak
- Abgabe-q_table-1000_nitsche_doering.npy
- Dokumentation Trainingsläufe.pdf

Dateierläuterung:

- agents: agent, q_learning_agent_mas, (deep_q_learning)
- qtables: q_tables aus Trainingsruns (siehe Dokumentation der Trainingsläufe pdf)
- plantsim: Schnittstelle (uebernommen)
- problem (uebernommen)
- ps_environment - auf unsere Simulation angepasst
- ps_training - zur Durchführung der Trainings
- ps_test_run - zur Durchführung der Tests
- DSRL_Pruefung - PlantSim Model

### Erläuterungen PlantSim-Model
- Funktionalitäten "binden", "färben" + "zu einem Spiel fassen" im vorderen Bereich des Modells
- bei Ankunft am PickAndPlaceRobot = unserem Agent, wird die CurrentState-Tabelle beschrieben + die Simulation gestoppt
- In Python wird die CurrentState-Tabelle eingelesen und eine Entscheidung über eine Action getroffen 
- Durch den Aufruf der Methode "restartAgent" des Modells aus Python heraus wird die Simulation wieder gestartet usw.
- Model kann 1000 Games durchlaufen lassen (siehe pauseEntry and entryExit Methoden in PlantSim-Model)
  - jedoch konnten wir bisher lediglich Trainingsläufe durchführen
  - es gab keinen erfolgreichen Testrun (weder mit goal state = 100 games noch 1000 games) 
  - Vermutung: nicht ausreichend Infos im state / current state, d.h. ohne ausprobieren, was er im Testlauf nicht mehr macht, schafft er es nicht

### Aufgetretene Probleme

Lagerspezifisches Lernen

- Der Q-Learning Agent hat immer wieder gelernt, eine bestimmte Sprache in Lager 1, eine andere in Lager 2 und die dritte in den Rücklauf zu schicken
- Viel Zeit auf Anpassungen in der Reward-Funktion dahingehend investiert

Deep-Q-Learning
- Einige Versuche gestartet, Training läuft auch an, verrennt sich dann aber (vermutlich anpassungen des DeepQTable NNs + Experience Replay notwendig)

Fehlende PlantSim Community
- wenig Anwendungs-Beispiele für Programmierungen in PlantSim (SimTalk)
- keine umfangreiche Austauschplattfrom für Bugs und Anwender:innen-Fehler
- dadurch Schwierigkeiten dabei Fehler zu identifizieren, um sie zu beheben / mühsames "Debuggen"

### Ideen zur Optimierung

Zusätzliche Infos im current state übergeben → dadurch würde das Problem komplexer (mehr states)
- z.B. ein Boolean, ob genug Teile von allen Sprachen drin sind, dass jeweils 10 wegsortiert werden können (Lösungsansatz für Problem Lagerspezifisches Lernen)
- oder die Lagerfüllstände der beiden Lager, was jedoch zu sehr viel mehr states führen würde (positiver)

### GitHub Repro
https://github.com/TjarkN/dsrl_pruefung 