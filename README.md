# DSRL Pruefung Sommersemester 2022
## Tjark Nitsche & Lina Doering
### Prüfungsabgabe in Diskrete Simulation und Reinforcement Learning

Dateien:

- agents: agent, q-learning agent, (deep-q-learning)
- qtables: q_tables aus Trainingsruns (siehe Dokumentation der Trainingsläufe)
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
- Model kann auch 1000 Teile durchlaufen lassen, jedoch konnten wir bisher lediglich Trainingsläufe mit 100 Teilen / Games durchführen (siehe pauseEntry and entryExit Methoden in PlantSim-Model)

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