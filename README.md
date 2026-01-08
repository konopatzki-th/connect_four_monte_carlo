\# Monte Carlo Simulation – Starting Player Advantage in Connect Four



\## Overview

This project analyzes whether the starting player in the game \*Connect Four\* has a statistical advantage.

A Monte Carlo simulation is used to simulate a large number of games between players of different

skill levels and to evaluate win probabilities.



The main focus lies on quantifying the advantage of the first move and understanding how it depends

on the relative strength of the players.



\## Research Question

Does the starting player in Connect Four have a measurable advantage, and how does this advantage

change for different player skill levels?



\## Methodology

\- Monte Carlo simulation of repeated Connect Four games

\- Board size: 6 × 7 (standard Connect Four rules)

\- Player strategies representing increasing skill levels:

&nbsp; - Random player

&nbsp; - Heuristic player

&nbsp; - Minimax-based intelligent players with limited search depth

\- Statistical evaluation of win probabilities and starting player advantage

\- Visualization of aggregated results and Monte Carlo distributions



\## Project Structure



├── src/

│   └── simulation.py        # Main simulation and analysis code

├── figures/

│   ├── starting\_player\_advantage\_overview.png

│   └── \*\_advantage\_distribution.png

├── results/                 

├── README.md

├── requirements.txt

├── .gitignore



\## Requirements

Python 3.9



NumPy 1.23.5



Matplotlib 3.6.2





\## Reproducibility

To ensure reproducibility of the simulation results, fixed package versions are used and a fixed

random seed is set within the simulation code. This allows the figures and numerical results to be

reproduced consistently across different systems.



\## How to Run the Simulation

After installing the required dependencies, run:



python src/simulation.py



All generated figures will be saved automatically in the figures/ directory.



\## Results Summary

The simulation results show that the starting player generally has a higher probability of winning.

The magnitude of this advantage depends on the difference in player skill levels.

For players of similar strength, the advantage is reduced but does not disappear entirely.



The Monte Carlo distributions indicate that the observed advantage is stable and not caused by

random fluctuations.



\## Authors

Therese Konopatzki

Leonardo Maurer

