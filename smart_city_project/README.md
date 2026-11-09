# AI-Driven Smart-City Traffic and Energy Optimization System

COS570 Advanced Artificial Intelligence, Assignment 1 (Margaret Gonzalez, Full Sail University)

## What the project does
- Builds a 3 x 3 city grid (intersections I0 to I8) with four energy zones (Residential, Commercial, Industrial, Mixed-Use) in SimPy.
- Compares three traffic-signal controllers: fixed-time (30 s greens), rule-based actuated, and a shared multi-agent Q-learning agent.
- Runs an energy grid manager every 15 minutes: a baseline policy and an artificial-intelligence-optimized policy that uses a gradient-boosting load forecast, peak shaving, solar sharing between zones, and traffic-adaptive street lighting.
- Plays the simulation back in a Unity 3D city (folder ../SmartCityUnity, see UNITY_README.md there).
- Produces figures, heatmaps, an animation (MP4 and GIF), CSV metrics, and automated tests.

## Files
| File | Purpose |
|---|---|
| smart_city_sim.py | City model, controllers, Q-learning, energy manager, forecaster |
| run_experiments.py | Trains the agent, runs 5 evaluation days and the +20 percent stress test, writes all outputs |
| tests/test_smart_city.py | 14 validation tests (vehicle conservation, no conflicting greens, energy balance, Unity trace consistency, and more) |
| export_unity_data.py | Exports vehicle traces, signal phases, and energy dispatch to the Unity project (../SmartCityUnity) |
| make_evidence_images.py | Renders the test and run logs as images |
| Smart_City_Simulation.ipynb | Notebook that runs everything step by step with outputs shown |
| outputs/ | Figures, animation, data tables, logs |

## How to run (step by step)
1. Install Python 3.10 or newer.
2. Open a terminal in this folder and install packages: `pip install -r requirements.txt`
3. Quick check (about 2 minutes): `python run_experiments.py --quick` (writes to outputs_quick/ so the report results in outputs/ are not changed)
4. Full experiment (about 4 minutes): `python run_experiments.py`
5. Run the tests: `python -m pytest -v tests`
6. Or open the notebook: `jupyter notebook Smart_City_Simulation.ipynb` and choose Run All.
7. Re-draw figures from saved data without re-running: `python run_experiments.py --figures-only`
8. Export data for Unity (about 15 s): `python export_unity_data.py --unity-dir ../SmartCityUnity`

## Unity 3D deliverable (step by step)
1. Install Unity Hub and Unity 2021.3 LTS or newer (2022.3 LTS or Unity 6 also work).
2. In Unity Hub choose New project, template 3D (Built-In Render Pipeline) or Universal 3D, and create it.
3. In Unity choose Assets > Import Package > Custom Package and select SmartCityPlayback.unitypackage, then Import.
4. Choose the menu Smart City > Create Playback Scene, then press Play.
5. Use the on-screen dashboard to switch controllers (Fixed, Rule, Q-learning), energy policy, speed, and heatmap view. Press P or the Screenshot button to save images to the Screenshots folder.

Results are deterministic (fixed random seeds), so a full run reproduces the numbers in the report.

## Key results (mean of 5 simulated days)
- Average delay: fixed-time 42.63 s, rule-based 23.03 s, Q-learning 19.18 s (55.0 percent and 16.7 percent lower).
- Energy: grid import down 5.7 percent, peak demand down 8.9 percent, cost down 12.7 percent, wastage down 98.6 percent, street-lighting energy down 59.1 percent.
- Forecaster: MAE 11.0 kW, MAPE 3.8 percent, R squared 0.996.
