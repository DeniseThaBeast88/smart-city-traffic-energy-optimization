# Smart-City Unity 3D Playback (Assignment 1 deliverable)

This Unity project is the 3D simulation environment for the AI-Driven Smart-City Traffic and Energy Optimization System. It replays evaluation day 1 of the Python/SimPy simulation in 3D:

- the 3 x 3 city grid with roads, nine signalized intersections, and four energy zones
- every vehicle trip (about 49,700 per day): cars enter at the city edge, wait in stop-line queues, cross links, and leave
- the exact signal changes chosen by each controller: fixed-time, rule-based actuated, and Q-learning (RL)
- the energy grid every 15 minutes: building demand, rooftop solar output and curtailment, battery charge, grid import, and street-light dimming, for the baseline and AI-optimized energy policies
- a live dashboard with traffic and energy metrics, a controller comparison, a traffic-density versus energy-use chart, and ground heatmaps

The AI (Q-learning signal agent and the gradient-boosting load forecaster) runs in Python. Unity is the visualization and simulation-environment layer, so every number on screen matches the report.

## What you need

- Unity Hub and Unity 6 (6000.x) or Unity 2022.3 LTS. The free Personal license is fine.
- Any 3D template works: "Universal 3D" (URP) or "3D (Built-In Render Pipeline)". No extra packages are required.
- About 25 MB of disk space for the simulation data.

## Setup, step by step

1. Open Unity Hub, select **New project**, pick the **Universal 3D** (or **3D Built-In**) template, name it `SmartCityPlayback`, and select **Create project**.
2. When the editor opens, choose one of these:
   - **Option A (package):** select **Assets > Import Package > Custom Package...**, choose `SmartCityPlayback.unitypackage`, and select **Import**.
   - **Option B (folders):** in File Explorer (or Finder), copy the `Assets/SmartCity` and `Assets/StreamingAssets` folders from this folder into your new project's `Assets` folder.
3. Wait for Unity to finish compiling (the spinner in the lower-right corner stops). The Console should show no red errors.
4. In the menu bar, select **Smart City > Create Playback Scene**. This saves `Assets/Scenes/SmartCity.unity`.
5. Press **Play**. The city builds itself in about two seconds and the day starts at 07:00.

Tip: if you skip step 4 and press Play in any scene, the playback starts automatically anyway.

## Controls

| Control | Action |
|---|---|
| Left-drag in the 3D view | Orbit the camera |
| Right-drag | Pan |
| Mouse wheel | Zoom |
| Play/Pause, Restart, speed buttons (1x to 900x), time slider | Control the simulated clock |
| Traffic-signal controller buttons (or keys 1, 2, 3) | Switch between Fixed-time, Rule-based, and Q-learning |
| Energy policy buttons (or keys B = Baseline, O = AI-optimized, E = toggle) | Switch the energy policy; the banner at the top center shows the active policy, and the Console logs each switch |
| Energy policy comparison panel (right side) | Shows Baseline and AI-optimized side by side: right now (grid import, curtailed solar, lighting, battery) and since midnight (grid energy, cost, wasted energy, lighting energy) with percent change |
| Ground view buttons | Zone colors, grid-import heatmap, or traffic-queue heatmap |
| Camera buttons | Overview, Top-down, Center I4, Arterial street view, Energy bars |
| Screenshot button (or P) | Saves a PNG in the project's `Screenshots` folder |
| Hide dashboard (or H), Space | Hide the dashboard, pause or resume |

## What you see

- **Cars:** white when moving; orange turning red the longer they wait at a signal.
- **Signals:** a light on every approach shows green, yellow, or red.
- **Buildings:** turn warmer as the zone's electricity demand rises. **Rooftop panels:** blue with solar output, pink when solar energy is curtailed (wasted). Compare Baseline and AI-optimized at noon to see the curtailment disappear.
- **Street lights:** brightness follows the zone's lighting power. With the AI-optimized policy, they dim when traffic is light.
- **Bars beside each zone:** green shows battery charge and red shows grid import.
- **Dashboard:** live vehicles, queues, completed trips, and average delay, plus city demand, solar, grid import, lighting, curtailment, battery charge, and tariff. The right panel compares the average delay of all three controllers so far today and charts traffic density against grid import over 24 hours.

## Regenerating the data from Python

The CSV files in `Assets/StreamingAssets/SmartCityData` come from the Python project. To regenerate them:

```
cd smart_city_project
python export_unity_data.py --unity-dir ../SmartCityUnity
```

The script prints each controller's average delay next to the report value for day 1 (42.49 s, 22.96 s, and 19.24 s) to confirm the replay is identical. If you imported the files into a different Unity project, point `--unity-dir` at that project folder instead.

## Scripts

| Script | Purpose |
|---|---|
| `SmartCityData.cs` | Loads the CSV traces (vehicle hops, signal events, trips, energy dispatch, summary) |
| `CityBuilder.cs` | Builds roads, zone tiles, buildings, solar panels, signals, street lights, and indicator bars |
| `TrafficPlayback.cs` | Places every vehicle each frame (entering, queued, moving, exiting) and colors the signals |
| `EnergyPlayback.cs` | Animates demand, solar, curtailment, batteries, grid import, lighting, and heatmaps |
| `SmartCityApp.cs` | Simulation clock, camera, dashboard, chart, and screenshots |
| `Editor/SmartCitySceneMenu.cs` | Adds the **Smart City** menu that creates and saves the scene |

## Troubleshooting

- **"Smart-city data could not be loaded":** the `StreamingAssets/SmartCityData` folder is missing. Copy it into `Assets/StreamingAssets` or run the export script.
- **Pink objects:** the render pipeline changed after the objects were created. Stop Play mode and press Play again.
- **Low frame rate at 900x:** use 60x or 300x. The scene draws a few hundred cars at most, so any recent laptop runs it smoothly.

## Where the AI-optimized energy policy shows a difference

The two policies use the same building demand, so building colors look the same. The differences are in the grid, batteries, solar panels, and street lights. They are easiest to see at these times:

| Time | What changes when you press B, then O |
|---|---|
| 12:00 (Grid heatmap, Energy bars camera) | Baseline curtails solar (pink rooftops, about 120 kW wasted); AI-optimized shares and stores it (blue rooftops, 0 kW wasted, grid import about 29 percent lower) |
| 18:00 | AI-optimized discharges the batteries it charged at midday, so the red grid bars are lower |
| 22:00 or 03:00 (Zones view) | Baseline lamps are fully on; AI-optimized lamps are dimmed by traffic (lighting about 55 to 65 percent lower) |
| 08:00 | Only a small difference (about 8 percent less grid import), so the comparison panel is the best evidence at this hour |
