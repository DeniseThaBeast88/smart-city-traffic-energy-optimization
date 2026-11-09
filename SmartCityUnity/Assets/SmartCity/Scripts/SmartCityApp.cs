// SmartCityApp.cs
// Entry point of the Unity smart-city playback. It loads the SimPy traces,
// builds the 3D city, runs the simulation clock, and draws the dashboard
// (IMGUI, so no UI packages or input settings are required).
//
// Press Play in any scene: if no SmartCityApp exists, one is created
// automatically. Use "Smart City > Create Playback Scene" to save a scene.
using System;
using System.IO;
using UnityEngine;

namespace SmartCity
{
    public class SmartCityApp : MonoBehaviour
    {
        public static SmartCityApp Instance;

        [Header("Playback")]
        [Tooltip("Simulated start time in hours (0-24).")]
        public float startHour = 7.0f;
        [Tooltip("Simulated seconds per real second.")]
        public float speed = 10f;
        public int controllerIndex = 2;    // 0 fixed-time, 1 rule-based, 2 Q-learning
        public int policyIndex = 0;        // 0 baseline, 1 AI-optimized (switch with B / O)
        public GroundView groundView = GroundView.ZoneColors;

        SmartCityData data;
        CityBuilder city;
        TrafficPlayback traffic;
        EnergyPlayback energy;
        string loadError;

        float simTime;            // seconds since midnight
        bool playing = true;
        bool showHud = true;
        string lastShot = "";
        float lastShotTime = -10f;

        // camera orbit
        Camera cam;
        Light sun;
        Vector3 camTarget = Vector3.zero;
        float yaw = 30f, pitch = 55f, dist = 380f;

        // dashboard chart
        Texture2D chart;
        int chartController = -1, chartPolicy = -1;
        float policySwitchTime = -10f;
        Rect[] panelRects = new Rect[0];
        const int ChartW = 288, ChartH = 110;
        Texture2D white;

        static readonly float[] Speeds = { 1f, 10f, 60f, 300f, 900f };
        static readonly string[] SpeedLabels = { "1x", "10x", "60x", "300x", "900x" };
        static readonly string[] GroundLabels = { "Zones", "Grid heatmap", "Queue heatmap" };

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        static void Bootstrap()
        {
            if (Instance == null) new GameObject("SmartCityApp").AddComponent<SmartCityApp>();
        }

        void Awake()
        {
            if (Instance != null && Instance != this) { Destroy(gameObject); return; }
            Instance = this;
        }

        void Start()
        {
            Application.targetFrameRate = 60;
            white = Texture2D.whiteTexture;
            SetupCameraAndLight();
            try
            {
                float t0 = Time.realtimeSinceStartup;
                data = SmartCityData.Load();
                city = new CityBuilder();
                city.Build();
                traffic = new TrafficPlayback(city);
                energy = new EnergyPlayback(city, data);
                simTime = Mathf.Clamp(startHour, 0f, 24f) * 3600f;
                Debug.Log(string.Format("Smart city data loaded in {0:F1} s from {1}",
                    Time.realtimeSinceStartup - t0, data.Folder));
            }
            catch (Exception ex)
            {
                loadError = ex.Message;
                Debug.LogError("Smart city playback could not start: " + ex);
            }
        }

        void SetupCameraAndLight()
        {
            cam = Camera.main;
            if (cam == null)
            {
                var go = new GameObject("Main Camera");
                go.tag = "MainCamera";
                cam = go.AddComponent<Camera>();
            }
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.farClipPlane = 3000f;
            cam.fieldOfView = 50f;

            sun = RenderSettings.sun;
            if (sun == null)
            {
                var go = new GameObject("Sun");
                sun = go.AddComponent<Light>();
                sun.type = LightType.Directional;
                sun.shadows = LightShadows.Soft;
                RenderSettings.sun = sun;
            }
        }

        void Update()
        {
            if (data == null) return;
            if (playing)
            {
                simTime += Time.deltaTime * speed;
                if (simTime >= 86399f) { simTime = 86399f; playing = false; }
            }
            TrafficTrace tr = data.Traces[controllerIndex];
            traffic.UpdateSignals(tr, simTime);
            traffic.UpdateVehicles(tr, simTime);
            energy.UpdateEnergy(policyIndex, simTime, groundView, traffic.QueueByIntersection);
            UpdateDaylight(simTime / 3600f);
        }

        void LateUpdate()
        {
            if (cam == null) return;
            pitch = Mathf.Clamp(pitch, 8f, 89f);
            dist = Mathf.Clamp(dist, 25f, 900f);
            Quaternion rot = Quaternion.Euler(pitch, yaw, 0f);
            cam.transform.position = camTarget + rot * new Vector3(0, 0, -dist);
            cam.transform.LookAt(camTarget);
        }

        void UpdateDaylight(float hour)
        {
            float elev = Mathf.Sin((hour - 6f) / 12f * Mathf.PI);           // > 0 between 06:00 and 18:00
            float day = Mathf.Clamp01(elev * 1.6f + 0.15f);
            sun.transform.rotation = Quaternion.Euler(Mathf.Lerp(-10f, 170f, (hour - 6f) / 12f), -40f, 0f);
            sun.intensity = Mathf.Lerp(0.12f, 1.15f, day);
            sun.color = Color.Lerp(new Color(0.55f, 0.6f, 0.9f), new Color(1f, 0.96f, 0.88f), day);
            cam.backgroundColor = Color.Lerp(new Color(0.03f, 0.04f, 0.10f), new Color(0.55f, 0.74f, 0.93f), day);
            RenderSettings.ambientLight = Color.Lerp(new Color(0.12f, 0.13f, 0.2f), new Color(0.55f, 0.55f, 0.6f), day);
        }

        // ------------------------------------------------------------------ dashboard
        float UiScale { get { return Mathf.Clamp(Screen.height / 1000f, 0.5f, 4f); } }

        void OnGUI()
        {
            float s = UiScale;
            GUI.matrix = Matrix4x4.Scale(new Vector3(s, s, 1f));
            float sw = Screen.width / s, sh = Screen.height / s;

            if (loadError != null)
            {
                GUI.Box(new Rect(20, 20, sw - 40, 120), "");
                GUI.Label(new Rect(35, 30, sw - 70, 100),
                    "<b>Smart-city data could not be loaded.</b>\n" + loadError +
                    "\nRun 'python export_unity_data.py --unity-dir <this project>' and press Play again.",
                    RichLabel(14));
                return;
            }
            if (data == null) return;

            Rect left = new Rect(10, 10, 345, 560);
            Rect right = new Rect(sw - 320, 10, 310, 470);
            Rect compare = new Rect(sw - 320, 490, 310, 250);
            Rect camPanel = new Rect(sw - 320, sh - 175, 310, 165);
            HandleCameraInput(new[] { left, right, compare, camPanel }, s);
            if (!showHud)
            {
                if (GUI.Button(new Rect(10, 10, 110, 26), "Show dashboard")) showHud = true;
                return;
            }

            panelRects = new[] { left, right, compare, camPanel };
            DrawWorldLabels(s);
            DrawLeftPanel(left);
            DrawRightPanel(right);
            DrawPolicyComparison(compare);
            DrawCameraPanel(camPanel);
            DrawPolicyBanner(sw);
            if (Time.unscaledTime - lastShotTime < 4f)
                GUI.Label(new Rect(370, sh - 40, sw - 700, 30), "Screenshot saved: " + lastShot, RichLabel(13));
        }

        void DrawLeftPanel(Rect r)
        {
            Panel(r);
            GUILayout.BeginArea(new Rect(r.x + 10, r.y + 8, r.width - 20, r.height - 16));
            GUILayout.Label("<b>Smart-City Traffic and Energy Playback</b>", RichLabel(15));
            GUILayout.Label("SimPy evaluation day 1 replayed in Unity", RichLabel(11));

            GUILayout.BeginHorizontal();
            GUILayout.Label("<size=22><b>" + Clock(simTime) + "</b></size>", RichLabel(14), GUILayout.Width(120));
            if (GUILayout.Button(playing ? "Pause" : "Play", GUILayout.Height(28))) playing = !playing;
            if (GUILayout.Button("Restart", GUILayout.Height(28))) { simTime = 0f; playing = true; }
            GUILayout.EndHorizontal();

            int si = Array.IndexOf(Speeds, speed);
            int nsi = GUILayout.Toolbar(si < 0 ? 1 : si, SpeedLabels);
            speed = Speeds[nsi];
            float hour = GUILayout.HorizontalSlider(simTime / 3600f, 0f, 23.99f);
            if (Mathf.Abs(hour * 3600f - simTime) > 1f) simTime = hour * 3600f;
            GUILayout.Space(8);

            GUILayout.Label("<b>Traffic-signal controller</b>", RichLabel(12));
            controllerIndex = GUILayout.Toolbar(controllerIndex, new[] { "Fixed-time", "Rule-based", "Q-learning" });
            GUILayout.Label("<b>Energy policy</b>", RichLabel(12));
            SetPolicy(GUILayout.Toolbar(policyIndex, SmartCityData.PolicyNames));
            GUILayout.Label("<b>Ground view</b>", RichLabel(12));
            groundView = (GroundView)GUILayout.Toolbar((int)groundView, GroundLabels);
            GUILayout.Space(8);

            TrafficTrace tr = data.Traces[controllerIndex];
            int done; double avg;
            tr.CompletedAt(simTime, out done, out avg);
            GUILayout.Label("<b>Live traffic (" + tr.DisplayName + ")</b>", RichLabel(12));
            GUILayout.Label(string.Format(
                "Vehicles in network: {0}\nQueued at signals: {1}\nWaiting longer than 60 s: {2}\n" +
                "Trips completed today: {3:N0}\nAverage delay so far: {4:F1} s",
                traffic.ActiveVehicles, traffic.QueuedVehicles, traffic.LongWaits, done, avg), RichLabel(12));
            GUILayout.Space(6);

            GUILayout.Label("<b>Live energy (" + SmartCityData.PolicyNames[policyIndex] + ")</b>", RichLabel(12));
            GUILayout.Label(string.Format(
                "City demand: {0:N0} kW\nSolar output: {1:N0} kW\nGrid import: {2:N0} kW\n" +
                "Street lighting: {3:N1} kW\nSolar curtailed (wasted): {4:N0} kW\n" +
                "Average battery charge: {5:P0}\nTariff: {6:F2} USD/kWh{7}",
                energy.CityLoad, energy.CityPv, energy.CityGrid, energy.CityLighting, energy.CityCurtail,
                energy.CityBatteryFrac, energy.Price, energy.Price >= 0.2f ? "  (on-peak)" : ""), RichLabel(12));
            GUILayout.EndArea();
        }

        void SetPolicy(int p)
        {
            p = Mathf.Clamp(p, 0, 1);
            if (p == policyIndex) return;
            policyIndex = p;
            policySwitchTime = Time.unscaledTime;
            float g, c, l, b;
            data.CityNowAt(policyIndex, simTime, out g, out c, out l, out b);
            Debug.Log(string.Format("Energy policy switched to {0} at {1}: grid import {2:N0} kW, curtailed {3:N0} kW, lighting {4:N1} kW",
                SmartCityData.PolicyNames[policyIndex], Clock(simTime), g, c, l));
        }

        void DrawPolicyBanner(float sw)
        {
            bool ai = policyIndex == 1;
            float w = 360f, x = (sw - w) / 2f;
            Rect r = new Rect(x, 10, w, 34);
            float flash = Mathf.Clamp01(1.5f - (Time.unscaledTime - policySwitchTime));
            Color baseCol = ai ? new Color(0.1f, 0.55f, 0.3f, 0.9f) : new Color(0.55f, 0.3f, 0.1f, 0.9f);
            GUI.color = Color.Lerp(baseCol, Color.white, flash * 0.6f);
            GUI.DrawTexture(r, white);
            GUI.color = Color.white;
            GUIStyle st = RichLabel(16);
            st.alignment = TextAnchor.MiddleCenter;
            GUI.Label(r, "<b>Energy policy: " + SmartCityData.PolicyNames[policyIndex] + "</b>   (B / O keys)", st);
        }

        void DrawPolicyComparison(Rect r)
        {
            Panel(r);
            GUILayout.BeginArea(new Rect(r.x + 10, r.y + 8, r.width - 20, r.height - 16));
            GUILayout.Label("<b>Energy policy comparison</b>", RichLabel(12));
            float g0, c0, l0, b0, g1, c1, l1, b1;
            data.CityNowAt(0, simTime, out g0, out c0, out l0, out b0);
            data.CityNowAt(1, simTime, out g1, out c1, out l1, out b1);
            float tg0, tc0, tw0, tl0, tg1, tc1, tw1, tl1;
            data.TotalsAt(0, simTime, out tg0, out tc0, out tw0, out tl0);
            data.TotalsAt(1, simTime, out tg1, out tc1, out tw1, out tl1);
            string bh = policyIndex == 0 ? "<b><color=#FFC870>Baseline</color></b>" : "Baseline";
            string ah = policyIndex == 1 ? "<b><color=#7CFFB0>AI-optimized</color></b>" : "AI-optimized";
            GUIStyle mono = RichLabel(11);
            GUILayout.BeginHorizontal();
            GUILayout.Label("", mono, GUILayout.Width(130));
            GUILayout.Label(bh, mono, GUILayout.Width(58));
            GUILayout.Label(ah, mono, GUILayout.Width(80));
            GUILayout.EndHorizontal();
            GUILayout.Label("<b>Right now (" + Clock(simTime) + ")</b>", mono);
            Row(mono, "Grid import (kW)", g0, g1, "N0");
            Row(mono, "Solar curtailed (kW)", c0, c1, "N0");
            Row(mono, "Street lighting (kW)", l0, l1, "N1");
            Row(mono, "Battery charge (%)", b0 * 100f, b1 * 100f, "N0");
            GUILayout.Label("<b>Since midnight</b>", mono);
            Row(mono, "Grid energy (kWh)", tg0, tg1, "N0");
            Row(mono, "Energy cost (USD)", tc0, tc1, "N0");
            Row(mono, "Wasted energy (kWh)", tw0, tw1, "N0");
            Row(mono, "Lighting energy (kWh)", tl0, tl1, "N0");
            GUILayout.EndArea();
        }

        static void Row(GUIStyle st, string label, float a, float b, string fmt)
        {
            string change = "";
            if (Mathf.Abs(a) > 0.5f) change = string.Format("{0:+0;-0;0}%", (b - a) / a * 100f);
            GUILayout.BeginHorizontal();
            GUILayout.Label(label, st, GUILayout.Width(130));
            GUILayout.Label(a.ToString(fmt), st, GUILayout.Width(58));
            GUILayout.Label(b.ToString(fmt), st, GUILayout.Width(50));
            GUILayout.Label(change, st);
            GUILayout.EndHorizontal();
        }

        void DrawRightPanel(Rect r)
        {
            Panel(r);
            GUILayout.BeginArea(new Rect(r.x + 10, r.y + 8, r.width - 20, r.height - 16));
            GUILayout.Label("<b>Controller comparison (today so far)</b>", RichLabel(12));
            for (int i = 0; i < 3; i++)
            {
                int done; double avg;
                data.Traces[i].CompletedAt(simTime, out done, out avg);
                string name = data.Traces[i].DisplayName;
                string line = string.Format("{0,-22} {1,6:F1} s", name, avg);
                GUILayout.Label(i == controllerIndex ? "<b>" + line + "</b>" : line, RichLabel(12));
            }
            GUILayout.Space(4);
            GUILayout.Label("<b>Five-day results (report Tables II and III)</b>", RichLabel(12));
            GUILayout.Label(string.Format(
                "Avg delay: fixed {0:F2} s | rule {1:F2} s | RL {2:F2} s\n" +
                "Energy cost: {3:N0} to {4:N0} USD/day\nWastage: {5:N0} to {6:N0} kWh/day",
                data.SummaryValue("Fixed-time", "avg_delay_s"),
                data.SummaryValue("Rule-based actuated", "avg_delay_s"),
                data.SummaryValue("Q-learning (RL)", "avg_delay_s"),
                data.SummaryValue("Baseline", "energy_cost_usd"), data.SummaryValue("AI-optimized", "energy_cost_usd"),
                data.SummaryValue("Baseline", "energy_waste_kwh"), data.SummaryValue("AI-optimized", "energy_waste_kwh")),
                RichLabel(11));
            GUILayout.Space(4);
            GUILayout.Label("<b>Traffic density vs. energy use (24 h)</b>", RichLabel(12));
            Rect cr = GUILayoutUtility.GetRect(ChartW, ChartH);
            DrawChart(cr);
            GUILayout.Label("<color=#4FA3FF>Blue</color>: vehicles in network   <color=#FF5A4A>Red</color>: grid import",
                RichLabel(11));
            GUILayout.Space(4);
            GUILayout.Label("<b>Legend</b>", RichLabel(12));
            GUILayout.Label("Cars: white moving, orange to red = waiting longer\n" +
                            "Signals: green, yellow, red per approach\n" +
                            "Rooftops: blue = solar output, pink = solar curtailed\n" +
                            "Buildings warm up with electricity demand\n" +
                            "Bars: green battery charge, red grid import", RichLabel(11));
            GUILayout.EndArea();
        }

        void DrawCameraPanel(Rect r)
        {
            Panel(r);
            GUILayout.BeginArea(new Rect(r.x + 10, r.y + 8, r.width - 20, r.height - 16));
            GUILayout.Label("<b>Camera</b>  (drag to orbit, scroll to zoom)", RichLabel(12));
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Overview")) { camTarget = Vector3.zero; yaw = 30; pitch = 55; dist = 380; }
            if (GUILayout.Button("Top-down")) { camTarget = Vector3.zero; yaw = 0; pitch = 89; dist = 470; }
            if (GUILayout.Button("Center I4")) { camTarget = CityLayout.IntersectionPos(4); yaw = 45; pitch = 35; dist = 70; }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Arterial")) { camTarget = CityLayout.IntersectionPos(7); yaw = 0; pitch = 18; dist = 90; }
            if (GUILayout.Button("Energy bars")) { camTarget = new Vector3(60, 10, 60); yaw = 225; pitch = 25; dist = 260; }
            GUILayout.EndHorizontal();
            GUILayout.BeginHorizontal();
            if (GUILayout.Button("Screenshot", GUILayout.Height(28))) TakeScreenshot();
            if (GUILayout.Button("Hide dashboard", GUILayout.Height(28))) showHud = false;
            GUILayout.EndHorizontal();
            GUILayout.EndArea();
        }

        void HandleCameraInput(Rect[] panels, float s)
        {
            Event e = Event.current;
            if (e == null) return;
            bool overPanel = false;
            if (showHud)
                foreach (Rect p in panels) if (p.Contains(e.mousePosition)) overPanel = true;
            if (e.type == EventType.MouseDrag && !overPanel)
            {
                if (e.button == 0) { yaw += e.delta.x * 0.3f; pitch += e.delta.y * 0.3f; }
                else
                {
                    Vector3 right = cam.transform.right; right.y = 0;
                    Vector3 fwd = Vector3.Cross(right, Vector3.up);
                    camTarget += (-right.normalized * e.delta.x + fwd.normalized * e.delta.y) * dist * 0.002f;
                }
                e.Use();
            }
            else if (e.type == EventType.ScrollWheel && !overPanel)
            {
                dist *= 1f + e.delta.y * 0.05f;
                e.Use();
            }
            else if (e.type == EventType.KeyDown)
            {
                if (e.keyCode == KeyCode.Space) { playing = !playing; e.Use(); }
                else if (e.keyCode == KeyCode.H) { showHud = !showHud; e.Use(); }
                else if (e.keyCode == KeyCode.P) { TakeScreenshot(); e.Use(); }
                else if (e.keyCode == KeyCode.Alpha1) controllerIndex = 0;
                else if (e.keyCode == KeyCode.Alpha2) controllerIndex = 1;
                else if (e.keyCode == KeyCode.Alpha3) controllerIndex = 2;
                else if (e.keyCode == KeyCode.B) { SetPolicy(0); e.Use(); }
                else if (e.keyCode == KeyCode.O) { SetPolicy(1); e.Use(); }
                else if (e.keyCode == KeyCode.E) { SetPolicy(1 - policyIndex); e.Use(); }
            }
        }

        void Panel(Rect r)
        {
            GUI.color = new Color(0.07f, 0.08f, 0.11f, 0.88f);   // nearly opaque so the 3D labels do not show through
            GUI.DrawTexture(r, white);
            GUI.color = Color.white;
            GUI.Box(r, "");
        }

        bool UnderPanel(Rect lr)
        {
            foreach (Rect p in panelRects) if (p.Overlaps(lr)) return true;
            return false;
        }

        void DrawWorldLabels(float s)
        {
            GUIStyle st = RichLabel(12);
            st.alignment = TextAnchor.MiddleCenter;
            for (int iid = 0; iid < 9; iid++)
            {
                Vector3 sp = cam.WorldToScreenPoint(CityLayout.IntersectionPos(iid) + Vector3.up * 14f);
                if (sp.z <= 0) continue;
                Rect lr = new Rect(sp.x / s - 40, (Screen.height - sp.y) / s - 12, 80, 24);
                if (UnderPanel(lr)) continue;
                GUI.color = new Color(0, 0, 0, 0.55f);
                GUI.DrawTexture(lr, white);
                GUI.color = Color.white;
                GUI.Label(lr, "<b>I" + iid + "</b>  q=" + traffic.QueueByIntersection[iid], st);
            }
            for (int z = 0; z < 4; z++)
            {
                Vector3 sp = cam.WorldToScreenPoint(city.ZoneLabelPos[z]);
                if (sp.z <= 0) continue;
                EnergyStep e = data.EnergyAt(policyIndex, z, simTime);
                Rect lr = new Rect(sp.x / s - 80, (Screen.height - sp.y) / s - 22, 160, 44);
                if (UnderPanel(lr)) continue;
                GUI.color = new Color(0, 0, 0, 0.6f);
                GUI.DrawTexture(lr, white);
                GUI.color = Color.white;
                GUI.Label(lr, string.Format("<b>{0}</b>\nBattery {1:P0} | Grid {2:N0} kW",
                    CityLayout.ZoneNames[z], e.SocFrac, e.Grid), st);
            }
        }

        void DrawChart(Rect r)
        {
            if (chart == null || chartController != controllerIndex || chartPolicy != policyIndex) BuildChart();
            GUI.DrawTexture(r, chart);
            float x = r.x + simTime / 86400f * r.width;
            GUI.color = Color.white;
            GUI.DrawTexture(new Rect(x - 1, r.y, 2, r.height), white);
        }

        void BuildChart()
        {
            chartController = controllerIndex; chartPolicy = policyIndex;
            if (chart == null) chart = new Texture2D(ChartW, ChartH, TextureFormat.RGBA32, false);
            var px = new Color[ChartW * ChartH];
            Color bg = new Color(0.08f, 0.09f, 0.12f, 0.95f);
            for (int i = 0; i < px.Length; i++) px[i] = bg;
            for (int hgrid = 0; hgrid <= 24; hgrid += 6)   // grid lines every 6 h
            {
                int gx = Mathf.Min(ChartW - 1, hgrid * ChartW / 24);
                for (int y = 0; y < ChartH; y++) px[y * ChartW + gx] = new Color(0.25f, 0.27f, 0.32f);
            }
            int maxVeh = 1;
            foreach (TrafficTrace t in data.Traces)
                foreach (int v in t.InNetworkByMinute) maxVeh = Mathf.Max(maxVeh, v);
            TrafficTrace tr = data.Traces[controllerIndex];
            int prevA = -1, prevB = -1;
            for (int x = 0; x < ChartW; x++)
            {
                int minute = Mathf.Min(1439, x * 1440 / ChartW);
                float veh = tr.InNetworkByMinute[minute] / (float)maxVeh;
                float grid = 0f;
                for (int z = 0; z < 4; z++) grid += data.EnergyAt(policyIndex, z, minute * 60f).Grid;
                grid /= Mathf.Max(1f, data.MaxCityGrid);
                int ya = Mathf.Clamp((int)(veh * (ChartH - 6)) + 3, 0, ChartH - 1);
                int yb = Mathf.Clamp((int)(grid * (ChartH - 6)) + 3, 0, ChartH - 1);
                Line(px, x, prevA < 0 ? ya : prevA, ya, new Color(0.31f, 0.64f, 1f));
                Line(px, x, prevB < 0 ? yb : prevB, yb, new Color(1f, 0.35f, 0.29f));
                prevA = ya; prevB = yb;
            }
            chart.SetPixels(px);
            chart.Apply();
        }

        static void Line(Color[] px, int x, int y0, int y1, Color c)
        {
            int a = Mathf.Min(y0, y1), b = Mathf.Max(y0, y1);
            for (int y = a; y <= b; y++)
            {
                px[y * ChartW + x] = c;
                if (y + 1 < ChartH) px[(y + 1) * ChartW + x] = c;
            }
        }

        void TakeScreenshot()
        {
            string dir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Screenshots"));
            Directory.CreateDirectory(dir);
            string name = string.Format("SmartCity_{0}_{1}_{2}.png", SmartCityData.ControllerKeys[controllerIndex],
                SmartCityData.PolicyNames[policyIndex].Replace(" ", ""), Clock(simTime).Replace(":", ""));
            lastShot = Path.Combine(dir, name);
            ScreenCapture.CaptureScreenshot(lastShot);
            lastShotTime = Time.unscaledTime;
            Debug.Log("Screenshot saved to " + lastShot);
        }

        static string Clock(float t)
        {
            int sec = Mathf.Clamp((int)t, 0, 86399);
            return string.Format("{0:00}:{1:00}:{2:00}", sec / 3600, (sec / 60) % 60, sec % 60);
        }

        static GUIStyle RichLabel(int size)
        {
            var st = new GUIStyle(GUI.skin.label) { richText = true, fontSize = size, wordWrap = true };
            st.normal.textColor = Color.white;
            return st;
        }
    }
}
