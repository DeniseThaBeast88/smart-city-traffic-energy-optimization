// SmartCityData.cs
// Loads the simulation traces exported by export_unity_data.py (Python/SimPy)
// from Assets/StreamingAssets/SmartCityData and stores them in compact arrays.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace SmartCity
{
    /// <summary>City layout constants shared by every script (1 Unity unit = 3 m).</summary>
    public static class CityLayout
    {
        public const int Rows = 3, Cols = 3;
        public const float Spacing = 100f;      // 300 m between intersections
        public const float StopOffset = 7f;     // stop line distance from intersection center
        public const float LaneOffset = 2.5f;   // right-hand lane offset from road center
        public const float QueueSpacing = 2.5f; // 7.5 m per queued vehicle
        public const float EntryDistance = 60f; // where entering vehicles appear
        public const float EntrySeconds = 8f;   // time to drive from the edge to the stop line
        public const float ExitSeconds = 11f;   // half-link travel time when leaving (from Python)
        public const float LinkSeconds = 22f;   // free-flow link travel time (from Python)

        // Heading codes match the Python model: N=0, S=1, E=2, W=3 (direction of travel).
        public static readonly string[] HeadingNames = { "N", "S", "E", "W" };
        public static readonly int[] DRow = { -1, 1, 0, 0 };
        public static readonly int[] DCol = { 0, 0, 1, -1 };

        public static Vector3 IntersectionPos(int iid)
        {
            int r = iid / Cols, c = iid % Cols;
            return new Vector3((c - 1) * Spacing, 0f, (1 - r) * Spacing);
        }

        public static Vector3 Dir(int h)
        {
            switch (h)
            {
                case 0: return Vector3.forward;  // north
                case 1: return Vector3.back;     // south
                case 2: return Vector3.right;    // east
                default: return Vector3.left;    // west
            }
        }

        /// <summary>Unit vector to the right of the direction of travel (right-hand traffic).</summary>
        public static Vector3 Right(int h)
        {
            Vector3 d = Dir(h);
            return new Vector3(d.z, 0f, -d.x);
        }

        /// <summary>Neighbor intersection in heading h, or -1 when the vehicle leaves the grid.</summary>
        public static int Neighbor(int iid, int h)
        {
            int r = iid / Cols + DRow[h], c = iid % Cols + DCol[h];
            if (r < 0 || r >= Rows || c < 0 || c >= Cols) return -1;
            return r * Cols + c;
        }

        public static int HeadingCode(string s)
        {
            switch (s)
            {
                case "N": return 0;
                case "S": return 1;
                case "E": return 2;
                case "W": return 3;
                default: return -1;
            }
        }

        public static readonly string[] ZoneNames = { "Residential", "Commercial", "Industrial", "Mixed-Use" };
        // Intersection -> zone index (same mapping as ZONES in smart_city_sim.py)
        public static readonly int[] ZoneOfIntersection = { 0, 0, 1, 0, 1, 1, 3, 2, 2 };
        public static readonly int[] StreetLightsPerZone = { 70, 90, 50, 60 };
        public const float LightKw = 0.15f;
    }

    /// <summary>All trips and signal changes for one traffic controller on evaluation day 1.</summary>
    public class TrafficTrace
    {
        public string Key, DisplayName;

        // Hop arrays (one entry per vehicle visit to an intersection)
        public byte[] HopIid, HopAppr, HopNext;   // HopNext = 255 when the vehicle is still queued at day end
        public float[] HopArr, HopDep;            // HopDep < 0 when the vehicle never departed

        // Vehicle arrays, sorted by spawn time
        public int[] VehFirstHop, VehHopCount;
        public float[] VehSpawn, VehEnd;
        public float MaxLifetime;

        // Signal events per intersection
        public List<float>[] SigTime = new List<float>[9];
        public List<byte>[] SigPhase = new List<byte>[9];
        public List<bool>[] SigYellow = new List<bool>[9];

        // Completed trips sorted by exit time, with cumulative delay sums
        public float[] ExitTimes;
        public double[] CumDelay;

        // Vehicles in the network for each minute of the day (for the dashboard chart)
        public int[] InNetworkByMinute = new int[1440];

        /// <summary>Signal phase (0 = north-south green, 1 = east-west green) and yellow flag at time t.</summary>
        public void SignalAt(int iid, float t, out int phase, out bool yellow)
        {
            List<float> times = SigTime[iid];
            int lo = 0, hi = times.Count - 1, idx = -1;
            while (lo <= hi)
            {
                int mid = (lo + hi) / 2;
                if (times[mid] <= t) { idx = mid; lo = mid + 1; } else hi = mid - 1;
            }
            if (idx < 0) { phase = 0; yellow = false; return; }
            phase = SigPhase[iid][idx];
            yellow = SigYellow[iid][idx];
        }

        /// <summary>Number of completed trips and their average delay up to time t.</summary>
        public void CompletedAt(float t, out int count, out double avgDelay)
        {
            int lo = 0, hi = ExitTimes.Length;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (ExitTimes[mid] <= t) lo = mid + 1; else hi = mid;
            }
            count = lo;
            avgDelay = count > 0 ? CumDelay[count - 1] / count : 0.0;
        }
    }

    /// <summary>One 15-minute energy dispatch record for one zone and one policy.</summary>
    public struct EnergyStep
    {
        public float Price, Load, LightKw, Pv, BatCharge, BatDischarge, Grid, ShareOut, ShareIn, Curtail, Density, SocFrac, LightWaste;
    }

    public class SmartCityData
    {
        public static readonly string[] ControllerKeys = { "fixed", "rule", "rl" };
        public static readonly string[] ControllerNames = { "Fixed-time", "Rule-based actuated", "Q-learning (RL)" };
        public static readonly string[] PolicyNames = { "Baseline", "AI-optimized" };

        public TrafficTrace[] Traces = new TrafficTrace[3];
        public EnergyStep[,,] Energy = new EnergyStep[2, 4, 96];   // [policy, zone, step]
        public float[] ZoneMaxLoad = new float[4], ZoneMaxPv = new float[4];
        public float MaxCityGrid;
        // running totals from midnight, [policy, completed 15-min steps 0..96]
        public float[,] CumGridKwh = new float[2, 97], CumCostUsd = new float[2, 97],
                        CumWasteKwh = new float[2, 97], CumLightKwh = new float[2, 97];
        public Dictionary<string, float> Summary = new Dictionary<string, float>();
        public string Folder;

        static readonly CultureInfo Inv = CultureInfo.InvariantCulture;
        static float F(string s) { return float.Parse(s, Inv); }

        public static SmartCityData Load()
        {
            var d = new SmartCityData();
            d.Folder = Path.Combine(Application.streamingAssetsPath, "SmartCityData");
            if (!Directory.Exists(d.Folder))
                throw new DirectoryNotFoundException("Simulation data folder not found: " + d.Folder +
                    ". Run 'python export_unity_data.py' in the smart_city_project folder.");
            for (int i = 0; i < 3; i++)
                d.Traces[i] = LoadTrace(d.Folder, ControllerKeys[i], ControllerNames[i]);
            d.LoadEnergy();
            d.LoadSummary();
            return d;
        }

        static TrafficTrace LoadTrace(string folder, string key, string name)
        {
            var tr = new TrafficTrace { Key = key, DisplayName = name };

            // ---- hops: vid,iid,appr,next,t_arr,t_dep (sorted by vid, t_arr)
            string[] lines = File.ReadAllLines(Path.Combine(folder, "hops_" + key + ".csv"));
            int n = lines.Length - 1;
            tr.HopIid = new byte[n]; tr.HopAppr = new byte[n]; tr.HopNext = new byte[n];
            tr.HopArr = new float[n]; tr.HopDep = new float[n];
            var vFirst = new List<int>(); var vCount = new List<int>();
            int lastVid = -1;
            for (int i = 0; i < n; i++)
            {
                string[] p = lines[i + 1].Split(',');
                int vid = int.Parse(p[0], Inv);
                tr.HopIid[i] = byte.Parse(p[1], Inv);
                tr.HopAppr[i] = (byte)CityLayout.HeadingCode(p[2]);
                int nx = CityLayout.HeadingCode(p[3]);
                tr.HopNext[i] = nx < 0 ? (byte)255 : (byte)nx;
                tr.HopArr[i] = F(p[4]);
                tr.HopDep[i] = F(p[5]);
                if (vid != lastVid) { vFirst.Add(i); vCount.Add(0); lastVid = vid; }
                vCount[vCount.Count - 1]++;
            }

            // ---- vehicle lifetimes
            int nv = vFirst.Count;
            var spawn = new float[nv]; var end = new float[nv];
            for (int v = 0; v < nv; v++)
            {
                int h0 = vFirst[v], last = h0 + vCount[v] - 1;
                spawn[v] = tr.HopArr[h0] - CityLayout.EntrySeconds;
                float dep = tr.HopDep[last];
                if (dep < 0f) end[v] = 86400f;                                    // still queued at day end
                else if (CityLayout.Neighbor(tr.HopIid[last], tr.HopNext[last]) < 0)
                    end[v] = dep + CityLayout.ExitSeconds;                          // left the network
                else end[v] = dep + CityLayout.LinkSeconds;                         // in transit at day end
            }
            int[] order = new int[nv];
            for (int v = 0; v < nv; v++) order[v] = v;
            Array.Sort((float[])spawn.Clone(), order);
            tr.VehFirstHop = new int[nv]; tr.VehHopCount = new int[nv];
            tr.VehSpawn = new float[nv]; tr.VehEnd = new float[nv];
            int[] diff = new int[1441];
            for (int k = 0; k < nv; k++)
            {
                int v = order[k];
                tr.VehFirstHop[k] = vFirst[v]; tr.VehHopCount[k] = vCount[v];
                tr.VehSpawn[k] = spawn[v]; tr.VehEnd[k] = end[v];
                float life = Mathf.Min(end[v], 86400f) - spawn[v];
                if (life > tr.MaxLifetime) tr.MaxLifetime = life;
                int m0 = Mathf.Clamp(Mathf.CeilToInt(Mathf.Max(0f, spawn[v] + CityLayout.EntrySeconds) / 60f), 0, 1440);
                int m1 = Mathf.Clamp(Mathf.CeilToInt(end[v] / 60f), 0, 1440);
                if (m1 > m0) { diff[m0]++; diff[m1]--; }
            }
            int run = 0;
            for (int m = 0; m < 1440; m++) { run += diff[m]; tr.InNetworkByMinute[m] = run; }

            // ---- signals: t,iid,phase,state
            for (int i = 0; i < 9; i++)
            {
                tr.SigTime[i] = new List<float>(); tr.SigPhase[i] = new List<byte>(); tr.SigYellow[i] = new List<bool>();
            }
            string[] sl = File.ReadAllLines(Path.Combine(folder, "signals_" + key + ".csv"));
            for (int i = 1; i < sl.Length; i++)
            {
                string[] p = sl[i].Split(',');
                int iid = int.Parse(p[1], Inv);
                tr.SigTime[iid].Add(F(p[0]));
                tr.SigPhase[iid].Add(byte.Parse(p[2], Inv));
                tr.SigYellow[iid].Add(p[3].Trim() == "Y");
            }

            // ---- completed trips: vid,t_enter,t_exit,delay
            string[] vl = File.ReadAllLines(Path.Combine(folder, "vehicles_" + key + ".csv"));
            int nc = vl.Length - 1;
            var exits = new float[nc]; var delays = new float[nc];
            for (int i = 0; i < nc; i++)
            {
                string[] p = vl[i + 1].Split(',');
                exits[i] = F(p[2]); delays[i] = F(p[3]);
            }
            Array.Sort(exits, delays);
            tr.ExitTimes = exits;
            tr.CumDelay = new double[nc];
            double sum = 0;
            for (int i = 0; i < nc; i++) { sum += delays[i]; tr.CumDelay[i] = sum; }
            return tr;
        }

        void LoadEnergy()
        {
            // time_h,zone,policy,price,load,light_kw,light_waste,net,pv,bat_ch,bat_dis,grid,
            // share_out,share_in,curtail,dens,soc,soc_frac
            string[] lines = File.ReadAllLines(Path.Combine(Folder, "energy_dispatch.csv"));
            string[] head = lines[0].Split(',');
            var col = new Dictionary<string, int>();
            for (int i = 0; i < head.Length; i++) col[head[i].Trim()] = i;
            var cityGrid = new float[2, 96];
            for (int i = 1; i < lines.Length; i++)
            {
                string[] p = lines[i].Split(',');
                int step = Mathf.Clamp(Mathf.RoundToInt(F(p[col["time_h"]]) * 4f), 0, 95);
                int zone = Array.IndexOf(CityLayout.ZoneNames, p[col["zone"]]);
                int pol = p[col["policy"]] == "Baseline" ? 0 : 1;
                if (zone < 0) continue;
                var e = new EnergyStep
                {
                    Price = F(p[col["price"]]), Load = F(p[col["load"]]), LightKw = F(p[col["light_kw"]]),
                    Pv = F(p[col["pv"]]), BatCharge = F(p[col["bat_ch"]]), BatDischarge = F(p[col["bat_dis"]]),
                    Grid = F(p[col["grid"]]), ShareOut = F(p[col["share_out"]]), ShareIn = F(p[col["share_in"]]),
                    Curtail = F(p[col["curtail"]]), Density = F(p[col["dens"]]), SocFrac = F(p[col["soc_frac"]]),
                    LightWaste = col.ContainsKey("light_waste") ? F(p[col["light_waste"]]) : 0f
                };
                Energy[pol, zone, step] = e;
                ZoneMaxLoad[zone] = Mathf.Max(ZoneMaxLoad[zone], e.Load);
                ZoneMaxPv[zone] = Mathf.Max(ZoneMaxPv[zone], e.Pv);
                cityGrid[pol, step] += e.Grid;
            }
            for (int pol = 0; pol < 2; pol++)
                for (int s = 0; s < 96; s++) MaxCityGrid = Mathf.Max(MaxCityGrid, cityGrid[pol, s]);

            for (int pol = 0; pol < 2; pol++)
                for (int s = 0; s < 96; s++)
                {
                    float g = 0f, c = 0f, w = 0f, l = 0f;
                    for (int z = 0; z < 4; z++)
                    {
                        EnergyStep e = Energy[pol, z, s];
                        g += e.Grid * 0.25f;
                        c += e.Grid * e.Price * 0.25f;
                        w += (e.Curtail + e.LightWaste) * 0.25f;
                        l += e.LightKw * 0.25f;
                    }
                    CumGridKwh[pol, s + 1] = CumGridKwh[pol, s] + g;
                    CumCostUsd[pol, s + 1] = CumCostUsd[pol, s] + c;
                    CumWasteKwh[pol, s + 1] = CumWasteKwh[pol, s] + w;
                    CumLightKwh[pol, s + 1] = CumLightKwh[pol, s] + l;
                }
        }

        /// <summary>Energy totals for one policy from midnight up to time t (seconds).</summary>
        public void TotalsAt(int policy, float t, out float gridKwh, out float costUsd, out float wasteKwh, out float lightKwh)
        {
            float k = Mathf.Clamp(t / 900f, 0f, 96f);
            int full = Mathf.Min(95, Mathf.FloorToInt(k));
            float frac = Mathf.Clamp01(k - full);
            gridKwh = Mathf.Lerp(CumGridKwh[policy, full], CumGridKwh[policy, full + 1], frac);
            costUsd = Mathf.Lerp(CumCostUsd[policy, full], CumCostUsd[policy, full + 1], frac);
            wasteKwh = Mathf.Lerp(CumWasteKwh[policy, full], CumWasteKwh[policy, full + 1], frac);
            lightKwh = Mathf.Lerp(CumLightKwh[policy, full], CumLightKwh[policy, full + 1], frac);
        }

        /// <summary>City-wide totals for one policy at the 15-minute step containing time t.</summary>
        public void CityNowAt(int policy, float t, out float grid, out float curtail, out float light, out float battery)
        {
            grid = curtail = light = battery = 0f;
            for (int z = 0; z < 4; z++)
            {
                EnergyStep e = EnergyAt(policy, z, t);
                grid += e.Grid; curtail += e.Curtail; light += e.LightKw; battery += e.SocFrac / 4f;
            }
        }

        void LoadSummary()
        {
            string path = Path.Combine(Folder, "summary.csv");
            if (!File.Exists(path)) return;
            string[] lines = File.ReadAllLines(path);
            for (int i = 1; i < lines.Length; i++)
            {
                string[] p = lines[i].Split(',');
                if (p.Length < 4) continue;
                Summary[p[1] + "|" + p[2]] = F(p[3]);
            }
        }

        public float SummaryValue(string name, string metric)
        {
            float v;
            return Summary.TryGetValue(name + "|" + metric, out v) ? v : float.NaN;
        }

        public EnergyStep EnergyAt(int policy, int zone, float t)
        {
            int s = Mathf.Clamp(Mathf.FloorToInt(t / 900f), 0, 95);
            return Energy[policy, zone, s];
        }
    }
}
