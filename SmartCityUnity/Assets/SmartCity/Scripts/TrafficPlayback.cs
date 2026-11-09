// TrafficPlayback.cs
// Replays the SimPy traffic trace: every vehicle drives in from the city edge,
// waits in its stop-line queue until the simulated signal releases it, crosses
// links between intersections, and leaves the grid. Signal heads show the
// exact phase changes chosen by the selected controller.
using System.Collections.Generic;
using UnityEngine;

namespace SmartCity
{
    public class TrafficPlayback
    {
        public int ActiveVehicles, QueuedVehicles, LongWaits;
        public int[] QueueByIntersection = new int[9];

        readonly CityBuilder city;
        readonly Transform carRoot;
        readonly List<Transform> pool = new List<Transform>();
        readonly List<Renderer> poolRenderers = new List<Renderer>();
        readonly List<Color> poolColors = new List<Color>();

        // per-frame scratch lists
        readonly List<int>[] queueVeh = new List<int>[36];       // vehicle index per (iid * 4 + heading)
        readonly List<float>[] queueArr = new List<float>[36];
        struct Mover { public int Veh, Kind, Iid, Head, TargetKey; public float Progress; }
        readonly List<Mover> movers = new List<Mover>();
        readonly int[] queueCount = new int[36];

        static readonly Color MovingColor = new Color(0.92f, 0.95f, 1.0f);
        static readonly Color ShortWait = new Color(1.0f, 0.75f, 0.2f);
        static readonly Color LongWait = new Color(0.85f, 0.1f, 0.1f);
        static readonly Color SignalRed = new Color(0.9f, 0.08f, 0.05f);
        static readonly Color SignalYellow = new Color(1f, 0.8f, 0.05f);
        static readonly Color SignalGreen = new Color(0.1f, 0.95f, 0.25f);

        public TrafficPlayback(CityBuilder city)
        {
            this.city = city;
            carRoot = new GameObject("Vehicles").transform;
            for (int i = 0; i < 36; i++) { queueVeh[i] = new List<int>(); queueArr[i] = new List<float>(); }
        }

        Transform GetCar(int i)
        {
            while (pool.Count <= i)
            {
                GameObject car = CityBuilder.Box("Car", carRoot, Vector3.zero, new Vector3(1.4f, 1.0f, 2.2f), MovingColor);
                CityBuilder.Box("Roof", car.transform, new Vector3(0, 0.7f, -0.1f), new Vector3(0.8f, 0.5f, 0.5f),
                    new Color(0.15f, 0.2f, 0.3f));
                pool.Add(car.transform);
                poolRenderers.Add(car.GetComponent<Renderer>());
                poolColors.Add(MovingColor);
            }
            return pool[i];
        }

        void Place(int slot, Vector3 pos, int heading, Color color)
        {
            Transform t = GetCar(slot);
            if (!t.gameObject.activeSelf) t.gameObject.SetActive(true);
            t.position = pos + Vector3.up * 0.55f;
            t.rotation = Quaternion.LookRotation(CityLayout.Dir(heading));
            if (poolColors[slot] != color)
            {
                poolRenderers[slot].material.color = color;
                poolColors[slot] = color;
            }
        }

        static Vector3 StopPoint(int iid, int h)
        {
            return CityLayout.IntersectionPos(iid) - CityLayout.Dir(h) * CityLayout.StopOffset
                   + CityLayout.Right(h) * CityLayout.LaneOffset;
        }

        public void UpdateSignals(TrafficTrace tr, float t)
        {
            for (int iid = 0; iid < 9; iid++)
            {
                int phase; bool yellow;
                tr.SignalAt(iid, t, out phase, out yellow);
                for (int h = 0; h < 4; h++)
                {
                    bool served = (phase == 0) ? (h <= 1) : (h >= 2);   // phase 0 = N/S travel, phase 1 = E/W
                    Color c = !served ? SignalRed : (yellow ? SignalYellow : SignalGreen);
                    city.SignalHeads[iid, h].material.color = c;
                }
            }
        }

        public void UpdateVehicles(TrafficTrace tr, float t)
        {
            for (int i = 0; i < 36; i++) { queueVeh[i].Clear(); queueArr[i].Clear(); }
            movers.Clear();
            for (int i = 0; i < 9; i++) QueueByIntersection[i] = 0;
            LongWaits = 0;

            // candidate vehicles: spawned before t and not older than the longest lifetime
            int hi = UpperBound(tr.VehSpawn, t);
            int lo = UpperBound(tr.VehSpawn, t - tr.MaxLifetime - 1f);
            for (int v = lo; v < hi; v++)
            {
                if (tr.VehEnd[v] <= t) continue;
                int h0 = tr.VehFirstHop[v], hn = tr.VehHopCount[v];
                if (t < tr.HopArr[h0])
                {
                    // entering from the city edge toward the first stop line
                    float p = Mathf.Clamp01((t - tr.VehSpawn[v]) / CityLayout.EntrySeconds);
                    int key = tr.HopIid[h0] * 4 + tr.HopAppr[h0];
                    movers.Add(new Mover { Veh = v, Kind = 0, Iid = tr.HopIid[h0], Head = tr.HopAppr[h0], TargetKey = key, Progress = p });
                    continue;
                }
                int k = h0;
                while (k + 1 < h0 + hn && tr.HopArr[k + 1] <= t) k++;
                float dep = tr.HopDep[k];
                if (dep < 0f || t < dep)
                {
                    int key = tr.HopIid[k] * 4 + tr.HopAppr[k];
                    queueVeh[key].Add(v);
                    queueArr[key].Add(tr.HopArr[k]);
                    continue;
                }
                int nh = tr.HopNext[k];
                int next = CityLayout.Neighbor(tr.HopIid[k], nh);
                if (next < 0)
                {
                    float p = Mathf.Clamp01((t - dep) / CityLayout.ExitSeconds);
                    movers.Add(new Mover { Veh = v, Kind = 2, Iid = tr.HopIid[k], Head = nh, TargetKey = -1, Progress = p });
                }
                else
                {
                    float arrive = (k + 1 < h0 + hn) ? tr.HopArr[k + 1] : dep + CityLayout.LinkSeconds;
                    float p = Mathf.Clamp01((t - dep) / Mathf.Max(1f, arrive - dep));
                    movers.Add(new Mover { Veh = v, Kind = 1, Iid = tr.HopIid[k], Head = nh, TargetKey = next * 4 + nh, Progress = p });
                }
            }

            int slot = 0;
            QueuedVehicles = 0;
            // queued vehicles, ordered by arrival time at the stop line
            for (int key = 0; key < 36; key++)
            {
                List<int> qv = queueVeh[key];
                queueCount[key] = qv.Count;
                if (qv.Count == 0) continue;
                List<float> qa = queueArr[key];
                SortByArrival(qv, qa);
                int iid = key / 4, h = key % 4;
                QueueByIntersection[iid] += qv.Count;
                QueuedVehicles += qv.Count;
                Vector3 stop = StopPoint(iid, h), d = CityLayout.Dir(h);
                for (int i = 0; i < qv.Count; i++)
                {
                    float waited = t - qa[i];
                    if (waited > 60f) LongWaits++;
                    Color c = waited < 2f ? MovingColor : Color.Lerp(ShortWait, LongWait, Mathf.Clamp01(waited / 60f));
                    Place(slot++, stop - d * (i * CityLayout.QueueSpacing), h, c);
                }
            }

            // moving vehicles: never drive into the back of a queue
            for (int i = 0; i < movers.Count; i++)
            {
                Mover m = movers[i];
                Vector3 pos;
                Vector3 d = CityLayout.Dir(m.Head);
                if (m.Kind == 2)
                {
                    pos = CityLayout.IntersectionPos(m.Iid) + CityLayout.Right(m.Head) * CityLayout.LaneOffset
                          + d * (m.Progress * 55f);
                }
                else
                {
                    int tIid = m.TargetKey / 4;
                    float length = m.Kind == 0 ? CityLayout.EntryDistance : CityLayout.Spacing - CityLayout.StopOffset;
                    float remaining = (1f - m.Progress) * length;
                    float queueTail = queueCount[m.TargetKey] * CityLayout.QueueSpacing;
                    remaining = Mathf.Max(remaining, queueTail);
                    pos = StopPoint(tIid, m.Head) - d * remaining;
                }
                Place(slot++, pos, m.Head, MovingColor);
            }

            ActiveVehicles = slot;
            for (int i = slot; i < pool.Count; i++)
                if (pool[i].gameObject.activeSelf) pool[i].gameObject.SetActive(false);
        }

        static void SortByArrival(List<int> veh, List<float> arr)
        {
            // insertion sort: queues are short and almost sorted already
            for (int i = 1; i < arr.Count; i++)
            {
                float a = arr[i]; int v = veh[i]; int j = i - 1;
                while (j >= 0 && arr[j] > a) { arr[j + 1] = arr[j]; veh[j + 1] = veh[j]; j--; }
                arr[j + 1] = a; veh[j + 1] = v;
            }
        }

        static int UpperBound(float[] a, float x)
        {
            int lo = 0, hi = a.Length;
            while (lo < hi)
            {
                int mid = (lo + hi) / 2;
                if (a[mid] <= x) lo = mid + 1; else hi = mid;
            }
            return lo;
        }
    }
}
