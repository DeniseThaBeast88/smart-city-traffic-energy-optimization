// CityBuilder.cs
// Builds the 3D city procedurally from primitives: ground, roads, energy-zone
// tiles, buildings with rooftop solar panels, signal heads, street lights,
// and zone battery / grid-import indicator bars. No imported art is needed,
// so the project works in any Unity render pipeline.
using System.Collections.Generic;
using UnityEngine;

namespace SmartCity
{
    public class BuildingInfo
    {
        public Renderer Body, Solar;
        public int Zone;
        public Color BaseColor;
    }

    public class CityBuilder
    {
        public Transform Root;
        public Renderer[,] SignalHeads = new Renderer[9, 4];   // [intersection, approach heading]
        public List<Renderer> Lamps = new List<Renderer>();
        public List<int> LampZone = new List<int>();
        public List<BuildingInfo> Buildings = new List<BuildingInfo>();
        public Transform[] BatteryBars = new Transform[4], GridBars = new Transform[4];
        public Renderer[] ZoneTiles = new Renderer[9];
        public Vector3[] ZoneLabelPos = new Vector3[4];

        public static readonly Color[] ZoneColors =
        {
            new Color(0.96f, 0.86f, 0.62f),   // Residential - sand
            new Color(0.62f, 0.78f, 0.95f),   // Commercial - blue
            new Color(0.80f, 0.70f, 0.90f),   // Industrial - lavender
            new Color(0.70f, 0.90f, 0.72f)    // Mixed-Use - green
        };

        static readonly Color RoadColor = new Color(0.22f, 0.22f, 0.24f);
        static readonly Color ArterialColor = new Color(0.16f, 0.16f, 0.18f);
        static readonly Color MarkingColor = new Color(0.95f, 0.95f, 0.85f);

        public static GameObject Box(string name, Transform parent, Vector3 pos, Vector3 scale, Color color)
        {
            GameObject g = GameObject.CreatePrimitive(PrimitiveType.Cube);
            g.name = name;
            Object.Destroy(g.GetComponent<Collider>());
            g.transform.SetParent(parent, false);
            g.transform.localPosition = pos;
            g.transform.localScale = scale;
            g.GetComponent<Renderer>().material.color = color;
            return g;
        }

        public static GameObject Ball(string name, Transform parent, Vector3 pos, float size, Color color)
        {
            GameObject g = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            g.name = name;
            Object.Destroy(g.GetComponent<Collider>());
            g.transform.SetParent(parent, false);
            g.transform.localPosition = pos;
            g.transform.localScale = Vector3.one * size;
            g.GetComponent<Renderer>().material.color = color;
            return g;
        }

        public void Build()
        {
            Root = new GameObject("City").transform;
            float S = CityLayout.Spacing;
            float half = S * 1.5f + 40f;

            Box("Ground", Root, new Vector3(0, -0.6f, 0), new Vector3(half * 2 + 60, 1f, half * 2 + 60),
                new Color(0.30f, 0.38f, 0.30f));

            // Energy-zone tiles (one 100 x 100 tile per intersection, as in Fig. 1 of the report)
            for (int iid = 0; iid < 9; iid++)
            {
                int z = CityLayout.ZoneOfIntersection[iid];
                Vector3 c = CityLayout.IntersectionPos(iid);
                GameObject tile = Box("ZoneTile_I" + iid, Root, c + new Vector3(0, -0.05f, 0), new Vector3(S, 0.1f, S),
                    ZoneColors[z] * 0.75f);
                ZoneTiles[iid] = tile.GetComponent<Renderer>();
            }

            // Roads: three north-south and three east-west; column 1 and row 1 are arterials
            for (int k = 0; k < 3; k++)
            {
                float off = (k - 1) * S;
                bool art = k == 1;
                Color rc = art ? ArterialColor : RoadColor;
                float w = art ? 12f : 10f;
                Box("Road_NS_col" + k, Root, new Vector3(off, 0.02f, 0), new Vector3(w, 0.1f, half * 2), rc);
                Box("Road_EW_row" + k, Root, new Vector3(0, 0.03f, -off), new Vector3(half * 2, 0.1f, w), rc);
                // dashed center line
                for (float p = -half + 4; p < half; p += 12f)
                {
                    if (NearIntersection(p)) continue;
                    Box("Mark", Root, new Vector3(off, 0.09f, p), new Vector3(0.4f, 0.02f, 5f), MarkingColor);
                    Box("Mark", Root, new Vector3(p, 0.09f, -off), new Vector3(5f, 0.02f, 0.4f), MarkingColor);
                }
            }

            // Stop lines and signal heads at every approach
            for (int iid = 0; iid < 9; iid++)
            {
                Vector3 c = CityLayout.IntersectionPos(iid);
                for (int h = 0; h < 4; h++)
                {
                    Vector3 d = CityLayout.Dir(h), r = CityLayout.Right(h);
                    Vector3 stop = c - d * CityLayout.StopOffset;
                    Vector3 lineScale = Mathf.Abs(d.x) > 0.5f ? new Vector3(0.5f, 0.02f, 5f) : new Vector3(5f, 0.02f, 0.5f);
                    Box("StopLine", Root, stop + r * 2.5f + Vector3.up * 0.1f, lineScale, Color.white);
                    Vector3 polePos = stop + r * 7.5f;
                    Box("SignalPole_I" + iid + "_" + CityLayout.HeadingNames[h], Root, polePos + Vector3.up * 3f,
                        new Vector3(0.5f, 6f, 0.5f), new Color(0.15f, 0.15f, 0.15f));
                    Box("SignalHousing", Root, polePos + Vector3.up * 6.6f, new Vector3(1.6f, 1.6f, 1.6f),
                        new Color(0.08f, 0.08f, 0.08f));
                    GameObject head = Ball("SignalHead_I" + iid + "_" + CityLayout.HeadingNames[h], Root,
                        polePos + Vector3.up * 6.6f - d * 0.6f, 1.9f, Color.red);
                    SignalHeads[iid, h] = head.GetComponent<Renderer>();
                }
            }

            // Buildings: four per tile, sized by zone type, each with a rooftop solar panel
            var rng = new System.Random(42);
            for (int iid = 0; iid < 9; iid++)
            {
                int z = CityLayout.ZoneOfIntersection[iid];
                Vector3 c = CityLayout.IntersectionPos(iid);
                for (int q = 0; q < 4; q++)
                {
                    float sx = (q % 2 == 0) ? -1 : 1, sz = (q < 2) ? -1 : 1;
                    float fx, fz, h;
                    switch (z)
                    {
                        case 0: fx = 24; fz = 24; h = 6 + (float)rng.NextDouble() * 6; break;    // houses / apartments
                        case 1: fx = 22; fz = 22; h = 30 + (float)rng.NextDouble() * 35; break;  // office towers
                        case 2: fx = 32; fz = 30; h = 10 + (float)rng.NextDouble() * 5; break;   // factories
                        default: fx = 24; fz = 24; h = 15 + (float)rng.NextDouble() * 15; break; // mixed use
                    }
                    Vector3 p = c + new Vector3(sx * 27.5f, h / 2f, sz * 27.5f);
                    Color bc = Color.Lerp(ZoneColors[z], new Color(0.55f, 0.55f, 0.58f), 0.45f);
                    GameObject b = Box("Building_" + CityLayout.ZoneNames[z] + "_I" + iid + "_" + q, Root, p,
                        new Vector3(fx, h, fz), bc);
                    GameObject solar = Box("SolarPanel", Root, new Vector3(p.x, h + 0.25f, p.z),
                        new Vector3(fx * 0.7f, 0.4f, fz * 0.7f), new Color(0.1f, 0.15f, 0.35f));
                    Buildings.Add(new BuildingInfo
                    {
                        Body = b.GetComponent<Renderer>(), Solar = solar.GetComponent<Renderer>(), Zone = z, BaseColor = bc
                    });
                }
            }

            // Street lights along every road, assigned to the zone of the nearest intersection
            for (int k = 0; k < 3; k++)
            {
                float off = (k - 1) * S;
                for (float p = -half + 10; p < half - 5; p += 20f)
                {
                    if (NearIntersection(p)) continue;
                    AddLamp(new Vector3(off + 7.5f, 0, p));
                    AddLamp(new Vector3(p, 0, -off - 7.5f));
                }
            }

            // Zone indicator bars: battery state of charge (green) and grid import (red)
            for (int z = 0; z < 4; z++)
            {
                Vector3 basePos = BarPositions[z];
                ZoneLabelPos[z] = basePos + Vector3.up * 52f;
                Box("BatteryFrame_" + CityLayout.ZoneNames[z], Root, basePos + new Vector3(-3, 20, 0),
                    new Vector3(4.6f, 40.2f, 4.6f), new Color(0.1f, 0.1f, 0.1f));
                BatteryBars[z] = Box("BatteryLevel_" + CityLayout.ZoneNames[z], Root, basePos + new Vector3(-3, 10, 0),
                    new Vector3(4.2f, 20f, 4.2f), new Color(0.2f, 0.85f, 0.35f)).transform;
                GridBars[z] = Box("GridImport_" + CityLayout.ZoneNames[z], Root, basePos + new Vector3(3, 10, 0),
                    new Vector3(4.2f, 20f, 4.2f), new Color(0.9f, 0.3f, 0.25f)).transform;
            }
        }

        // Indicator bars sit in the margin outside the 3 x 3 grid, between roads, beside each zone
        static readonly Vector3[] BarPositions =
        {
            new Vector3(-50f, 0f, 172f),    // Residential (north-west tiles)
            new Vector3(172f, 0f, 50f),     // Commercial (north-east and center tiles)
            new Vector3(50f, 0f, -172f),    // Industrial (south tiles)
            new Vector3(-172f, 0f, -50f)    // Mixed-Use (south-west tile)
        };

        static bool NearIntersection(float p)
        {
            for (int k = -1; k <= 1; k++)
                if (Mathf.Abs(p - k * CityLayout.Spacing) < 14f) return true;
            return false;
        }

        void AddLamp(Vector3 pos)
        {
            // nearest intersection decides the zone that powers this light
            int best = 0; float bd = float.MaxValue;
            for (int iid = 0; iid < 9; iid++)
            {
                float dd = (CityLayout.IntersectionPos(iid) - pos).sqrMagnitude;
                if (dd < bd) { bd = dd; best = iid; }
            }
            Box("LampPost", Root, pos + Vector3.up * 3f, new Vector3(0.35f, 6f, 0.35f), new Color(0.3f, 0.3f, 0.32f));
            GameObject head = Ball("LampHead", Root, pos + Vector3.up * 6.3f, 1.3f, Color.gray);
            Lamps.Add(head.GetComponent<Renderer>());
            LampZone.Add(CityLayout.ZoneOfIntersection[best]);
        }
    }
}
