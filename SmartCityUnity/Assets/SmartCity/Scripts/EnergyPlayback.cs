// EnergyPlayback.cs
// Animates the energy grid from the 15-minute dispatch records produced by the
// Python EnergyGridManager: building demand, rooftop solar output and
// curtailment, battery state of charge, grid import, and street-light dimming.
using UnityEngine;

namespace SmartCity
{
    public enum GroundView { ZoneColors, GridImportHeatmap, TrafficQueueHeatmap }

    public class EnergyPlayback
    {
        public float CityLoad, CityPv, CityGrid, CityLighting, CityCurtail, CityBatteryFrac, Price;

        readonly CityBuilder city;
        readonly SmartCityData data;
        readonly float[] zoneMaxGrid = new float[4];

        static readonly Color Warm = new Color(1.0f, 0.55f, 0.2f);
        static readonly Color PanelIdle = new Color(0.08f, 0.1f, 0.25f);
        static readonly Color PanelActive = new Color(0.25f, 0.75f, 1.0f);
        static readonly Color PanelCurtailed = new Color(1.0f, 0.2f, 0.6f);
        static readonly Color LampOff = new Color(0.35f, 0.35f, 0.35f);
        static readonly Color LampOn = new Color(1.0f, 0.95f, 0.55f);
        static readonly Color HeatLow = new Color(0.2f, 0.6f, 0.3f);
        static readonly Color HeatMid = new Color(1.0f, 0.85f, 0.2f);
        static readonly Color HeatHigh = new Color(0.85f, 0.1f, 0.1f);

        public EnergyPlayback(CityBuilder city, SmartCityData data)
        {
            this.city = city;
            this.data = data;
            for (int p = 0; p < 2; p++)
                for (int z = 0; z < 4; z++)
                    for (int s = 0; s < 96; s++)
                        zoneMaxGrid[z] = Mathf.Max(zoneMaxGrid[z], data.Energy[p, z, s].Grid);
        }

        public static Color Heat(float x)
        {
            x = Mathf.Clamp01(x);
            return x < 0.5f ? Color.Lerp(HeatLow, HeatMid, x * 2f) : Color.Lerp(HeatMid, HeatHigh, (x - 0.5f) * 2f);
        }

        public void UpdateEnergy(int policy, float t, GroundView view, int[] queueByIntersection)
        {
            CityLoad = CityPv = CityGrid = CityLighting = CityCurtail = 0f;
            float soc = 0f;
            var steps = new EnergyStep[4];
            for (int z = 0; z < 4; z++)
            {
                EnergyStep e = data.EnergyAt(policy, z, t);
                steps[z] = e;
                CityLoad += e.Load; CityPv += e.Pv; CityGrid += e.Grid;
                CityLighting += e.LightKw; CityCurtail += e.Curtail; soc += e.SocFrac;
                Price = e.Price;

                // battery and grid-import bars (40 units = full scale)
                SetBar(city.BatteryBars[z], Mathf.Clamp01(e.SocFrac) * 40f);
                SetBar(city.GridBars[z], zoneMaxGrid[z] > 0 ? e.Grid / zoneMaxGrid[z] * 40f : 0f);
            }
            CityBatteryFrac = soc / 4f;

            foreach (BuildingInfo b in city.Buildings)
            {
                EnergyStep e = steps[b.Zone];
                float load = data.ZoneMaxLoad[b.Zone] > 0 ? e.Load / data.ZoneMaxLoad[b.Zone] : 0f;
                b.Body.material.color = Color.Lerp(b.BaseColor, Warm, Mathf.Clamp01(load) * 0.7f);
                float pv = data.ZoneMaxPv[b.Zone] > 0 ? e.Pv / data.ZoneMaxPv[b.Zone] : 0f;
                b.Solar.material.color = e.Curtail > 1f ? PanelCurtailed : Color.Lerp(PanelIdle, PanelActive, pv);
            }

            for (int i = 0; i < city.Lamps.Count; i++)
            {
                int z = city.LampZone[i];
                float full = CityLayout.StreetLightsPerZone[z] * CityLayout.LightKw;
                float level = full > 0 ? Mathf.Clamp01(steps[z].LightKw / full) : 0f;
                city.Lamps[i].material.color = level <= 0.01f ? LampOff : Color.Lerp(LampOff, LampOn, 0.25f + 0.75f * level);
            }

            for (int iid = 0; iid < 9; iid++)
            {
                int z = CityLayout.ZoneOfIntersection[iid];
                Color c;
                if (view == GroundView.GridImportHeatmap)
                    c = Heat(zoneMaxGrid[z] > 0 ? steps[z].Grid / zoneMaxGrid[z] : 0f);
                else if (view == GroundView.TrafficQueueHeatmap)
                    c = Heat(queueByIntersection[iid] / 30f);
                else
                    c = CityBuilder.ZoneColors[z] * 0.75f;
                city.ZoneTiles[iid].material.color = c;
            }
        }

        static void SetBar(Transform bar, float height)
        {
            height = Mathf.Max(0.1f, height);
            Vector3 s = bar.localScale; s.y = height; bar.localScale = s;
            Vector3 p = bar.localPosition; p.y = height / 2f; bar.localPosition = p;
        }
    }
}
