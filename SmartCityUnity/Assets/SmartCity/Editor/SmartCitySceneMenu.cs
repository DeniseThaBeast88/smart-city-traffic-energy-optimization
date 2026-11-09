// SmartCitySceneMenu.cs (Editor only)
// Adds "Smart City > Create Playback Scene" to the Unity menu bar. It creates
// Assets/Scenes/SmartCity.unity with a camera, a sun light, and the
// SmartCityApp component, saves it, and adds it to the build settings.
using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace SmartCity.EditorTools
{
    public static class SmartCitySceneMenu
    {
        const string ScenePath = "Assets/Scenes/SmartCity.unity";

        [MenuItem("Smart City/Create Playback Scene")]
        public static void CreateScene()
        {
            if (!EditorSceneManager.SaveCurrentModifiedScenesIfUserWantsTo()) return;
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var camGo = new GameObject("Main Camera");
            camGo.tag = "MainCamera";
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            camGo.transform.position = new Vector3(0, 300, -250);
            camGo.transform.LookAt(Vector3.zero);

            var sunGo = new GameObject("Sun");
            var sun = sunGo.AddComponent<Light>();
            sun.type = LightType.Directional;
            sun.shadows = LightShadows.Soft;
            sunGo.transform.rotation = Quaternion.Euler(50, -40, 0);
            RenderSettings.sun = sun;

            new GameObject("SmartCityApp").AddComponent<SmartCityApp>();

            Directory.CreateDirectory("Assets/Scenes");
            EditorSceneManager.SaveScene(scene, ScenePath);

            var list = new List<EditorBuildSettingsScene>(EditorBuildSettings.scenes);
            if (!list.Exists(s => s.path == ScenePath)) list.Insert(0, new EditorBuildSettingsScene(ScenePath, true));
            EditorBuildSettings.scenes = list.ToArray();
            AssetDatabase.Refresh();
            EditorUtility.DisplayDialog("Smart City",
                "Scene saved to " + ScenePath + ".\nPress Play to start the smart-city playback.", "OK");
        }

        [MenuItem("Smart City/Open Screenshots Folder")]
        public static void OpenScreenshots()
        {
            string dir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "Screenshots"));
            Directory.CreateDirectory(dir);
            EditorUtility.RevealInFinder(dir);
        }
    }
}
