#!/usr/bin/env python
"""3D 轨道演示（可选组件，需要 `pip install .[viz3d]`）。

低轨卫星沿圆轨道运动，实时显示卫星到最近 3 架 UAV 的距离；
UAV 群体分布在轨道星下点轨迹附近。合并自原项目 3D_main.py 与
class_show.py 两个高度雷同的 demo，并加入随机种子控制。

用法:
    python viz/orbit_demo.py [--uavs 100] [--seed 42] [--speed 100]
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import vtk  # noqa: E402

from config import parameters as params                        # noqa: E402
from config.parameters import EARTH_RADIUS                     # noqa: E402
from environment import Satellite, UAV                         # noqa: E402
from utils.geometry import cartesian_to_latlon, calculate_line_of_sight_distance  # noqa: E402
from viz.vtk_helpers import (                                  # noqa: E402
    create_earth_actor, create_satellite_actor, create_uav_actor,
    create_distance_line_actor, create_text_actor,
    create_starfield_background, create_distance_label,
    create_lighting, create_equator_actor, create_orbit_line_actor,
)

TEX_DIR = Path(__file__).resolve().parent / "textures"


class CustomInteractorStyle(vtk.vtkInteractorStyleTrackballCamera):
    """轨迹球交互 + 转发定时器事件给动画对象。"""

    def __init__(self):
        vtk.vtkInteractorStyleTrackballCamera.__init__(self)
        self.dragging = False
        self.animation = None
        self.AddObserver("LeftButtonPressEvent", self._on_press)
        self.AddObserver("LeftButtonReleaseEvent", self._on_release)
        self.AddObserver("MouseMoveEvent", self._on_move)
        self.AddObserver(vtk.vtkCommand.TimerEvent, self._on_timer)

    def _on_press(self, obj, event):
        self.dragging = True
        self.OnLeftButtonDown()

    def _on_release(self, obj, event):
        self.dragging = False
        self.OnLeftButtonUp()

    def _on_move(self, obj, event):
        if self.dragging:
            self.OnMouseMove()

    def _on_timer(self, obj, event):
        if self.animation:
            self.animation.update(obj, event)


class OrbitAnimation:
    """卫星轨道运动 + 最近 UAV 距离的实时 3D 可视化。"""

    def __init__(self, num_uavs=100, seed=42, time_acceleration=100):
        rng = np.random.default_rng(seed)
        self.time_acceleration = time_acceleration

        # ---- VTK 骨架 ----
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)
        self.render_window.SetSize(1200, 800)
        self.render_window.SetWindowName("UAV - LEO satellite orbit demo")

        # ---- 地球 / 赤道 / 星空 ----
        texture = TEX_DIR / "earth_texture.jpg"
        self.renderer.AddActor(create_earth_actor(
            radius=EARTH_RADIUS,
            texture_path=texture if texture.exists() else None))
        self.renderer.AddActor(create_equator_actor(
            radius=EARTH_RADIUS * 1.001, width=3.0, color=(0.35, 0.0, 0.0)))
        create_starfield_background(self.renderer, rng=rng)

        # ---- 卫星与轨道线 ----
        sp = params.SATELLITE_PARAMS
        self.satellite = Satellite(altitude=sp['altitude'],
                                   inclination=sp['inclination'],
                                   raan=sp['raan'])
        self.satellite_actor = create_satellite_actor(size=100000.0)
        self.satellite_actor.SetPosition(*self.satellite.get_position())
        self.renderer.AddActor(self.satellite_actor)
        self.renderer.AddActor(create_orbit_line_actor(self.satellite))

        # ---- UAV：沿轨道星下点轨迹分布 + 切向随机偏移 ----
        self.uavs = []
        inc, raan = np.radians(sp['inclination']), np.radians(sp['raan'])
        orbit_radius = EARTH_RADIUS + sp['altitude']
        max_offset = 100000.0  # 100 km 切向偏移
        for i in range(num_uavs):
            angle = 2 * np.pi * i / num_uavs
            x_orb = orbit_radius * np.cos(angle)
            y_orb = orbit_radius * np.sin(angle)
            # 绕 X 轴（倾角）再绕 Z 轴（升交点赤经）
            y1 = y_orb * np.cos(inc)
            z1 = y_orb * np.sin(inc)
            x_f = x_orb * np.cos(raan) - y1 * np.sin(raan)
            y_f = x_orb * np.sin(raan) + y1 * np.cos(raan)
            pos = np.array([x_f, y_f, z1])
            pos = pos / np.linalg.norm(pos) * EARTH_RADIUS

            # 切平面随机偏移
            normal = pos / np.linalg.norm(pos)
            t1 = np.array([-normal[1], normal[0], 0.0])
            if np.linalg.norm(t1) < 1e-6:
                t1 = np.array([1.0, 0.0, 0.0])
            t1 = t1 / np.linalg.norm(t1)
            t2 = np.cross(normal, t1)
            pos += t1 * rng.uniform(-max_offset, max_offset) \
                + t2 * rng.uniform(-max_offset, max_offset)

            lat, lon, _ = cartesian_to_latlon(*pos)
            uav = UAV(i, (lat, lon))
            self.uavs.append(uav)
            self.renderer.AddActor(create_uav_actor(uav.position, size=80000.0))

        # ---- 最近 3 架 UAV 的连线与标签 ----
        self.closest_lines, self.closest_labels = [], []

        # ---- 信息文本 / 光照 / 相机 ----
        self.info_text = create_text_actor(
            "UAV - LEO satellite real-time distance", position=(20, 20),
            font_size=20)
        self.renderer.AddActor2D(self.info_text)
        create_lighting(self.renderer)

        camera = self.renderer.GetActiveCamera()
        camera.SetPosition(0, -20000000, 10000000)
        camera.SetFocalPoint(0, 0, 0)
        camera.SetViewUp(0, 0, 1)
        camera.SetClippingRange(100000, 50000000)

        # ---- 定时器 ----
        self.last_update = time.time()
        self.simulation_time = 0.0
        self.interactor.Initialize()
        style = CustomInteractorStyle()
        style.animation = self
        self.interactor.SetInteractorStyle(style)
        self.interactor.CreateRepeatingTimer(100)

    def update(self, obj, event):
        """定时器回调：推进卫星、重画最近 3 架 UAV 的连线与标签。"""
        now = time.time()
        dt = now - self.last_update
        self.last_update = now
        self.simulation_time += dt * self.time_acceleration

        self.satellite.update_position(self.simulation_time)
        sat_pos = self.satellite.get_position()
        self.satellite_actor.SetPosition(*sat_pos)

        for actor in self.closest_lines + self.closest_labels:
            self.renderer.RemoveActor(actor)
        self.closest_lines, self.closest_labels = [], []

        distances = [(i, calculate_line_of_sight_distance(sat_pos, u.position))
                     for i, u in enumerate(self.uavs)]
        distances.sort(key=lambda x: x[1])
        for uav_idx, distance in distances[:3]:
            uav_pos = self.uavs[uav_idx].position
            self.closest_lines.append(create_distance_line_actor(
                sat_pos, uav_pos, color=(1.0, 1.0, 0.0), width=2.0))
            self.renderer.AddActor(self.closest_lines[-1])
            self.closest_labels.append(
                create_distance_label(sat_pos, uav_pos, distance, self.renderer))

        orbital = self.satellite.get_orbital_parameters()
        lines = "\n".join(
            f"UAV {i}: {d / 1000:.1f} km" for i, d in distances[:3])
        self.info_text.SetInput(
            f"Closest UAVs:\n{lines}\n"
            f"Altitude: {self.satellite.altitude / 1000:.0f} km | "
            f"Inclination: {np.degrees(orbital['inclination']):.0f} deg | "
            f"RAAN: {np.degrees(orbital['raan']):.0f} deg | "
            f"sim time: {self.simulation_time:.0f} s")

        self.render_window.Render()

    def start(self):
        self.render_window.Render()
        self.interactor.Start()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--uavs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=int, default=100,
                        help="time acceleration factor")
    args = parser.parse_args()

    animation = OrbitAnimation(num_uavs=args.uavs, seed=args.seed,
                               time_acceleration=args.speed)
    animation.start()


if __name__ == "__main__":
    main()
