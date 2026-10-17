# viz/vtk_helpers.py
"""VTK 3D 可视化辅助函数（可选组件，需要 `pip install .[viz3d]`）。

自原项目 utils/visualization.py 迁移，裁剪与修改：
- 只保留轨道演示实际用到的 actor 构建函数；
- 删除中文字体文件探测逻辑（标签全部使用英文 + 默认字体，
  彻底摆脱对系统字体路径的依赖）；
- 删除从未被调用的 create_beam_label / create_beam_boundary_actor；
- 新增 create_orbit_line_actor（原逻辑在 Satellite.create_orbit_actor
  中，为保持核心包零 VTK 依赖而移到此处）。
"""

import numpy as np
import vtk

from config.parameters import EARTH_RADIUS
from utils.geometry import theta_sat, phi_sat, spherical_to_cartesian


def create_earth_actor(radius=EARTH_RADIUS, texture_path=None):
    """带纹理的地球球体（无纹理时回退为蓝色材质）。"""
    source = vtk.vtkSphereSource()
    source.SetRadius(radius)
    source.SetThetaResolution(250)
    source.SetPhiResolution(200)

    tex_coords = vtk.vtkTextureMapToSphere()
    tex_coords.SetInputConnection(source.GetOutputPort())
    tex_coords.PreventSeamOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(tex_coords.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetAmbient(0.3)
    actor.GetProperty().SetDiffuse(1.0)
    actor.GetProperty().SetSpecular(0.8)
    actor.GetProperty().SetSpecularPower(30)

    if texture_path:
        reader = vtk.vtkJPEGReader()
        reader.SetFileName(str(texture_path))
        texture = vtk.vtkTexture()
        texture.SetInputConnection(reader.GetOutputPort())
        texture.InterpolateOn()
        actor.SetTexture(texture)
    else:
        actor.GetProperty().SetColor(0.2, 0.3, 0.8)

    return actor


def create_satellite_actor(size=100000.0):
    """卫星模型：立方体主体 + 两侧太阳能板。"""
    cube = vtk.vtkCubeSource()
    cube.SetXLength(size)
    cube.SetYLength(size / 2)
    cube.SetZLength(size / 2)

    panel1 = vtk.vtkCubeSource()
    panel1.SetXLength(size * 2)
    panel1.SetYLength(size / 10)
    panel1.SetZLength(size / 2)
    panel1.SetCenter(size * 1.5, 0, 0)

    panel2 = vtk.vtkCubeSource()
    panel2.SetXLength(size * 2)
    panel2.SetYLength(size / 10)
    panel2.SetZLength(size / 2)
    panel2.SetCenter(-size * 1.5, 0, 0)

    append = vtk.vtkAppendPolyData()
    append.AddInputConnection(cube.GetOutputPort())
    append.AddInputConnection(panel1.GetOutputPort())
    append.AddInputConnection(panel2.GetOutputPort())

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(append.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(1.0, 0.0, 0.0)
    actor.GetProperty().SetAmbient(0.5)
    return actor


def create_uav_actor(position, size=80000.0):
    """UAV 模型：绿色三角锥。"""
    pyramid = vtk.vtkConeSource()
    pyramid.SetResolution(3)
    pyramid.SetHeight(size * 2)
    pyramid.SetRadius(size)
    pyramid.SetDirection(1, 0, 0)

    transform = vtk.vtkTransform()
    transform.Translate(position[0], position[1], position[2])
    transform_filter = vtk.vtkTransformPolyDataFilter()
    transform_filter.SetInputConnection(pyramid.GetOutputPort())
    transform_filter.SetTransform(transform)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(transform_filter.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(0.0, 1.0, 0.0)
    prop.SetAmbient(0.5)
    prop.SetDiffuse(0.7)
    prop.SetSpecular(0.3)
    return actor


def create_distance_line_actor(start, end, color=(1.0, 1.0, 0.0), width=2.0):
    """两点连线。"""
    line = vtk.vtkLineSource()
    line.SetPoint1(start)
    line.SetPoint2(end)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(line.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)
    return actor


def create_trajectory_actor(points, color=(0.5, 0.5, 1.0), width=1.5):
    """多点折线（用于轨道线）。"""
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(point)

    poly_line = vtk.vtkPolyLine()
    poly_line.GetPointIds().SetNumberOfIds(len(points))
    for i in range(len(points)):
        poly_line.GetPointIds().SetId(i, i)

    cells = vtk.vtkCellArray()
    cells.InsertNextCell(poly_line)

    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    poly_data.SetLines(cells)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(poly_data)

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(color)
    actor.GetProperty().SetLineWidth(width)
    return actor


def create_orbit_line_actor(satellite, num_points=100):
    """根据卫星轨道参数生成轨道线 actor（原 Satellite.create_orbit_actor）。"""
    orbit_radius = satellite.semi_major_axis
    points = []
    for i in range(num_points):
        nu = 2 * np.pi * i / (num_points - 1)
        theta = theta_sat(satellite.inclination_rad, nu)
        phi = phi_sat(satellite.raan_rad, satellite.inclination_rad, nu)
        points.append(spherical_to_cartesian(orbit_radius, theta, phi))
    return create_trajectory_actor(points, color=(0.5, 0.5, 1.0), width=1.5)


def create_text_actor(text, position=(20, 20), font_size=20, color=(1, 1, 0.8)):
    """2D 信息文本（英文，使用默认字体）。"""
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput(text)
    prop = text_actor.GetTextProperty()
    prop.SetFontSize(font_size)
    prop.SetColor(color)
    prop.SetBold(True)
    text_actor.SetPosition(position)
    return text_actor


def create_starfield_background(renderer, num_stars=2000, rng=None):
    """渐变深空背景 + 随机星点（程序化生成，无需纹理图片）。"""
    renderer.GradientBackgroundOn()
    renderer.SetBackground(0.05, 0.05, 0.15)
    renderer.SetBackground2(0.0, 0.0, 0.0)

    rng = rng or np.random.default_rng()
    points = vtk.vtkPoints()
    theta = rng.uniform(0, 2 * np.pi, num_stars)
    phi = rng.uniform(0, np.pi, num_stars)
    r = 1e9
    for i in range(num_stars):
        points.InsertNextPoint(
            r * np.sin(phi[i]) * np.cos(theta[i]),
            r * np.sin(phi[i]) * np.sin(theta[i]),
            r * np.cos(phi[i]))

    stars = vtk.vtkPolyData()
    stars.SetPoints(points)

    vertices = vtk.vtkCellArray()
    for i in range(num_stars):
        vertices.InsertNextCell(1)
        vertices.InsertCellPoint(i)
    stars.SetVerts(vertices)

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(stars)

    star_actor = vtk.vtkActor()
    star_actor.SetMapper(mapper)
    star_actor.GetProperty().SetColor(1, 1, 1)
    star_actor.GetProperty().SetPointSize(2)
    renderer.AddActor(star_actor)


def create_distance_label(start, end, distance, renderer):
    """跟随相机的中点距离标签（"xxx km"）。"""
    mid = ((start[0] + end[0]) / 2,
           (start[1] + end[1]) / 2,
           (start[2] + end[2]) / 2)

    text_source = vtk.vtkVectorText()
    text_source.SetText(f"{distance / 1000:.1f} km")

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(text_source.GetOutputPort())

    follower = vtk.vtkFollower()
    follower.SetMapper(mapper)
    follower.SetPosition(mid)
    follower.SetScale(300)
    prop = follower.GetProperty()
    prop.SetColor(1, 1, 0)
    prop.SetAmbient(1.0)
    prop.SetDiffuse(0.0)

    renderer.AddActor(follower)
    follower.SetCamera(renderer.GetActiveCamera())
    return follower


def create_lighting(renderer):
    """四光源照明：太阳主光 + 头灯环境光 + 补光 + 背光。"""
    sun = vtk.vtkLight()
    sun.SetLightTypeToSceneLight()
    sun.SetPosition(1.5e7, 0, 0)
    sun.SetFocalPoint(0, 0, 0)
    sun.SetColor(1.0, 1.0, 0.9)
    sun.SetIntensity(1.5)
    renderer.AddLight(sun)

    ambient = vtk.vtkLight()
    ambient.SetLightTypeToHeadlight()
    ambient.SetColor(0.8, 0.8, 1)
    ambient.SetIntensity(0.8)
    renderer.AddLight(ambient)

    fill = vtk.vtkLight()
    fill.SetLightTypeToSceneLight()
    fill.SetPosition(-1.5e7, 0, 0)
    fill.SetFocalPoint(0, 0, 0)
    fill.SetColor(0.8, 0.8, 1.0)
    fill.SetIntensity(0.5)
    renderer.AddLight(fill)

    back = vtk.vtkLight()
    back.SetLightTypeToSceneLight()
    back.SetPosition(0, 0, 1.5e7)
    back.SetFocalPoint(0, 0, 0)
    back.SetColor(0.9, 0.9, 1.0)
    back.SetIntensity(0.3)
    renderer.AddLight(back)


def create_equator_actor(radius=EARTH_RADIUS, width=3.0, color=(0.35, 0.0, 0.0)):
    """赤道环。"""
    source = vtk.vtkRegularPolygonSource()
    source.SetNumberOfSides(200)
    source.SetRadius(radius)
    source.SetCenter(0, 0, 0)
    source.SetNormal(0, 0, 1)
    source.GeneratePolygonOn()

    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(source.GetOutputPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    prop = actor.GetProperty()
    prop.SetColor(color)
    prop.SetLineWidth(width)
    prop.SetAmbient(1.0)
    prop.SetDiffuse(0.0)
    return actor
