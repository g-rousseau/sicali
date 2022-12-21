import enum
import math
import matplotlib.pyplot as plt
import numpy as np
import typing

from .coordinates import (
    CartesianCoordinate2D,
    CartesianCoordinate3D,
    SphericalCoordinate3D,
)
from .lens_model import (
    EquidistantModel,
    EquisolidModel,
    LensModelType,
    OrthographicModel,
    RectilinearModel,
    StereographicModel,
)


class FocalAdjustment(enum.Enum):
    """Automatic focal length adjustment.

    NONE: No adjustment, use input value
    INSCRIBE_FOV: Focal length such that the lens field of view is inscribed in the
        sensor frame
    CIRCUMSCRIBE_SENSOR: Focal length such that the lens field of view circumscribes the
        sensor frame
    """

    NONE = 0
    INSCRIBE_FOV = 1
    CIRCUMSCRIBE_SENSOR = 2


class CameraConfig(typing.NamedTuple):
    """Camera model configuration parameters.

    focal_adjustment: Focal length adjustment
    lens_fov: Lens angular field of view [rad]
    lens_focal: Lens focal length [m] (ignored if focal_adjustment is not NONE)
    lens_model: Lens projection model
    sensor_definition_x: Sensor definition (pixel count) on the x axis (width) [pixels]
    sensor_definition_y: Sensor definition (pixel count) on the y axis (height) [pixels]
    sensor_pixel_size: Sensor pixel size [m]
    """

    focal_adjustment: FocalAdjustment
    lens_fov: float
    lens_focal: float
    lens_model: LensModelType
    sensor_definition_x: int
    sensor_definition_y: int
    sensor_pixel_size: float


class CameraModel:
    """Simple camera projection model."""

    def __init__(self, config: CameraConfig) -> None:
        if config.lens_fov < 0.0:
            raise Exception("Lens field of view must be positive")
        if config.lens_focal < 0.0:
            raise Exception("Lens focal length must be positive")
        if config.sensor_definition_x < 0.0 or config.sensor_definition_y < 0.0:
            raise Exception("Sensor definition must be positive")
        if config.sensor_pixel_size < 0.0:
            raise Exception("Sensor pixel size must be positive")

        if config.lens_model is LensModelType.EQUIDISTANT:
            self._lens_model = EquidistantModel(config.lens_fov, config.lens_focal)
        elif config.lens_model is LensModelType.EQUISOLID:
            self._lens_model = EquisolidModel(config.lens_fov, config.lens_focal)
        elif config.lens_model is LensModelType.ORTHOGRAPHIC:
            self._lens_model = OrthographicModel(config.lens_fov, config.lens_focal)
        elif config.lens_model is LensModelType.RECTILINEAR:
            self._lens_model = RectilinearModel(config.lens_fov, config.lens_focal)
        elif config.lens_model is LensModelType.STEREOGRAPHIC:
            self._lens_model = StereographicModel(config.lens_fov, config.lens_focal)
        else:
            raise Exception("Undefined lens model")

        self._config = config
        self._sensor_size_x = config.sensor_definition_x * config.sensor_pixel_size
        self._sensor_size_y = config.sensor_definition_y * config.sensor_pixel_size

        if config.focal_adjustment is FocalAdjustment.INSCRIBE_FOV:
            self._lens_model.set_focal_to_inscribe_sensor_in_fov(
                self._sensor_size_x, self._sensor_size_y
            )
        elif config.focal_adjustment is FocalAdjustment.CIRCUMSCRIBE_SENSOR:
            self._lens_model.set_focal_to_circumscribe_sensor_in_fov(
                self._sensor_size_x, self._sensor_size_y
            )

    def display_lens_model(self, use_tex: bool = False) -> None:
        """Plot the lens projection model projected radius vs. incidence.

        Args:
            use_tex (optional): True to use tex font rendering. Defaults to False.
        """
        POINT_COUNT = 50
        incidence = np.linspace(0, self._lens_model.fov / 2, POINT_COUNT)
        radius_ideal = self._lens_model.focal * incidence
        radius_proj = np.zeros(POINT_COUNT)
        for i in range(POINT_COUNT):
            radius_proj[i] = self._lens_model._incidence_to_projection_radius(
                incidence[i]
            )

        if use_tex:
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        ax.plot(np.degrees(incidence), radius_ideal, "k:", label="equidistant")
        ax.plot(np.degrees(incidence), radius_proj, "b", label=self._lens_model.name)

        ax.grid()

        if use_tex:
            ax.set_title(
                r"\textbf{Lens projection model: radius $r$ vs. incidence $\theta$}"
            )
            ax.set_xlabel(r"$\theta$ (°)")
            ax.set_ylabel(r"$r$ (m)")
        else:
            ax.set_title(f"Lens projection model: radius vs. incidence")
            ax.set_xlabel(f"incidence (°)")
            ax.set_ylabel(f"radius (m)")

        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.legend()

        plt.show()

    def display_plane_projection(
        self,
        plane_size: float,
        plane_depth: float,
        line_count: float,
        use_tex: bool = False,
    ) -> None:
        """Plot the lens projection of a square plane grid.

        Args:
            plane_size: square plane side length [m].
            plane_depth: plane distance to camera [m].
            line_count: count of lines of grid to project.
            use_tex (optional): True to use tex font rendering. Defaults to False.
        """
        POINT_COUNT = 50
        line_steps = np.linspace(-plane_size / 2, plane_size / 2, line_count)
        line_points = np.linspace(-plane_size / 2, plane_size / 2, POINT_COUNT)

        if use_tex:
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")

        fig = plt.figure()
        ax_scene = fig.add_subplot(1, 2, 1, projection="3d")
        ax_proj = fig.add_subplot(1, 2, 2)

        # horizontal lines
        for i_line in range(line_count):
            line_x = np.full(POINT_COUNT, line_steps[i_line])
            line_y = line_points
            line_z = np.full(POINT_COUNT, plane_depth)

            proj_x = np.full(POINT_COUNT, math.nan)
            proj_y = np.full(POINT_COUNT, math.nan)
            points_in_fov_and_sensor = np.zeros((3, 0))
            for i_point in range(POINT_COUNT):
                point = CartesianCoordinate3D(
                    line_x[i_point], line_y[i_point], line_z[i_point]
                )
                direction = point.to_spherical().direction
                proj = self._lens_model.project(direction)

                proj_x[i_point] = proj.x
                proj_y[i_point] = proj.y

                if self._lens_model.is_incidence_out_of_fov(
                    direction.elevation
                ) or self._is_out_of_sensor(proj):
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[math.nan], [math.nan], [math.nan]]),
                        axis=1,
                    )
                else:
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[point.x], [point.y], [point.z]]),
                        axis=1,
                    )

            ax_scene.plot(
                line_x, line_y, line_z, color="cornflowerblue", linewidth=0.75
            )
            ax_proj.plot(proj_x, proj_y, color="blue")
            ax_scene.plot(
                points_in_fov_and_sensor[0],
                points_in_fov_and_sensor[1],
                points_in_fov_and_sensor[2],
                color="blue",
                linewidth=1.75,
            )

        # vertical lines
        for i_line in range(line_count):
            line_x = line_points
            line_y = np.full(POINT_COUNT, line_steps[i_line])
            line_z = np.full(POINT_COUNT, plane_depth)

            proj_x = np.full(POINT_COUNT, math.nan)
            proj_y = np.full(POINT_COUNT, math.nan)
            points_in_fov_and_sensor = np.zeros((3, 0))
            for i_point in range(POINT_COUNT):
                point = CartesianCoordinate3D(
                    line_x[i_point], line_y[i_point], line_z[i_point]
                )
                direction = point.to_spherical().direction
                proj = self._lens_model.project(direction)

                proj_x[i_point] = proj.x
                proj_y[i_point] = proj.y

                if self._lens_model.is_incidence_out_of_fov(
                    direction.elevation
                ) or self._is_out_of_sensor(proj):
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[math.nan], [math.nan], [math.nan]]),
                        axis=1,
                    )
                else:
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[point.x], [point.y], [point.z]]),
                        axis=1,
                    )

            ax_scene.plot(line_x, line_y, line_z, color="lightcoral", linewidth=0.75)
            ax_proj.plot(proj_x, proj_y, color="red")
            ax_scene.plot(
                points_in_fov_and_sensor[0],
                points_in_fov_and_sensor[1],
                points_in_fov_and_sensor[2],
                color="red",
                linewidth=1.75,
            )

        # format figure
        ax_scene.grid()

        if use_tex:
            fig.suptitle(r"\textbf{3D plane projection}")

            ax_scene.set_title(r"3D scene")
            ax_scene.set_xlabel(r"$x$ (m)")
            ax_scene.set_ylabel(r"$y$ (m)")
            ax_scene.set_zlabel(r"$z$ (m)")

            ax_proj.set_title(r"Scene projection onto sensor")
            ax_proj.set_xlabel(r"$x$ (m)")
            ax_proj.set_ylabel(r"$y$ (m)")
        else:
            fig.suptitle(f"3D plane projection")

            ax_scene.set_title(f"3D scene")
            ax_scene.set_xlabel(f"x (m)")
            ax_scene.set_ylabel(f"y (m)")
            ax_scene.set_zlabel(f"z (m)")

            ax_proj.set_title(f"Scene projection onto sensor")
            ax_proj.set_xlabel(f"x (m)")
            ax_proj.set_ylabel(f"y (m)")

        ax_scene.set_aspect("equal")
        ax_proj.set_aspect("equal")
        ax_proj.set_xlim(-self._sensor_size_x / 2, self._sensor_size_x / 2)
        ax_proj.set_ylim(-self._sensor_size_y / 2, self._sensor_size_y / 2)

        plt.show()

    def display_latitude_longitude_projection(
        self, line_count: float, use_tex: bool = False
    ) -> None:
        """Plot the lens projection of iso-longitude and iso-latitude curves.

        Args:
            line_count: count of lines of grid to project.
            use_tex (bool, optional): _description_. Defaults to False.
        """
        POINT_COUNT = 100
        SPHERE_RADIUS = 1

        if use_tex:
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")

        fig = plt.figure()
        ax_scene = fig.add_subplot(1, 2, 1, projection="3d")
        ax_proj = fig.add_subplot(1, 2, 2)

        # iso-latitude
        line_points = np.linspace(0, 2 * math.pi, POINT_COUNT)
        line_steps = np.linspace(0, math.pi, line_count + 2)
        line_steps = line_steps[1 : (line_count + 1)]

        for i_line in range(line_count):
            line_x = SPHERE_RADIUS * np.cos(line_points) * math.sin(line_steps[i_line])
            line_y = SPHERE_RADIUS * np.sin(line_points) * math.sin(line_steps[i_line])
            line_z = SPHERE_RADIUS * np.full(POINT_COUNT, math.cos(line_steps[i_line]))

            proj_x = np.full(POINT_COUNT, math.nan)
            proj_y = np.full(POINT_COUNT, math.nan)
            points_in_fov_and_sensor = np.zeros((3, 0))
            for i_point in range(POINT_COUNT):
                point = CartesianCoordinate3D(
                    line_x[i_point], line_y[i_point], line_z[i_point]
                )
                direction = point.to_spherical().direction
                proj = self._lens_model.project(direction)

                proj_x[i_point] = proj.x
                proj_y[i_point] = proj.y

                if self._lens_model.is_incidence_out_of_fov(
                    direction.elevation
                ) or self._is_out_of_sensor(proj):
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[math.nan], [math.nan], [math.nan]]),
                        axis=1,
                    )
                else:
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[point.x], [point.y], [point.z]]),
                        axis=1,
                    )

            ax_scene.plot(line_x, line_y, line_z, color="lightcoral", linewidth=0.75)
            ax_proj.plot(proj_x, proj_y, color="red")
            ax_scene.plot(
                points_in_fov_and_sensor[0],
                points_in_fov_and_sensor[1],
                points_in_fov_and_sensor[2],
                color="red",
                linewidth=1.75,
            )

        # iso-longitude
        line_points = np.linspace(0, math.pi, POINT_COUNT)
        line_steps = np.linspace(0, 2 * math.pi, line_count + 1)
        line_steps = line_steps[0:line_count]

        for i_line in range(line_count):
            line_x = SPHERE_RADIUS * np.sin(line_points) * math.cos(line_steps[i_line])
            line_y = SPHERE_RADIUS * np.sin(line_points) * math.sin(line_steps[i_line])
            line_z = SPHERE_RADIUS * np.cos(line_points)

            proj_x = np.full(POINT_COUNT, math.nan)
            proj_y = np.full(POINT_COUNT, math.nan)
            points_in_fov_and_sensor = np.zeros((3, 0))
            for i_point in range(POINT_COUNT):
                point = CartesianCoordinate3D(
                    line_x[i_point], line_y[i_point], line_z[i_point]
                )
                direction = point.to_spherical().direction
                proj = self._lens_model.project(direction)

                proj_x[i_point] = proj.x
                proj_y[i_point] = proj.y

                if self._lens_model.is_incidence_out_of_fov(
                    direction.elevation
                ) or self._is_out_of_sensor(proj):
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[math.nan], [math.nan], [math.nan]]),
                        axis=1,
                    )
                else:
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[point.x], [point.y], [point.z]]),
                        axis=1,
                    )

            ax_scene.plot(
                line_x, line_y, line_z, color="cornflowerblue", linewidth=0.75
            )
            ax_proj.plot(proj_x, proj_y, color="blue")
            ax_scene.plot(
                points_in_fov_and_sensor[0],
                points_in_fov_and_sensor[1],
                points_in_fov_and_sensor[2],
                color="blue",
                linewidth=1.75,
            )

        # format figure
        ax_scene.grid()

        if use_tex:
            fig.suptitle(r"\textbf{Iso latitude/longitude projection}")

            ax_scene.set_title(r"3D scene")
            ax_scene.set_xlabel(r"$x$ (m)")
            ax_scene.set_ylabel(r"$y$ (m)")
            ax_scene.set_zlabel(r"$z$ (m)")

            ax_proj.set_title(r"Scene projection onto sensor")
            ax_proj.set_xlabel(r"$x$ (m)")
            ax_proj.set_ylabel(r"$y$ (m)")
        else:
            fig.suptitle(f"Iso latitude/longitude projection")

            ax_scene.set_title(f"3D scene")
            ax_scene.set_xlabel(f"x (m)")
            ax_scene.set_ylabel(f"y (m)")
            ax_scene.set_zlabel(f"z (m)")

            ax_proj.set_title(f"Scene projection onto sensor")
            ax_proj.set_xlabel(f"x (m)")
            ax_proj.set_ylabel(f"y (m)")

        ax_scene.set_aspect("equal")
        ax_proj.set_aspect("equal")
        ax_proj.set_xlim(-self._sensor_size_x / 2, self._sensor_size_x / 2)
        ax_proj.set_ylim(-self._sensor_size_y / 2, self._sensor_size_y / 2)

        plt.show()

    def display_sphere_slices_projection(
        self, line_count: float, use_tex: bool = False
    ) -> None:
        """Plot the lens projection of sphere slices.

        Args:
            line_count: count of lines of grid to project.
            use_tex (bool, optional): _description_. Defaults to False.
        """
        POINT_COUNT = 100
        SPHERE_RADIUS = 1
        line_points = np.linspace(0, 2 * math.pi, POINT_COUNT)
        line_steps = np.linspace(0, math.pi, line_count + 2)
        line_steps = line_steps[1 : (line_count + 1)]

        if use_tex:
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")

        fig = plt.figure()
        ax_scene = fig.add_subplot(1, 2, 1, projection="3d")
        ax_proj = fig.add_subplot(1, 2, 2)

        # x slices
        for i_line in range(line_count):
            line_x = SPHERE_RADIUS * np.full(POINT_COUNT, math.cos(line_steps[i_line]))
            line_y = SPHERE_RADIUS * np.cos(line_points) * math.sin(line_steps[i_line])
            line_z = SPHERE_RADIUS * np.sin(line_points) * math.sin(line_steps[i_line])

            proj_x = np.full(POINT_COUNT, math.nan)
            proj_y = np.full(POINT_COUNT, math.nan)
            points_in_fov_and_sensor = np.zeros((3, 0))
            for i_point in range(POINT_COUNT):
                point = CartesianCoordinate3D(
                    line_x[i_point], line_y[i_point], line_z[i_point]
                )
                direction = point.to_spherical().direction
                proj = self._lens_model.project(direction)

                proj_x[i_point] = proj.x
                proj_y[i_point] = proj.y

                if self._lens_model.is_incidence_out_of_fov(
                    direction.elevation
                ) or self._is_out_of_sensor(proj):
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[math.nan], [math.nan], [math.nan]]),
                        axis=1,
                    )
                else:
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[point.x], [point.y], [point.z]]),
                        axis=1,
                    )

            ax_scene.plot(
                line_x, line_y, line_z, color="cornflowerblue", linewidth=0.75
            )
            ax_proj.plot(proj_x, proj_y, color="blue")
            ax_scene.plot(
                points_in_fov_and_sensor[0],
                points_in_fov_and_sensor[1],
                points_in_fov_and_sensor[2],
                color="blue",
                linewidth=1.75,
            )

        # y-slices
        for i_line in range(line_count):
            line_x = SPHERE_RADIUS * np.sin(line_points) * math.sin(line_steps[i_line])
            line_y = SPHERE_RADIUS * np.full(POINT_COUNT, math.cos(line_steps[i_line]))
            line_z = SPHERE_RADIUS * np.cos(line_points) * math.sin(line_steps[i_line])

            proj_x = np.full(POINT_COUNT, math.nan)
            proj_y = np.full(POINT_COUNT, math.nan)
            points_in_fov_and_sensor = np.zeros((3, 0))
            for i_point in range(POINT_COUNT):
                point = CartesianCoordinate3D(
                    line_x[i_point], line_y[i_point], line_z[i_point]
                )
                direction = point.to_spherical().direction
                proj = self._lens_model.project(direction)

                proj_x[i_point] = proj.x
                proj_y[i_point] = proj.y

                if self._lens_model.is_incidence_out_of_fov(
                    direction.elevation
                ) or self._is_out_of_sensor(proj):
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[math.nan], [math.nan], [math.nan]]),
                        axis=1,
                    )
                else:
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[point.x], [point.y], [point.z]]),
                        axis=1,
                    )

            ax_scene.plot(line_x, line_y, line_z, color="lightgreen", linewidth=0.75)
            ax_proj.plot(proj_x, proj_y, color="green")
            ax_scene.plot(
                points_in_fov_and_sensor[0],
                points_in_fov_and_sensor[1],
                points_in_fov_and_sensor[2],
                color="green",
                linewidth=1.75,
            )

        # z-slices
        for i_line in range(line_count):
            line_x = SPHERE_RADIUS * np.cos(line_points) * math.sin(line_steps[i_line])
            line_y = SPHERE_RADIUS * np.sin(line_points) * math.sin(line_steps[i_line])
            line_z = SPHERE_RADIUS * np.full(POINT_COUNT, math.cos(line_steps[i_line]))

            proj_x = np.full(POINT_COUNT, math.nan)
            proj_y = np.full(POINT_COUNT, math.nan)
            points_in_fov_and_sensor = np.zeros((3, 0))
            for i_point in range(POINT_COUNT):
                point = CartesianCoordinate3D(
                    line_x[i_point], line_y[i_point], line_z[i_point]
                )
                direction = point.to_spherical().direction
                proj = self._lens_model.project(direction)

                proj_x[i_point] = proj.x
                proj_y[i_point] = proj.y

                if self._lens_model.is_incidence_out_of_fov(
                    direction.elevation
                ) or self._is_out_of_sensor(proj):
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[math.nan], [math.nan], [math.nan]]),
                        axis=1,
                    )
                else:
                    points_in_fov_and_sensor = np.append(
                        points_in_fov_and_sensor,
                        np.array([[point.x], [point.y], [point.z]]),
                        axis=1,
                    )

            ax_scene.plot(line_x, line_y, line_z, color="lightcoral", linewidth=0.75)
            ax_proj.plot(proj_x, proj_y, color="red")
            ax_scene.plot(
                points_in_fov_and_sensor[0],
                points_in_fov_and_sensor[1],
                points_in_fov_and_sensor[2],
                color="red",
                linewidth=1.75,
            )

        # format figure
        ax_scene.grid()

        if use_tex:
            fig.suptitle(r"\textbf{Sphere slices projection}")

            ax_scene.set_title(r"3D scene")
            ax_scene.set_xlabel(r"$x$ (m)")
            ax_scene.set_ylabel(r"$y$ (m)")
            ax_scene.set_zlabel(r"$z$ (m)")

            ax_proj.set_title(r"Scene projection onto sensor")
            ax_proj.set_xlabel(r"$x$ (m)")
            ax_proj.set_ylabel(r"$y$ (m)")
        else:
            fig.suptitle(f"Sphere slices projection")

            ax_scene.set_title(f"3D scene")
            ax_scene.set_xlabel(f"x (m)")
            ax_scene.set_ylabel(f"y (m)")
            ax_scene.set_zlabel(f"z (m)")

            ax_proj.set_title(f"Scene projection onto sensor")
            ax_proj.set_xlabel(f"x (m)")
            ax_proj.set_ylabel(f"y (m)")

        ax_scene.set_aspect("equal")
        ax_proj.set_aspect("equal")
        ax_proj.set_xlim(-self._sensor_size_x / 2, self._sensor_size_x / 2)
        ax_proj.set_ylim(-self._sensor_size_y / 2, self._sensor_size_y / 2)

        plt.show()

    def display_pixels_propagation(self, use_tex: bool = False) -> None:
        """Invert projection of the sensor grid onto a sphere.

        Args:
            use_tex (bool, optional): _description_. Defaults to False.
        """
        POINT_COUNT = 100
        SPHERE_RADIUS = 1

        if use_tex:
            plt.rc("text", usetex=True)
            plt.rc("font", family="serif")

        fig = plt.figure()
        ax_sensor = fig.add_subplot(1, 2, 1)
        ax_prop = fig.add_subplot(1, 2, 2, projection="3d")

        # horizontal lines
        line_points = np.linspace(
            -self._sensor_size_x / 2, self._sensor_size_x / 2, POINT_COUNT
        )
        line_steps = np.linspace(
            -self._sensor_size_y / 2,
            self._sensor_size_y / 2,
            self._config.sensor_definition_y + 1,
        )
        for i_line in range(self._config.sensor_definition_y + 1):
            line_x = line_points
            line_y = np.full(POINT_COUNT, line_steps[i_line])

            prop_x = np.full(POINT_COUNT, math.nan)
            prop_y = np.full(POINT_COUNT, math.nan)
            prop_z = np.full(POINT_COUNT, math.nan)
            for i_point in range(POINT_COUNT):
                proj = CartesianCoordinate2D(line_x[i_point], line_y[i_point])
                direction = self._lens_model.propagate(proj)
                point = SphericalCoordinate3D(SPHERE_RADIUS, direction).to_cartesian()

                prop_x[i_point] = point.x
                prop_y[i_point] = point.y
                prop_z[i_point] = point.z

            ax_sensor.plot(line_x, line_y, color="blue")
            ax_prop.plot(prop_x, prop_y, prop_z, color="blue")

        # vertical lines
        line_points = np.linspace(
            -self._sensor_size_y / 2, self._sensor_size_y / 2, POINT_COUNT
        )
        line_steps = np.linspace(
            -self._sensor_size_x / 2,
            self._sensor_size_x / 2,
            self._config.sensor_definition_x + 1,
        )
        for i_line in range(self._config.sensor_definition_y + 1):
            line_x = np.full(POINT_COUNT, line_steps[i_line])
            line_y = line_points

            prop_x = np.full(POINT_COUNT, math.nan)
            prop_y = np.full(POINT_COUNT, math.nan)
            prop_z = np.full(POINT_COUNT, math.nan)
            for i_point in range(POINT_COUNT):
                proj = CartesianCoordinate2D(line_x[i_point], line_y[i_point])
                direction = self._lens_model.propagate(proj)
                point = SphericalCoordinate3D(SPHERE_RADIUS, direction).to_cartesian()

                prop_x[i_point] = point.x
                prop_y[i_point] = point.y
                prop_z[i_point] = point.z

            ax_sensor.plot(line_x, line_y, color="red")
            ax_prop.plot(prop_x, prop_y, prop_z, color="red")

        # sphere
        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x = SPHERE_RADIUS * np.outer(np.cos(u), np.sin(v))
        y = SPHERE_RADIUS * np.outer(np.sin(u), np.sin(v))
        z = SPHERE_RADIUS * np.outer(np.ones(np.size(u)), np.cos(v))
        ax_prop.plot_surface(x, y, z, color="lightgrey", alpha=0.2)

        # format figure
        ax_prop.grid()

        if use_tex:
            fig.suptitle(r"\textbf{Sensor grid retro-projection}")

            ax_sensor.set_title(r"Sensor pixel grid")
            ax_sensor.set_xlabel(r"$x$ (m)")
            ax_sensor.set_ylabel(r"$y$ (m)")

            ax_prop.set_title(r"Pixel grid retro-projection onto a sphere")
            ax_prop.set_xlabel(r"$x$ (m)")
            ax_prop.set_ylabel(r"$y$ (m)")
            ax_prop.set_zlabel(r"$z$ (m)")
        else:
            fig.suptitle(f"Sensor grid retro-projection")

            ax_sensor.set_title(f"Sensor pixel grid")
            ax_sensor.set_xlabel(f"x (m)")
            ax_sensor.set_ylabel(f"y (m)")

            ax_prop.set_title(f"Pixel grid retro-projection onto a sphere")
            ax_prop.set_xlabel(f"x (m)")
            ax_prop.set_ylabel(f"y (m)")
            ax_prop.set_zlabel(f"z (m)")

        ax_sensor.set_aspect("equal")
        ax_sensor.set_xlim(-self._sensor_size_x / 2, self._sensor_size_x / 2)
        ax_sensor.set_ylim(-self._sensor_size_y / 2, self._sensor_size_y / 2)
        ax_prop.set_aspect("equal")

        plt.show()

    def _is_out_of_sensor(self, coord: CartesianCoordinate2D) -> bool:
        """True if the input 2D coordinate is not within the sensor bounds."""
        return (abs(coord.x) > self._sensor_size_x / 2) or (
            abs(coord.y) > self._sensor_size_y / 2
        )
