import abc
import enum
import math
import warnings

from .coordinates import CartesianCoordinate2D, SphericalDirection3D
from .math_utils import wrap_to_pi

# LENS MODEL INTERFACE
########################################################################################


class LensModelType(enum.Enum):
    """Lens model type.

    EQUIDISTANT: r = f * theta
    EQUISOLID: r = 2 * f * sin(theta / 2)
    ORTHOGRAPHIC: r = f * sin(theta)
    RECTILINEAR: r = f tan(theta)
    STEREOGRAPHIC: r = 2 * f * tan(theta / 2)
    """

    EQUIDISTANT = 0
    EQUISOLID = 1
    ORTHOGRAPHIC = 2
    RECTILINEAR = 3
    STEREOGRAPHIC = 4


class LensModel(abc.ABC):
    """Simple lens model."""

    def __init__(self, fov: float, focal: float) -> None:
        """Constructor (abstract)

        NOTE: The lens field of view might be adjusted if it exceeds the maximum
        field of view admissible by the lens projection model.

        Args:
            fov: lens angular field of view [rad] (field of view considered conical)
            focal: focal length [m]
        """
        self._fov = fov
        self._focal = focal

        if self._fov > self._max_fov:
            warnings.warn(
                f"Lens field of view exceeding maximum admissible value "
                f"field of view reduced to the latter "
                f"({math.degrees(self._max_fov)}°)"
            )
            self._fov = self._max_fov
            self._fov_projection_radius = self._incidence_to_projection_radius(
                self._fov / 2
            )

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Lens model name."""
        pass

    @property
    def fov(self) -> float:
        """Lens field of view [rad]."""
        return self._fov

    @property
    def focal(self) -> float:
        """Lens focal length [m]."""
        return self._focal

    @property
    def fov_projection_radius(self) -> float:
        """Radius of the projection of the lens field of view on the sensor [m]."""
        return self._fov_projection_radius

    def is_incidence_out_of_fov(self, incidence: float) -> bool:
        """True if the input incidence is not in the field of view of the lens.

        Args:
            incidence: incidence [rad]

        Returns:
            True if out of FOV
        """
        return incidence > self._fov / 2

    def is_projection_radius_out_of_fov(self, radius: float) -> bool:
        """True if the input projection radius is unreachable from the lens FOV.

        Args:
            radius: projection radius [m]

        Returns:
            True if out of FOV
        """
        return radius > self._fov_projection_radius

    def incidence_to_projection_radius(self, incidence: float) -> float:
        """Lens projection model, convert an incidence to a projection radius.

        Args:
            incidence: incidence (elevation) of the direction to project [rad]

        Returns:
            projection radius on the sensor [m]
        """
        if self.is_incidence_out_of_fov(incidence):
            return math.nan
        else:
            return self._incidence_to_projection_radius(incidence)

    def projection_radius_to_incidence(self, radius: float) -> float:
        """Lens invert projection model, convert a projection radius to an incidence.

        Args:
            radius: projection radius on the sensor [m]

        Returns:
            incidence (elevation) associated to the input projection radius [rad]
        """
        if self.is_projection_radius_out_of_fov(radius):
            return math.nan
        else:
            return self._projection_radius_to_incidence(radius)

    def project(self, direction: SphericalDirection3D) -> CartesianCoordinate2D:
        """Lens projection of the input 3D direction on the sensor.

        The z-axis is considered as the optical, i.e. the incidence is the elevation.

        Args:
            direction: 3D direction to project [rad]

        Returns:
            2D position of the projection in the sensor frame [m]
        """
        if direction.elevation > self.fov / 2:
            return CartesianCoordinate2D(math.nan, math.nan)
        else:
            azimuth_proj = wrap_to_pi(direction.azimuth + math.pi)
            radius_proj = self._incidence_to_projection_radius(direction.elevation)

            x_proj = radius_proj * math.cos(azimuth_proj)
            y_proj = radius_proj * math.sin(azimuth_proj)

            return CartesianCoordinate2D(x_proj, y_proj)

    def propagate(self, coord_sensor: CartesianCoordinate2D) -> SphericalDirection3D:
        """3D direction projected by the lens on the input coordinates on the sensor.

        The z-axis is considered as the optical, i.e. the incidence is the elevation.

        Args:
            coord_sensor: 2D position of the point to propagate in the sensor frame [m]
                (with (0,0) being the center of the sensor frame)

        Returns:
            3D direction associated to the input position [rad]
        """
        radius_proj = coord_sensor.norm()

        if self.is_projection_radius_out_of_fov(radius_proj):
            return SphericalDirection3D(math.nan, math.nan)
        else:
            azimuth = wrap_to_pi(math.atan2(coord_sensor.y, coord_sensor.x) + math.pi)
            elevation = self._projection_radius_to_incidence(radius_proj)
            return SphericalDirection3D(azimuth, elevation)

    def set_focal_to_inscribe_sensor_in_fov(
        self, sensor_size_x: float, sensor_size_y: float
    ) -> None:
        """Inscribe the lens' FOV in the sensor.

        Adjust the focal length such that the disk obtained by projecting the whole
        field of view of the lens exactly fit inside the sensor.

        Args:
            sensor_size_x: sensor width [m]
            sensor_size_y: sensor height [m]
        """
        original_focal = self._focal

        # radius of the projection of the edge of the FOV with a unit focal
        self._focal = 1
        fov_edge_projection_radius = self._incidence_to_projection_radius(self._fov / 2)

        # adjust focal to fit the sensor
        sensor_radius = min(sensor_size_x, sensor_size_y) / 2
        self._focal = sensor_radius / fov_edge_projection_radius
        self._fov_projection_radius = self._incidence_to_projection_radius(
            self._fov / 2
        )
        print(f"Focal length changed from {original_focal} m to {self._focal} m")

    def set_focal_to_circumscribe_sensor_in_fov(
        self, sensor_size_x: float, sensor_size_y: float
    ) -> None:
        """Circumscribe the sensor in the lens' FOV.

        Adjust the focal length such that the sensor exactly fit inside the disk
        obtained by projecting the whole field of view of the lens.

        Args:
            sensor_size_x: sensor width [m]
            sensor_size_y: sensor height [m]
        """
        original_focal = self._focal

        # radius of the projection of the edge of the FOV with a unit focal
        self._focal = 1
        fov_edge_projection_radius = self._incidence_to_projection_radius(self._fov / 2)

        # adjust focal to fit the sensor
        sensor_radius = math.sqrt(sensor_size_x**2 + sensor_size_y**2) / 2
        self._focal = sensor_radius / fov_edge_projection_radius
        self._fov_projection_radius = self._incidence_to_projection_radius(
            self._fov / 2
        )
        print(f"Focal length changed from {original_focal} m to {self._focal} m")

    @property
    @abc.abstractmethod
    def _max_fov(self) -> float:
        """Maximum field of view admissible for this lens model [rad]"""
        pass

    @abc.abstractmethod
    def _incidence_to_projection_radius(self, incidence: float) -> float:
        """Implementation of the lens projection model.

        Args:
            incidence: incidence (elevation) of the direction to project [rad]

        Returns:
            projection radius on the sensor [m]
        """
        pass

    @abc.abstractmethod
    def _projection_radius_to_incidence(self, radius: float) -> float:
        """Implementation of the lens invert projection model.

        Args:
            radius: projection radius on the sensor [m]

        Returns:
            incidence (elevation) associated to the input projection radius [rad]
        """
        pass


# LENS MODELS
########################################################################################


class EquidistantModel(LensModel):
    """Equidistant lens model with no distorsion (r = f theta)."""

    _MAX_FOV = math.inf
    _NAME = "equidistant"

    def __init__(self, fov, focal) -> None:
        super().__init__(fov, focal)

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def _max_fov(self) -> float:
        return self._MAX_FOV

    def _incidence_to_projection_radius(self, incidence: float) -> float:
        return self.focal * incidence

    def _projection_radius_to_incidence(self, radius: float) -> float:
        return radius / self.focal


class StereographicModel(LensModel):
    """Stereographic lens model with no distorsion (r = 2 f tan(theta / 2))"""

    _MAX_FOV = 2 * math.pi - math.radians(1.0)
    _NAME = "stereographic"

    def __init__(self, fov, focal) -> None:
        super().__init__(fov, focal)

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def _max_fov(self) -> float:
        return self._MAX_FOV

    def _incidence_to_projection_radius(self, incidence: float) -> float:
        return 2 * self.focal * math.tan(incidence / 2)

    def _projection_radius_to_incidence(self, radius: float) -> float:
        return 2 * math.atan(radius / (2 * self.focal))


class EquisolidModel(LensModel):
    """Equisolid lens model with no distorsion (r = 2 f sin(theta / 2))"""

    _MAX_FOV = 2 * math.pi
    _NAME = "equisolid"

    def __init__(self, fov, focal) -> None:
        super().__init__(fov, focal)

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def _max_fov(self) -> float:
        return self._MAX_FOV

    def _incidence_to_projection_radius(self, incidence: float) -> float:
        return 2 * self.focal * math.sin(incidence / 2)

    def _projection_radius_to_incidence(self, radius: float) -> float:
        return 2 * math.asin(radius / (2 * self.focal))


class OrthographicModel(LensModel):
    """Orthographic lens model with no distorsion (r = f sin(theta))"""

    _MAX_FOV = math.pi
    _NAME = "orthographic"

    def __init__(self, fov, focal) -> None:
        super().__init__(fov, focal)

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def _max_fov(self) -> float:
        return self._MAX_FOV

    def _incidence_to_projection_radius(self, incidence: float) -> float:
        return self.focal * math.sin(incidence)

    def _projection_radius_to_incidence(self, radius: float) -> float:
        return math.asin(radius / self.focal)


class RectilinearModel(LensModel):
    """Rectilinear lens model with no distorsion (r = f tan(theta))"""

    _MAX_FOV = math.pi - math.radians(1.0)
    _NAME = "rectilinear"

    def __init__(self, fov, focal) -> None:
        super().__init__(fov, focal)

    @property
    def name(self) -> str:
        return self._NAME

    @property
    def _max_fov(self) -> float:
        return self._MAX_FOV

    def _incidence_to_projection_radius(self, incidence: float) -> float:
        return self.focal * math.tan(incidence)

    def _projection_radius_to_incidence(self, radius: float) -> float:
        return math.atan(radius / self.focal)
