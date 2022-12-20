import math


class SphericalDirection3D:
    """3D spherical direction.

    Define a direction in the 3D space, from the origin, via
    - an azimuth: angle from the x axis to the projection of the 3D
    direction on the xy plan
    - an elevation: angle from the z axis to the 3D direction

    The x, y and z axes thus have the following directions
    - x axis: azimuth = 0, elevation = pi/2 [rad]
    - y axis: azimuth = pi/2, elevation = pi/2 [rad]
    - z axis: azimuth = 0 (undefined), elevation = pi/2 [rad]
    """

    def __init__(self, azimuth: float, elevation: float) -> None:
        self.azimuth = azimuth
        self.elevation = elevation


class SphericalCoordinate3D:
    """3D spherical coordinates.

    Define a 3D position via
    - a radius, distance to the origin
    - a 3D spherical direction
    """

    def __init__(self, radius: float, direction: SphericalDirection3D) -> None:
        self.direction = direction
        self.radius = radius

    def to_cartesian(self):
        x = (
            self.radius
            * math.cos(self.direction.azimuth)
            * math.sin(self.direction.elevation)
        )
        y = (
            self.radius
            * math.sin(self.direction.azimuth)
            * math.sin(self.direction.elevation)
        )
        z = self.radius * math.cos(self.direction.elevation)

        return CartesianCoordinate3D(x, y, z)


class CartesianCoordinate3D:
    """3D cartesian coordinates."""

    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def norm(self):
        """L2 norm."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def to_spherical(self):
        """Convert to 3D spherical coordinates."""
        radius = self.norm()
        azimuth = math.atan2(self.y, self.x)
        elevation = math.acos(self.z / radius)

        return SphericalCoordinate3D(radius, SphericalDirection3D(azimuth, elevation))


class CartesianCoordinate2D:
    """2D cartesian coordinates."""

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y

    def norm(self):
        """L2 norm."""
        return math.sqrt(self.x**2 + self.y**2)
