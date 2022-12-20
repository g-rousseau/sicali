import math

from sicali import CameraConfig, CameraModel, FocalAdjustment, LensModelType


def main():
    # Set USE_TEX to False if no tex distribution is available on the operating device
    USE_TEX = True

    # Create a camera model
    camera_config = CameraConfig(
        focal_adjustment=FocalAdjustment.INSCRIBE_FOV,
        lens_fov=math.radians(220),
        lens_focal=1,
        lens_model=LensModelType.EQUISOLID,
        sensor_definition_x=20,
        sensor_definition_y=20,
        sensor_pixel_size=0.05,
    )
    camera_model = CameraModel(camera_config)

    # Simulate the projection model of the camera
    camera_model.display_lens_model(use_tex=USE_TEX)
    camera_model.display_plane_projection(
        plane_size=10, plane_depth=3, line_count=10, use_tex=USE_TEX
    )
    camera_model.display_latitude_longitude_projection(line_count=12, use_tex=USE_TEX)
    camera_model.display_sphere_slices_projection(line_count=12, use_tex=USE_TEX)
    camera_model.display_pixels_propagation(use_tex=USE_TEX)


if __name__ == "__main__":
    main()
