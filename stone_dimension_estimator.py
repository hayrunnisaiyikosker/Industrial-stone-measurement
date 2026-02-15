"""
Industrial Stone Measurement – Prototype v1.0

This script implements a computer vision–based 3D stone measurement system
using a ZED stereo camera. It performs point cloud processing, segmentation,
dimension estimation, and basic industrial analysis.

"""

import sys
import os
import time
import datetime
import traceback
import argparse
import signal
import json

import numpy as np
import cv2
import open3d as o3d
import pyzed.sl as sl


# -------------------- Helper Class --------------------
class GracefulKiller:
    """Catches CTRL+C / SIGINT to safely close the loop."""

    def __init__(self):
        self.kill_now = False
        signal.signal(signal.SIGINT, self.exit_gracefully)

    def exit_gracefully(self, *args):
        print("\n🛑 CTRL+C detected, preparing to exit...")
        self.kill_now = True


# -------------------- Main Application --------------------
class StoneDimensionEstimator:
    def __init__(self, args):
        """
        Opens ZED, retrieves intrinsics, initializes visualizer,
        sets up chessboard parameters.
        """
        self.args = args
        self.zed = sl.Camera()
        self.kerf = args.kerf
        self.margin = args.margin
        self.min_piece_thickness = args.min_piece_thickness
        self.plate_dimensions = {
            'x': args.plate_size_x,
            'y': args.plate_size_y,
            'z': args.plate_size_z
        }

        # --- ZED initialization parameters ---
        init_params = sl.InitParameters()
        init_params.camera_resolution = sl.RESOLUTION.HD720
        init_params.depth_mode = sl.DEPTH_MODE.NEURAL
        init_params.coordinate_units = sl.UNIT.MILLIMETER
        init_params.depth_minimum_distance = self.args.depth_min
        init_params.depth_maximum_distance = self.args.depth_max
        init_params.camera_fps = self.args.fps
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.enable_right_side_measure = False

        # Open camera
        err = self.zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to open ZED camera: {repr(err)}")

        # Runtime parameters
        self.runtime_parameters = sl.RuntimeParameters()
        self.runtime_parameters.confidence_threshold = 50
        self.runtime_parameters.enable_fill_mode = True

        # Buffers
        self.point_cloud = sl.Mat()
        self.image = sl.Mat()
        self.points_grid_mm = None

        # Chessboard parameters
        self.cb_cols = args.cb_cols  # inner corners (columns)
        self.cb_rows = args.cb_rows  # inner corners (rows)
        self.square_size_mm = args.square_size_mm
        self.scale_smooth = float(np.clip(args.scale_smooth, 0.0, 1.0))

        # Scale factor and calibration status
        self.scale_factor = 1.0
        self.calibration_done = False

        # Load calibration file
        if self.load_calibration():
            print(
                f"✅ Previously calibrated scale loaded: {self.scale_factor:.4f}")
        else:
            print("ℹ Calibration not found, using default scale: 1.0")

        # If autocalib is enabled, perform calibration at startup
        if self.args.autocalib:
            print(
                "Automatic startup calibration enabled: Please show chessboard to camera...")
            success = self.perform_calibration()
            if success:
                print(
                    f"✅ Calibration complete. Scale: {self.scale_factor:.4f}")
            else:
                print("❌ Calibration failed, using default scale")

        # Camera intrinsics
        cam_info = self.zed.get_camera_information()
        self.camera_intrinsics = cam_info.camera_configuration.calibration_parameters.left_cam

        print("Camera intrinsic parameters retrieved successfully.")
        print(f"fx={self.camera_intrinsics.fx:.2f}, fy={self.camera_intrinsics.fy:.2f}, "
              f"cx={self.camera_intrinsics.cx:.2f}, cy={self.camera_intrinsics.cy:.2f}")

        # Open3D visualizer
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window("3D Analysis", width=960,
                               height=720, left=50, top=50)
        opt = self.vis.get_render_option()
        opt.background_color = np.asarray([0.15, 0.15, 0.18])
        opt.mesh_show_back_face = True
        self.view_control = self.vis.get_view_control()

        # Mode and state holders
        self.mode = "realtime"
        self.last_mesh = None
        self.last_obb = None
        self.last_image = None
        self.snapshot_prepared = False
        self.should_exit = False

        self.register_keys()  # Register Open3D key callbacks

    def run(self):
        self.run_customer()

    def run_customer(self):
        print("Keyboard shortcuts:")
        print("R=realtime, S=snapshot, K=save image, O=export OBJ, P=export PLY")
        print("C=manual calibration, T=cut report, I=industrial report, Q=quit")

        # Create and resize CV2 window
        cv2.namedWindow("Camera Image", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera Image", 640, 480)
        cv2.setWindowProperty("Camera Image", cv2.WND_PROP_TOPMOST, 1)

        killer = GracefulKiller()
        prev_cam_params = None
        last_t = time.time()

        try:
            while not (self.should_exit or killer.kill_now):
                # Capture keys from CV2 window
                k = cv2.waitKey(10) & 0xFF
                if k != 255:
                    if k == ord("q") or k == ord("Q"):
                        print("Exit key pressed (cv2).")
                        break
                    elif k == ord("r") or k == ord("R"):
                        self.mode = "realtime"
                        self.snapshot_prepared = False
                        print("🎥 Switched to realtime mode (cv2 key).")
                    elif k == ord("s") or k == ord("S"):
                        if self.last_mesh is not None or self.last_obb is not None:
                            self.mode = "snapshot"
                            if not self.snapshot_prepared:
                                try:
                                    prev_cam = self.view_control.convert_to_pinhole_camera_parameters()
                                except Exception:
                                    prev_cam = None
                                self.vis.clear_geometries()
                                if self.last_mesh is not None:
                                    self.vis.add_geometry(self.last_mesh)
                                if self.last_obb is not None:
                                    self.vis.add_geometry(self.last_obb)
                                if prev_cam:
                                    try:
                                        self.view_control.convert_from_pinhole_camera_parameters(
                                            prev_cam)
                                    except Exception:
                                        pass
                                self.snapshot_prepared = True
                            print("📸 Switched to snapshot mode (cv2 key).")
                        else:
                            print("⚠ No data for snapshot")
                    elif k == ord("k") or k == ord("K"):
                        if self.last_image is not None:
                            os.makedirs(self.args.out_dir, exist_ok=True)
                            filename = os.path.join(
                                self.args.out_dir,
                                f"snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                            )
                            cv2.imwrite(filename, self.last_image)
                            print(f"💾 Camera image saved: {filename}")
                        else:
                            print("⚠ No image available.")
                    elif k == ord("o") or k == ord("O"):
                        if self.last_mesh is not None:
                            os.makedirs(self.args.out_dir, exist_ok=True)
                            fname = os.path.join(
                                self.args.out_dir,
                                f"stone_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
                            )
                            o3d.io.write_triangle_mesh(
                                fname, self.last_mesh, write_ascii=True)
                            print(f"💾 3D mesh saved as OBJ: {fname}")
                        else:
                            print("⚠ No mesh for export.")
                    elif k == ord("p") or k == ord("P"):
                        if self.last_mesh is not None:
                            os.makedirs(self.args.out_dir, exist_ok=True)
                            fname = os.path.join(
                                self.args.out_dir,
                                f"stone_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
                            )
                            o3d.io.write_triangle_mesh(fname, self.last_mesh)
                            print(f"💾 3D mesh saved as PLY: {fname}")
                        else:
                            print("⚠ No mesh for export.")
                    elif k == ord("c") or k == ord("C"):
                        print("🔧 Starting manual calibration...")
                        success = self.perform_calibration()
                        if success:
                            print(
                                f"✅ Calibration complete. New scale: {self.scale_factor:.4f}")
                        else:
                            print("❌ Calibration failed")
                    elif k == ord("t") or k == ord("T"):
                        self._trigger_cut_report()
                    elif k == ord("i") or k == ord("I"):
                        self._trigger_industrial_report()

                # Main processing loop (realtime or snapshot)
                if self.mode == "realtime":
                    try:
                        points_np, image_bgr = self.get_point_cloud_and_image()
                        if points_np is None or len(points_np) < 100:
                            blank = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(blank, "Waiting for ZED data...", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                            cv2.imshow("Camera Image", blank)
                            self.vis.poll_events()
                            self.vis.update_renderer()
                            continue

                        # Update scale and process point cloud
                        self.update_scale_from_chessboard(image_bgr)
                        _, stone_pcd = self.process_point_cloud(
                            points_np, voxel_size=self.args.voxel)
                        measurements = self.calculate_measurements(stone_pcd)
                        stone_mesh = self.create_stone_mesh(stone_pcd)

                        self.last_mesh = stone_mesh
                        self.last_obb = measurements["oriented_bounding_box"] if measurements else None
                        self.last_image = image_bgr.copy()

                        prev_cam_params = None
                        try:
                            prev_cam_params = self.view_control.convert_to_pinhole_camera_parameters()
                        except Exception:
                            pass

                        self.vis.clear_geometries()
                        if stone_mesh is not None:
                            self.vis.add_geometry(stone_mesh)
                        if measurements is not None and self.last_obb is not None:
                            self.vis.add_geometry(self.last_obb)

                        if prev_cam_params is not None:
                            try:
                                self.view_control.convert_from_pinhole_camera_parameters(
                                    prev_cam_params)
                            except Exception:
                                pass

                        self.snapshot_prepared = False
                    except Exception:
                        traceback.print_exc()

                # Open3D render
                self.vis.poll_events()
                self.vis.update_renderer()

                # 2D overlay
                if self.mode == "realtime" and self.last_image is not None:
                    img_show = image_bgr if 'image_bgr' in locals(
                    ) and image_bgr is not None else self.last_image
                    if 'measurements' in locals() and measurements is not None:
                        dims = measurements["dimensions_mm"]
                        dim_text = f"{dims[0]/10:.1f} x {dims[1]/10:.1f} x {dims[2]/10:.1f} cm"
                        volume_text = f"Volume: {measurements['volume_cm3']:.0f} cm3"
                        weight_text = f"Weight: {measurements['weight_kg']:.2f} kg"
                        status_text = "Calibrated" if self.calibration_done else "Not Calibrated"

                        cv2.putText(img_show, dim_text, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(img_show, volume_text, (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                        cv2.putText(img_show, weight_text, (10, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                        cv2.putText(img_show, f"Scale: {self.scale_factor:.3f}", (10, 120),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)
                        cv2.putText(img_show, f"Status: {status_text}", (10, 150),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                    (0, 255, 0) if self.calibration_done else (0, 0, 255), 2)

                        if self.last_obb is not None:
                            center_3d = self.last_obb.get_center()
                            center_2d = self.project_3d_to_2d(center_3d)
                            if center_2d is not None:
                                cv2.circle(img_show, center_2d,
                                           6, (0, 0, 255), -1)

                    # FPS
                    now = time.time()
                    fps = 1.0 / max(1e-6, (now - last_t))
                    last_t = now
                    cv2.putText(img_show, f"FPS: {fps:.1f}", (10, img_show.shape[0] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

                    # Show image
                    cv2.imshow("Camera Image", img_show)
                else:
                    if self.last_image is not None:
                        cv2.imshow("Camera Image", self.last_image)
                    else:
                        blank = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.imshow("Camera Image", blank)

        finally:
            if self.calibration_done:
                self.save_calibration()
            print("Closing...")
            try:
                self.vis.destroy_window()
            except Exception:
                pass
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            try:
                self.zed.close()
            except Exception:
                pass

    def get_all_orientations(self, dims):
        """Returns 6 different orientations"""
        return {
            'XYZ': [dims[0], dims[1], dims[2]],
            'XZY': [dims[0], dims[2], dims[1]],
            'YXZ': [dims[1], dims[0], dims[2]],
            'YZX': [dims[1], dims[2], dims[0]],
            'ZXY': [dims[2], dims[0], dims[1]],
            'ZYX': [dims[2], dims[1], dims[0]]
        }

    def _calculate_axis_plates(self, available_length, plate_length):
        """Calculates plate count on a single axis"""
        if available_length <= 0 or plate_length <= 0:
            return {'count': 0, 'waste': available_length, 'cuts': 0}

        # Effective plate length including kerf
        effective_plate_length = plate_length + self.kerf

        # How many plates fit?
        max_plates = int(available_length // effective_plate_length)

        # Remaining material (waste)
        remaining_material = available_length - \
            (max_plates * effective_plate_length) + self.kerf

        return {
            'count': max_plates,
            'waste': max(0, remaining_material),
            'cuts': max_plates  # One cut per plate
        }

    def _calculate_cutting_loss(self, plates_x, plates_y, plates_z, plate_x, plate_y, plate_z):
        """Calculates cutting loss"""
        # Cutting areas
        cut_area_x = plate_y * plate_z  # X-axis cutting area
        cut_area_y = plate_x * plate_z  # Y-axis cutting area
        cut_area_z = plate_x * plate_y  # Z-axis cutting area

        # Total cutting loss (kerf * cutting area * number of cuts)
        cutting_loss = (
            (plates_x['cuts'] * self.kerf * cut_area_x) +
            (plates_y['cuts'] * self.kerf * cut_area_y) +
            (plates_z['cuts'] * self.kerf * cut_area_z)
        )

        return cutting_loss

    def _get_plate_sizes(self):
        """Gets factory plate dimensions"""
        try:
            # Can be retrieved from factory database or config here
            if hasattr(self.args, 'plate_sizes'):
                return self.args.plate_sizes
            else:
                # Standard plate dimensions (mm)
                return [
                    (500, 500, 50),    # Standard plate
                    (600, 600, 40),    # Large plate
                    (400, 400, 60),    # Thick plate
                    (300, 300, 30)     # Small plate
                ]
        except:
            return [(500, 500, 50)]

    def _calculate_realistic_cutting_loss(self, nx, ny, nz, px, py, pz):
        """Realistic cutting loss calculation"""
        # Diamond wire/disk thickness (mm)
        cutting_thickness = 2.5

        # Cutting surface areas
        cut_area_x = py * pz  # X-axis cutting area
        cut_area_y = px * pz  # Y-axis cutting area
        cut_area_z = px * py  # Z-axis cutting area

        # Total cutting loss (mm³)
        cutting_loss = (
            (nx * cutting_thickness * cut_area_x) +  # X-axis loss
            (ny * cutting_thickness * cut_area_y) +  # Y-axis loss
            (nz * cutting_thickness * cut_area_z)    # Z-axis loss
        )

        return cutting_loss

    def _format_plan_results(self, plan):
        """Rounds and formats results"""
        return {
            "stone_dims": plan["stone_dims"],
            "optimal_orientation": plan["optimal_orientation"],
            "plates_per_axis": plan["plates_per_axis"],
            "total_plates": plan["total_plates"],
            "plate_size_used": plan["plate_size_used"],
            "stone_volume": round(plan["stone_volume"], 2),
            "gross_usable_volume": round(plan["gross_usable_volume"], 2),
            "net_usable_volume": round(plan["net_usable_volume"], 2),
            "cutting_loss": round(plan["cutting_loss"], 2),
            "waste_volume": round(plan["waste_volume"], 2),
            "efficiency": round(plan["efficiency"], 2),
            "waste_ratio": round(plan["waste_ratio"], 2),
            "recommended_rotation": plan["recommended_rotation"],
            "orientation_index": plan["orientation_index"]
        }

    def _get_empty_plan(self, stone_dims):
        """Empty plan for error case"""
        x, y, z = stone_dims
        stone_volume = x * y * z
        return {
            "stone_dims": stone_dims,
            "optimal_orientation": stone_dims,
            "plates_per_axis": (0, 0, 0),
            "total_plates": 0,
            "plate_size_used": (0, 0, 0),
            "stone_volume": stone_volume,
            "gross_usable_volume": 0,
            "net_usable_volume": 0,
            "cutting_loss": 0,
            "waste_volume": stone_volume,
            "efficiency": 0.0,
            "waste_ratio": 100.0,
            "recommended_rotation": False,
            "orientation_index": 0
        }

    def calculate_multiple_plates_optimization(self, stone_dims):
        """
        Tries multiple plate size combinations.
        Optimizes by mixing different plate types.
        """
        # Try different plate combinations
        plate_combinations = self._generate_plate_combinations()

        best_combination = None
        best_efficiency = 0

        for combination in plate_combinations:
            plan = self._calculate_with_plate_combination(
                stone_dims, combination)
            if plan and plan["efficiency"] > best_efficiency:
                best_efficiency = plan["efficiency"]
                best_combination = plan

        return best_combination

    def _generate_plate_combinations(self):
        """Generates possible plate combinations"""
        plate_sizes = self._get_plate_sizes()
        combinations = []

        # Single plate type usage
        for plate in plate_sizes:
            combinations.append([plate])

        # Two different plate combinations
        for i, plate1 in enumerate(plate_sizes):
            for plate2 in plate_sizes[i+1:]:
                combinations.append([plate1, plate2])

        return combinations

    def calculate_with_kerf_optimization(self, stone_dims):
        """
        Optimizes cutting thickness for less waste.
        """
        # Cutting thickness optimization
        kerf_options = [2.0, 2.5, 3.0, 3.5]  # different cutting thicknesses (mm)

        best_kerf_plan = None
        best_kerf_efficiency = 0

        for kerf in kerf_options:
            self.args.cutting_thickness = kerf  # Temporarily change cutting thickness
            plan = self.calculate_industrial_cutting_plan(stone_dims)

            if plan and plan["efficiency"] > best_kerf_efficiency:
                best_kerf_efficiency = plan["efficiency"]
                best_kerf_plan = plan.copy()
                best_kerf_plan["optimal_kerf"] = kerf

        return best_kerf_plan

    def _calculate_single_orientation_plan(self, stone_dims):
        """
        Cutting plan for single orientation
        """
        # Effective dimensions (minus margin)
        effective_stone = [
            max(0, stone_dims[0] - 2 * self.margin),
            max(0, stone_dims[1] - 2 * self.margin),
            max(0, stone_dims[2] - 2 * self.margin)
        ]

        # How many plates fit on each axis
        plates_x = self._calculate_axis_cuts(
            effective_stone[0], self.plate_dimensions['x'])
        plates_y = self._calculate_axis_cuts(
            effective_stone[1], self.plate_dimensions['y'])
        plates_z = self._calculate_axis_cuts(
            effective_stone[2], self.plate_dimensions['z'])

        total_plates = plates_x['count'] * \
            plates_y['count'] * plates_z['count']

        # Volume calculations
        total_volume = stone_dims[0] * stone_dims[1] * stone_dims[2]
        usable_volume = total_plates * (self.plate_dimensions['x'] *
                                        self.plate_dimensions['y'] *
                                        self.plate_dimensions['z'])

        waste_volume = total_volume - usable_volume
        waste_ratio = (waste_volume / total_volume) * \
            100 if total_volume > 0 else 100

        return {
            'total_plates': total_plates,
            'plates_per_axis': [plates_x['count'], plates_y['count'], plates_z['count']],
            'waste_ratio': waste_ratio,
            'waste_volume': waste_volume,
            'usable_volume': usable_volume,
            'total_volume': total_volume,
            'efficiency': 100 - waste_ratio,
            'cutting_details': {
                'x_axis': plates_x,
                'y_axis': plates_y,
                'z_axis': plates_z
            }
        }

    def _calculate_axis_cuts(self, available_length, plate_length):
        """
        Cutting calculation on single axis
        """
        if available_length <= 0 or plate_length <= 0:
            return {'count': 0, 'waste': available_length, 'cuts': 0}

        # Consider edge margin and cutting loss
        usable_length = available_length - 2 * self.margin
        if usable_length <= 0:
            return {'count': 0, 'waste': available_length, 'cuts': 0}

        # Effective plate length including kerf
        effective_plate_length = plate_length + self.kerf

        # How many plates fit?
        max_plates = int(usable_length // effective_plate_length)

        # Remaining material (waste)
        remaining_material = usable_length - \
            (max_plates * effective_plate_length) + self.kerf

        # Number of cuts required
        cuts_required = max_plates  # One cut per plate

        return {
            'count': max_plates,
            'waste': max(0, remaining_material),
            'cuts': cuts_required,
            'effective_plate_length': effective_plate_length
        }

    def generate_industrial_report(self, stone_dims):
        """
        Generates detailed cutting report for factory.
        """
        plan = self.calculate_industrial_cutting_plan(stone_dims)

        report = f"""
        🏭 STONE CUTTING OPTIMIZATION REPORT
        {'='*50}
        📏 STONE INFORMATION:
        Dimensions: {stone_dims[0]:.1f} × {stone_dims[1]:.1f} × {stone_dims[2]:.1f} mm
        Total Volume: {plan['stone_volume']/1e6:.1f} liters

        ⚙️ CUTTING SETTINGS:
        Plate: {plan['plate_size_used'][0]} × {plan['plate_size_used'][1]} × {plan['plate_size_used'][2]} mm
        Cutting Loss: {self.kerf} mm
        Edge Margin: {self.margin} mm

        🎯 OPTIMAL CUTTING PLAN:
        Total Plates: {plan['total_plates']} pieces
        Axis Distribution: X={plan['plates_per_axis'][0]}, Y={plan['plates_per_axis'][1]}, Z={plan['plates_per_axis'][2]}

        📈 EFFICIENCY ANALYSIS:
        Usable Volume: {plan['net_usable_volume']/1e6:.1f} liters
        Waste Volume: {plan['waste_volume']/1e6:.1f} liters
        Waste Ratio: {plan['waste_ratio']:.1f}%
        Efficiency: {plan['efficiency']:.1f}%

        🔪 CUTTING DETAILS:
        X Axis: {plan['cutting_details']['x_axis']['cuts']} cuts, {plan['cutting_details']['x_axis']['waste']:.1f} mm waste
        Y Axis: {plan['cutting_details']['y_axis']['cuts']} cuts, {plan['cutting_details']['y_axis']['waste']:.1f} mm waste
        Z Axis: {plan['cutting_details']['z_axis']['cuts']} cuts, {plan['cutting_details']['z_axis']['waste']:.1f} mm waste

        """
        return report

    def _calculate_plate_value(self, plate_size):
        """Calculates value per plate"""
        # Simple calculation: volume * unit price
        volume = plate_size[0] * plate_size[1] * plate_size[2]  # mm³
        return volume * self.args.material_value_per_mm3

    # ---------------------- REPORT FUNCTION ----------------------
    def print_industrial_report(self, plan):
        print("\n--- INDUSTRIAL CUTTING REPORT ---")
        x, y, z = plan['stone_dims']
        print(f"Stone dimensions: {x:.1f} × {y:.1f} × {z:.1f} mm")
        print(f"Total stone volume: {plan['total_volume']:,.0f} mm³")
        print(
            f"Plate dimensions: {plan['slab_dims'][0]} × {plan['slab_dims'][1]} × {plan['slab_dims'][2]} mm")
        print(
            f"Plate volume: {plan['slab_dims'][0] * plan['slab_dims'][1] * plan['slab_dims'][2]:,.0f} mm³")
        print(f"Total extractable plates: {plan['slab_count']}")
        print(
            f"Net usable volume: {plan['net_usable_volume']:,.0f} mm³")
        print(f"Waste volume: {plan['waste_volume']:,.0f} mm³")
        print(f"Yield ratio: %{plan['yield_ratio']:.2f}")
        print(
            f"Plate distribution per axis: X={plan['plates_per_axis'][0]}, Y={plan['plates_per_axis'][1]}, Z={plan['plates_per_axis'][2]}")
        print("-------------------------------\n")

# ---------------------- CUTTING PLAN CALCULATION ----------------------
    def calculate_industrial_cutting_plan(self, stone_dims):
        """
        Calculates industrial cutting plan.
        stone_dims: (x, y, z) tuple or numpy array format stone dimensions (mm)
        """
        try:
            stone_dims = np.asarray(stone_dims).flatten()
            if len(stone_dims) < 3:
                print(f"[ERROR] Invalid measurement dimensions: {stone_dims}")
                return None

            x, y, z = stone_dims[:3]

            # Stone volume
            total_volume = x * y * z

            # Plate size (example: 500x500x20 mm)
            slab_dims = (500, 500, 20)
            slab_volume = slab_dims[0] * slab_dims[1] * slab_dims[2]

            # How many plates can be extracted
            slab_count = max(1, int(total_volume // slab_volume))

            # Net usable volume
            net_usable_volume = slab_count * slab_volume

            # Waste and cutting loss
            waste_volume = total_volume - net_usable_volume
            cutting_loss = self.kerf * \
                (slab_count - 1) * (slab_dims[0] * slab_dims[1])

            # Ratios
            yield_ratio = (net_usable_volume / total_volume) * \
                100 if total_volume else 0
            efficiency = (net_usable_volume / total_volume) * \
                100 if total_volume else 0
            waste_ratio = (waste_volume / total_volume) * \
                100 if total_volume else 100

            # Plate distribution on axes
            plates_per_axis = (
                max(1, int(x // slab_dims[0])),
                max(1, int(y // slab_dims[1])),
                max(1, int(z // slab_dims[2]))
            )

           # Axis-based cutting details
            cutting_details = {
                'x_axis': self._calculate_axis_cuts(x, slab_dims[0]),
                'y_axis': self._calculate_axis_cuts(y, slab_dims[1]),
                'z_axis': self._calculate_axis_cuts(z, slab_dims[2]),
            }

            plan = {
                'stone_dims': (x, y, z),
                'stone_volume': total_volume,
                'total_volume': total_volume,
                'slab_dims': slab_dims,
                'slab_count': slab_count,
                'total_plates': slab_count,
                'plate_size_used': slab_dims,
                'net_usable_volume': net_usable_volume,
                'waste_volume': waste_volume,
                'cutting_loss': cutting_loss,
                'yield_ratio': yield_ratio,
                'efficiency': efficiency,
                'waste_ratio': waste_ratio,
                'plates_per_axis': plates_per_axis,
                'cutting_details': cutting_details
            }

            return plan

        except Exception as e:
            print(f"[ERROR] Error creating cutting plan: {e}")
            traceback.print_exc()
            return None


# ---------------------- KEY FUNCTIONS ----------------------


    def _trigger_cut_report(self):
        if self.last_obb is not None:
            # Get first 3 values and convert to float
            stone_dims = tuple(float(d) for d in self.last_obb.extent[:3])
            plan = self.calculate_industrial_cutting_plan(stone_dims)
            if plan:
                self.print_industrial_report(plan)
            else:
                print("⚠ Could not calculate cutting plan!")
        else:
            print("⚠ No measurements yet! Cannot generate cutting report.")

    def _trigger_industrial_report(self):
        """I key: detailed industrial report"""
        if self.last_obb is not None:
            stone_dims = tuple(float(d) for d in self.last_obb.extent[:3])
            report = self.generate_industrial_report(
                stone_dims)  # Detailed report function
            print(report)
        else:
            print("⚠ No measurements yet! Cannot generate industrial report.")


# ---------------------- KEY REGISTRATIONS ----------------------

    def register_keys(self):
        # Realtime key
        self.vis.register_key_callback(
            ord("R"), lambda vis: self.set_mode_realtime())

        # Snapshot key
        self.vis.register_key_callback(
            ord("S"), lambda vis: self.set_mode_snapshot())

        # Manual calibration
        self.vis.register_key_callback(
            ord("C"), lambda vis: self.perform_manual_calibration())

        # T key: cutting report
        self.vis.register_key_callback(
            ord("T"), lambda vis: self._trigger_cut_report())

        # I key: industrial report
        self.vis.register_key_callback(
            ord("I"), lambda vis: self._trigger_industrial_report())

    def set_mode_realtime(self):
        self.mode = "realtime"
        self.snapshot_prepared = False
        print("🎥 Switched to realtime mode (Open3D key).")

    def set_mode_snapshot(self):
        if self.last_mesh is not None or self.last_obb is not None:
            self.mode = "snapshot"
            print("📸 Switched to snapshot mode (Open3D key).")
        else:
            print("⚠ No data for snapshot")

    def perform_manual_calibration(self):
        print("🔧 Starting manual calibration...")
        success = self.perform_calibration()
        if success:
            print(
                f"✅ Calibration complete. New scale: {self.scale_factor:.4f}")
        else:
            print("❌ Calibration failed")

    def get_point_cloud_and_image(self):
        """
        Returns point cloud from ZED (Nx3 float64, mm) and BGR image.
        """
        if self.zed.grab(self.runtime_parameters) != sl.ERROR_CODE.SUCCESS:
            return None, None

        # Get point cloud
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
        pc = self.point_cloud.get_data()
        points = pc[:, :, :3].astype(np.float64)  # XYZ in mm
        self.points_grid_mm = points  # 2D grid (H x W x 3)

        # Filter invalid points
        mask = np.isfinite(points).all(axis=2)
        if self.args.require_positive_z:
            mask &= points[:, :, 2] > 0
        nonzero = ~np.isclose(points, 0.0).all(axis=2)
        mask &= nonzero
        valid_points = points[mask]

        # Get image
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        img = self.image.get_data()

        # RGBA -> BGR conversion
        try:
            image_bgr = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        except Exception:
            image_bgr = img[..., :3][:, :, ::-1]

        return valid_points, image_bgr

    # -------------------- Calibration --------------------

    def perform_calibration(self, num_frames=10):
        """Performs calibration over a specific number of frames (median stabilization)."""
        successful_frames = 0
        scale_values = []

        for _ in range(num_frames * 2):
            _, img = self.get_point_cloud_and_image()
            if img is not None:
                old_scale = self.scale_factor
                calibration_updated = self.update_scale_from_chessboard(img)
                if calibration_updated and abs(self.scale_factor - old_scale) > 0.001:
                    scale_values.append(self.scale_factor)
                    successful_frames += 1
                    print(
                        f"Calibration frame {successful_frames}/{num_frames}: {self.scale_factor:.4f}")
            if successful_frames >= num_frames:
                break
            time.sleep(0.1)

        if scale_values:
            self.scale_factor = float(np.median(scale_values))
            self.calibration_done = True
            self.save_calibration()
            return True
        return False

    def load_calibration(self, filepath="calibration_data.json"):
        """Loads calibration data from file."""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                self.scale_factor = data.get('scale_factor', 1.0)
                self.calibration_done = data.get('calibration_done', False)
                return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
        return False

    def save_calibration(self, filepath="calibration_data.json"):
        """Saves calibration data to file."""
        try:
            data = {
                'scale_factor': self.scale_factor,
                'calibration_done': self.calibration_done,
                'cb_cols': self.cb_cols,
                'cb_rows': self.cb_rows,
                'square_size_mm': self.square_size_mm,
                'save_date': datetime.datetime.now().isoformat()
            }
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"✅ Calibration saved: {filepath}")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False

    def update_scale_from_chessboard(self, image_bgr):
        """Updates scale from chessboard (mm/pixel -> 3D square edge ratio)."""
        try:
            if image_bgr is None or self.points_grid_mm is None:
                return False

            gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
            pattern_size = (self.cb_cols, self.cb_rows)

            found, corners = cv2.findChessboardCorners(
                gray, pattern_size,
                flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            if not found:
                return False

            # Subpixel refinement
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
            corners = cv2.cornerSubPix(
                gray, corners, (5, 5), (-1, -1), criteria)
            cv2.drawChessboardCorners(image_bgr, pattern_size, corners, found)

            pts2d = corners.reshape(-1, 2)
            H, W, _ = self.points_grid_mm.shape

            def pt3_from_uv(u, v):
                uu = int(round(u))
                vv = int(round(v))
                if uu < 0 or vv < 0 or uu >= W or vv >= H:
                    return None
                p = self.points_grid_mm[vv, uu, :]
                if not np.isfinite(p).all() or (p == 0).all():
                    return None
                if self.args.require_positive_z and p[2] <= 0:
                    return None
                return p

            dists = []
            # Horizontal edges
            for r in range(self.cb_rows):
                for c in range(self.cb_cols - 1):
                    i = r * self.cb_cols + c
                    j = r * self.cb_cols + (c + 1)
                    p1 = pt3_from_uv(pts2d[i, 0], pts2d[i, 1])
                    p2 = pt3_from_uv(pts2d[j, 0], pts2d[j, 1])
                    if p1 is not None and p2 is not None:
                        dists.append(np.linalg.norm(p1 - p2))
            # Vertical edges
            for r in range(self.cb_rows - 1):
                for c in range(self.cb_cols):
                    i = r * self.cb_cols + c
                    j = (r + 1) * self.cb_cols + c
                    p1 = pt3_from_uv(pts2d[i, 0], pts2d[i, 1])
                    p2 = pt3_from_uv(pts2d[j, 0], pts2d[j, 1])
                    if p1 is not None and p2 is not None:
                        dists.append(np.linalg.norm(p1 - p2))

            dists = [d for d in dists if np.isfinite(d) and d > 0]
            if not dists:
                return False

            measured_edge_mm = float(np.median(dists))
            if measured_edge_mm <= 0:
                return False

            # New scale (expected square edge / measured edge)
            s_new = self.square_size_mm / measured_edge_mm
            a = self.scale_smooth
            self.scale_factor = (1.0 - a) * self.scale_factor + a * s_new
            self.calibration_done = True

            cv2.putText(image_bgr, f"Scale: {self.scale_factor:.3f}", (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)
            cv2.putText(image_bgr, "Calibrated", (10, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            return True

        except Exception as e:
            print(f"Calibration error: {e}")
            return False

    # -------------------- Point Cloud Processing --------------------

    def process_point_cloud(self, points_np, voxel_size=5.0):
        """Downsamples, cleans, separates ground, and returns largest cluster."""
        if points_np is None or len(points_np) == 0:
            return None, None

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_np)

        # Voxel downsampling
        pcd_down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
        if len(pcd_down.points) == 0:
            return pcd_down, None

        # Statistical noise removal
        pcd_clean, _ = pcd_down.remove_statistical_outlier(
            nb_neighbors=self.args.nb_neighbors,
            std_ratio=self.args.std_ratio
        )
        if len(pcd_clean.points) == 0:
            return pcd_clean, None

        objects_pcd = pcd_clean
        # Separate ground (optional)
        if not self.args.skip_ground:
            try:
                plane_model, inliers = pcd_clean.segment_plane(
                    distance_threshold=self.args.plane_thresh,
                    ransac_n=3,
                    num_iterations=1000
                )
                objects_pcd = pcd_clean.select_by_index(inliers, invert=True)
            except Exception:
                pass

        if len(objects_pcd.points) == 0:
            return pcd_clean, None

        # DBSCAN clustering
        labels = np.array(objects_pcd.cluster_dbscan(
            eps=self.args.cluster_eps,
            min_points=self.args.cluster_min_pts,
            print_progress=False
        ))
        if labels.size == 0 or labels.max() == -1:
            return pcd_clean, None

        counts = np.bincount(labels[labels >= 0])
        if counts.size == 0:
            return pcd_clean, None

        largest_cluster_label = int(np.argmax(counts))
        stone_indices = np.where(labels == largest_cluster_label)[0]
        stone_pcd = objects_pcd.select_by_index(stone_indices)
        return pcd_clean, stone_pcd

    def _estimate_alpha(self, stone_pcd):
        """Adaptive alpha estimation for alpha-shape (local neighborhood median)."""
        try:
            if len(stone_pcd.points) < 30:
                return max(self.args.alpha, 10.0)
            pcd_tree = o3d.geometry.KDTreeFlann(stone_pcd)
            dists = []
            idxs = np.random.choice(len(stone_pcd.points), size=min(
                200, len(stone_pcd.points)), replace=False)
            pts = np.asarray(stone_pcd.points)
            for i in idxs:
                k, idx, _ = pcd_tree.search_knn_vector_3d(pts[i], 10)
                if k > 1:
                    nn = pts[idx[1:k]] - pts[i]
                    di = np.linalg.norm(nn, axis=1)
                    if di.size:
                        dists.append(np.median(di))
            if not dists:
                return self.args.alpha
            base = float(np.median(dists))
            return float(np.clip(self.args.alpha_multiplier * base, 5.0, 80.0))
        except Exception:
            return self.args.alpha

    def calculate_measurements(self, stone_pcd):
        """Estimates dimensions (mm), volume (cm³ ~ box volume), weight (kg) from OBB."""
        if stone_pcd is None or len(stone_pcd.points) < 10:
            return None

        obb = stone_pcd.get_oriented_bounding_box()
        obb.color = (1.0, 0.0, 0.0)
        # Scale OBB dimensions (ZED mm -> scale)
        dimensions = np.asarray(obb.extent) * self.scale_factor  # mm

        volume_mm3 = dimensions[0] * dimensions[1] * dimensions[2]
        volume_cm3 = volume_mm3 / 1000.0
        weight_kg = (volume_cm3 * self.args.density_g_cm3) / 1000.0

        return {
            "dimensions_mm": dimensions,
            "volume_cm3": volume_cm3,
            "weight_kg": weight_kg,
            "oriented_bounding_box": obb,
        }

    def estimate_cut_count(self, stone_dims):
        """
        Returns estimated cut count and axis-based count based on stone dimensions (mm) and plate size.
        """
        plate_dims = np.array([self.args.plate_size_x,
                               self.args.plate_size_y,
                               self.args.plate_size_z])
        min_piece = self.args.min_piece_mm
        # Axis-based piece count
        cuts = np.ceil(stone_dims / min_piece).astype(int)
        # Limit cuts to not exceed plate size
        cuts = np.minimum(cuts, np.ceil(plate_dims / min_piece).astype(int))
        total_pieces = np.prod(cuts)
        return total_pieces, cuts

    def calculate_waste_ratio(self, stone_dims):
        """
        Calculates waste ratio (%) based on optimal cutting plan.
        Waste = (Stone Volume - Usable Volume) / Stone Volume * 100
        """
        try:
            plan = self.calculate_industrial_cutting_plan(stone_dims)

            # Check if plan is valid
            if not plan or "waste_ratio" not in plan:
                return 100.0  # If no cuts possible, 100% waste

            return float(plan["waste_ratio"])

        except Exception as e:
            print(f"[ERROR] Could not calculate waste: {e}")
            return 100.0  # Maximum waste on error

    def create_stone_mesh(self, stone_pcd):
        """Generates mesh using alpha-shape (if applicable) / convex hull."""
        if stone_pcd is None or len(stone_pcd.points) < 20:
            return None
        try:
            alpha = self._estimate_alpha(stone_pcd)
            mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
                stone_pcd, alpha)
            try:
                mesh = mesh.remove_unreferenced_vertices() \
                    .remove_degenerate_triangles() \
                    .remove_duplicated_vertices() \
                    .remove_duplicated_triangles() \
                    .remove_non_manifold_edges()
            except Exception:
                pass
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.7, 0.7, 0.7])
            return mesh
        except Exception:
            try:
                hull, _ = stone_pcd.compute_convex_hull()
                hull.compute_vertex_normals()
                hull.paint_uniform_color([0.7, 0.7, 0.7])
                return hull
            except Exception:
                return None

    # -------------------- 3D->2D Projection --------------------

    def project_3d_to_2d(self, point_3d):
        """Projects 3D point to left camera's 2D pixel plane."""
        fx = self.camera_intrinsics.fx
        fy = self.camera_intrinsics.fy
        cx = self.camera_intrinsics.cx
        cy = self.camera_intrinsics.cy
        x, y, z = point_3d
        if z <= 0:
            return None
        u = int((x * fx / z) + cx)
        v = int((y * fy / z) + cy)
        return (u, v)

    def handle_key_press(self, k):
        """Handles key presses"""
        # --- Exit ---
        if k in (ord("q"), ord("Q")):
            print("Exit key pressed (cv2).")
            self.should_exit = True
            self.cleanup()  # Cleanup method
            return

        # --- Other key operations (original code) ---
        elif k in (ord("r"), ord("R")):
            self.mode = "realtime"
            self.snapshot_prepared = False
            print("🎥 Switched to realtime mode (cv2 key).")

        elif k in (ord("s"), ord("S")):
            if self.last_mesh is not None or self.last_obb is not None:
                self.mode = "snapshot"
                if not self.snapshot_prepared:
                    try:
                        prev_cam = self.view_control.convert_to_pinhole_camera_parameters()
                    except Exception:
                        prev_cam = None

                    self.vis.clear_geometries()
                    if self.last_mesh is not None:
                        self.vis.add_geometry(self.last_mesh)
                    if self.last_obb is not None:
                        self.vis.add_geometry(self.last_obb)

                    if prev_cam:
                        try:
                            self.view_control.convert_from_pinhole_camera_parameters(
                                prev_cam)
                        except Exception:
                            pass

                    self.snapshot_prepared = True

                print("📸 Switched to snapshot mode (cv2 key).")
            else:
                print("⚠ No data for snapshot")

        elif k in (ord("k"), ord("K")):
            if self.last_image is not None:
                os.makedirs(self.args.out_dir, exist_ok=True)
                filename = os.path.join(
                    self.args.out_dir,
                    f"snapshot_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                )
                cv2.imwrite(filename, self.last_image)
                print(f"💾 Camera image saved: {filename}")
            else:
                print("⚠ No image available.")

        elif k in (ord("o"), ord("O")):
            if self.last_mesh is not None:
                os.makedirs(self.args.out_dir, exist_ok=True)
                fname = os.path.join(
                    self.args.out_dir,
                    f"stone_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.obj"
                )
                o3d.io.write_triangle_mesh(
                    fname, self.last_mesh, write_ascii=True)
                print(f"💾 3D mesh saved as OBJ: {fname}")
            else:
                print("⚠ No mesh for export.")

        elif k in (ord("p"), ord("P")):
            if self.last_mesh is not None:
                os.makedirs(self.args.out_dir, exist_ok=True)
                fname = os.path.join(
                    self.args.out_dir,
                    f"stone_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.ply"
                )
                o3d.io.write_triangle_mesh(fname, self.last_mesh)
                print(f"💾 3D mesh saved as PLY: {fname}")
            else:
                print("⚠ No mesh for export.")

        elif k in (ord("c"), ord("C")):
            print("🔧 Starting manual calibration...")
            success = self.perform_calibration()
            if success:
                print(
                    f"✅ Calibration complete. New scale: {self.scale_factor:.4f}")
            else:
                print("❌ Calibration failed")

        elif k in (ord("t"), ord("T")):
            print("📏 Starting stone measurement process...")
            if self.last_mesh is not None or self.last_obb is not None:
                self.process_stone_measurement()
            else:
                print("⚠ No mesh or OBB for measurement")

        elif k == ord("i") or k == ord("I"):
            if self.last_obb is not None:
                # Get only dimensions from measurement dict
                stone_dims = np.asarray(
                    self.last_obb.extent) * self.scale_factor
                stone_dims = tuple(map(float, stone_dims[:3]))
                self.calculate_industrial_cutting_plan(stone_dims)
            else:
                print("⚠ No measurements for industrial report.")

    # --- Main processing loop ---
        try:
            if self.mode == "realtime":
                points_np, image_bgr = self.get_point_cloud_and_image()

                # If no data, show blank screen
                if points_np is None or len(points_np) < 100:
                    blank = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(blank, "Waiting for ZED data...", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.imshow("Camera Image", blank)
                    self.vis.poll_events()
                    self.vis.update_renderer()
                    return

                # Update scale if chessboard visible
                self.update_scale_from_chessboard(image_bgr)

                # Process point cloud
                _, stone_pcd = self.process_point_cloud(
                    points_np, voxel_size=self.args.voxel)

                # Calculate measurements and create mesh
                measurements = self.calculate_measurements(stone_pcd)
                stone_mesh = self.create_stone_mesh(stone_pcd)

                self.last_mesh = stone_mesh
                self.last_obb = measurements["oriented_bounding_box"] if measurements else None
                self.last_image = image_bgr.copy()
                self.last_measurements = measurements  # Save measurements

                # Update Open3D scene
                try:
                    prev_cam_params = self.view_control.convert_to_pinhole_camera_parameters()
                except Exception:
                    prev_cam_params = None

                self.vis.clear_geometries()
                if stone_mesh is not None:
                    self.vis.add_geometry(stone_mesh)
                if measurements is not None and self.last_obb is not None:
                    self.vis.add_geometry(self.last_obb)

                if prev_cam_params is not None:
                    try:
                        self.view_control.convert_from_pinhole_camera_parameters(
                            prev_cam_params)
                    except Exception:
                        pass

                self.snapshot_prepared = False

        except Exception as e:
            print(f"Error occurred: {e}")
            traceback.print_exc()

        # --- Open3D render ---
        self.vis.poll_events()
        self.vis.update_renderer()

        # --- 2D overlay ---
        if self.mode == "realtime" and self.last_image is not None:
            img_show = self.last_image.copy()

            # Show measurement information
            if hasattr(self, 'last_measurements') and self.last_measurements is not None:
                dims = self.last_measurements["dimensions_mm"]
                dim_text = f"{dims[0]/10:.1f} x {dims[1]/10:.1f} x {dims[2]/10:.1f} cm"
                volume_text = f"Volume: {self.last_measurements['volume_cm3']:.0f} cm3"
                weight_text = f"Weight: {self.last_measurements['weight_kg']:.2f} kg"
                status_text = "Calibrated" if self.calibration_done else "Not Calibrated"

                cv2.putText(img_show, dim_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(img_show, volume_text, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(img_show, weight_text, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(img_show, f"Scale: {self.scale_factor:.3f}", (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 255, 150), 2)
                cv2.putText(img_show, f"Status: {status_text}", (10, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if self.calibration_done else (0, 0, 255), 2)

                if self.last_obb is not None:
                    center_3d = self.last_obb.get_center()
                    center_2d = self.project_3d_to_2d(center_3d)
                    if center_2d is not None:
                        cv2.circle(img_show, center_2d, 6, (0, 0, 255), -1)

            # FPS
            now = time.time()
            fps = 1.0 / max(1e-6, (now - self.last_t))
            self.last_t = now
            cv2.putText(img_show, f"FPS: {fps:.1f}",
                        (10, img_show.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            cv2.imshow("Camera Image", img_show)

        else:
            # In snapshot mode show last image
            if self.last_image is not None:
                cv2.imshow("Camera Image", self.last_image)
            else:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.imshow("Camera Image", blank)

    def cleanup(self):
        """Clean up resources"""
        print("Closing...")
        if self.calibration_done:
            self.save_calibration()

        try:
            self.vis.destroy_window()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        try:
            self.zed.close()
        except Exception:
            pass

    # -------------------- Arguments --------------------


def parse_args():
    p = argparse.ArgumentParser(description="ZED-based stone dimension measurement demo")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--max-fps", dest="max_fps", type=int, default=0)
    p.add_argument("--depth-min", dest="depth_min", type=int, default=400)
    p.add_argument("--depth-max", dest="depth_max", type=int, default=5000)
    p.add_argument("--voxel", type=float, default=5.0)
    p.add_argument("--nb-neighbors", dest="nb_neighbors", type=int, default=20)
    p.add_argument("--std-ratio", dest="std_ratio", type=float, default=2.0)
    p.add_argument("--plane-thresh", dest="plane_thresh",
                   type=float, default=15.0)
    p.add_argument("--cluster-eps", dest="cluster_eps",
                   type=float, default=25.0)
    p.add_argument("--cluster-min-pts", dest="cluster_min_pts",
                   type=int, default=25)
    p.add_argument("--alpha", type=float, default=25.0)
    p.add_argument("--alpha-multiplier", dest="alpha_multiplier",
                   type=float, default=2.5)
    p.add_argument("--skip-ground", action="store_true")
    p.add_argument("--require-positive-z", action="store_true")
    p.add_argument("--density", dest="density_g_cm3", type=float, default=2.7)
    p.add_argument("--out-dir", dest="out_dir", type=str, default="outputs")

    p.add_argument("--kerf", type=float, default=3.0, help="Cutting loss (mm)")
    p.add_argument("--margin", type=float, default=5.0,
                   help="Edge safety margin (mm)")
    p.add_argument("--min-piece-thickness", dest="min_piece_thickness",
                   type=float, default=20.0, help="Minimum piece thickness (mm)")
    p.add_argument("--min-piece-mm", dest="min_piece_mm",
                   type=float, default=100.0, help="Minimum piece size (mm)")

    p.add_argument("--material-value-per-mm3", type=float, default=0.00001,
                   help="Material unit price (₺/mm³)")
    p.add_argument("--material-cost-per-mm3", type=float, default=0.000005,
                   help="Waste material cost (₺/mm³)")

    # Chessboard
    p.add_argument("--cb-cols", dest="cb_cols",
                   type=int, default=9)   # inner corners
    p.add_argument("--cb-rows", dest="cb_rows",
                   type=int, default=6)   # inner corners
    p.add_argument("--square-size-mm", dest="square_size_mm",
                   type=float, default=25.0)
    p.add_argument("--scale-smooth", dest="scale_smooth",
                   type=float, default=0.2)

    p.add_argument("--autocalib", action="store_true")

    # Plate size (mm)
    p.add_argument("--plate-size-x", type=float,
                   default=1500.0, help="Plate length (mm)")
    p.add_argument("--plate-size-y", type=float,
                   default=750.0, help="Plate width (mm)")
    p.add_argument("--plate-size-z", type=float,
                   default=50.0, help="Plate thickness (mm)")
    return p.parse_args()


    # -------------------- Entry Point --------------------
if __name__ == "__main__":
    args = parse_args()

    # Get user inputs before starting StoneDimensionEstimator
    try:
        user_density = input(
            "Enter stone density (g/cm³) (press enter for default=2.7): ").strip()
        if user_density:
            args.density_g_cm3 = float(user_density)
    except Exception:
        print("Invalid input! Using default density 2.7 g/cm³.")
        args.density_g_cm3 = 2.7

    try:
        plate_x = input(
            "Enter plate length (mm) [default 1500]: ").strip()
        plate_y = input(
            "Enter plate width (mm) [default 750]: ").strip()
        plate_z = input(
            "Enter plate thickness (mm) [default 50]: ").strip()
        min_piece = input(
            "Enter minimum piece size (mm) [default 50]: ").strip()

        if plate_x:
            args.plate_size_x = float(plate_x)
        if plate_y:
            args.plate_size_y = float(plate_y)
        if plate_z:
            args.plate_size_z = float(plate_z)
        if min_piece:
            args.min_piece_mm = float(min_piece)
    except Exception:
        print("Invalid input, using default plate size.")

    try:
        kerf = input(
            "Enter cutting loss (kerf) (mm) [default 2.0]: ").strip()
        margin = input(
            "Enter edge margin (mm) [default 3.0]: ").strip()
        min_thickness = input(
            "Enter minimum piece thickness (mm) [default 15.0]: ").strip()

        if kerf:
            args.kerf = float(kerf)
        if margin:
            args.margin = float(margin)
        if min_thickness:
            args.min_piece_thickness = float(min_thickness)
    except Exception:
        print("Invalid input! Using default cutting parameters.")

    # Now start the application
    try:
        app = StoneDimensionEstimator(args)
        app.run()
    except Exception as e:
        print("Error occurred:", e)
        traceback.print_exc()
