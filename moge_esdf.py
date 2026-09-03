"""
see https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/depth_to_pointcloud.py
although not using the metric version of DA2


"""
import argparse
import os
import time
from email.mime import image
from pathlib import Path

import cv2
import torch
import numpy as np
import open3d as o3d
from PIL import Image
from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
from moge.model.v2 import MoGeModel
# from moge.model.v3 import MoGeModel # Let's try MoGe-3

from custom_utils.esdf_utils import parse_args, visualize_path, save_debug_figure
from custom_utils.stream_handler import FrameStatus, InputStreamHandler
from custom_utils.io_utils import colorize_pred, save_depth_video_mp4
from custom_utils.pointcloud_utils import camera_to_base_transform, pointcloud_to_esdf_pipeline

def main():
    # Initialize predictor (single-GPU streaming)
    show_depth_img = True
    save_video_toggle = False
    stream_type = "video" # ["yarp", "video", "webcam"]
    video_path = "/home/jim/Projects/steernav/assets/Cars_and_Gasstation.mp4"
    output_folder = "demo_video"
    webcam_index = 0
    yarp_port = "/sam3/rgbImage:i"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model_name = "Ruicheng/moge-2-vitl-normal"
    # model_name = "Ruicheng/moge-3-vitl"
    # model_name = "depth-anything/da3nested-giant-large"
    print("running", model_name)
    # model = DepthAnything3.from_pretrained(model_name)
    # model = model.to(device).eval()
    model = MoGeModel.from_pretrained(model_name).to(device).eval()
    # Initialize input source
    src = InputStreamHandler(
        kind=stream_type,
        video_path=video_path,
        webcam_index=webcam_index,
        # fps_request=2,
        yarp_port_name=yarp_port,
    )
    print(f"Opening source: {stream_type}")
    src.open()
    stream_buffer = src.read()
    print(f"stream size: {stream_buffer.frame.shape}")

    peak_memory = 0
    frame_idx = 0

    frame_timestamps = []  # To compute output fps
    video_frames = []  # Buffer of frames for final video save

    stop_processing = False

    prev_time = time.time()
    try:
        while stop_processing is not True:
            # Read frame (RGB)
            stream_buffer = src.read()
            if stream_buffer.status == FrameStatus.NO_FRAME:
                # YARP: no new frame yet; try again.
                continue
            if stream_buffer.status == FrameStatus.EOS:
                # End of stream for video/webcam or closed YARP port.
                break
            # Calculate FPS
            current_time = time.time()
            fps = 1 / (current_time - prev_time)
            prev_time = current_time

            frame_rgb = stream_buffer.frame
            frame_gbr = frame_rgb[:, :, ::-1]
            frame_rgb = cv2.resize(frame_rgb, dsize=(640, 480), interpolation=cv2.INTER_CUBIC)
            input_image = torch.tensor(frame_rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

            output = model.infer(input_image)
            """
            `output` contains the final prediction. Pass `return_per_step=True` to also return every refinement step.
            All maps have the same height and width as the input image.
            {
              "points": (H, W, 3),                  # final metric point map in OpenCV camera coordinates (x right, y down, z forward)
              "depth": (H, W),                      # final metric depth map
              "intrinsics": (3, 3),                 # normalized camera intrinsics for the final prediction
              "mask": (H, W),                       # binary mask for valid pixels
              "normal": (H, W, 3),                 # normal map in OpenCV camera coordinates (optional)
            }
            With `return_per_step=True`, `points_per_step`, `depth_per_step`, and `intrinsics_per_step`
            contain `refine_steps + 1` entries, including the initial prediction.
            """
            depth = output['depth'].cpu().numpy()
            # pred_color = colorize_pred(depth, vmin=0, vmax=10, add_colorbar=True)

            # if show_depth_img:
            #     # Display FPS
            #     cv2.putText(pred_color,f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            #         1,(0, 255, 0),2)
            #     pred_color = cv2.resize(pred_color, (640, 480))
            #     cv2.imshow(
            #         f"{stream_type}", cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
            #     )
            #     # cv2.imshow("Livestream", img_after, cv2.COLOR_RGB2BGR)
            #     if cv2.waitKey(1) & 0xFF == ord("q"):
            #         stop_processing = True

            moge_points = output['points'].cpu().numpy()
            points_input = moge_points.astype(np.float32, copy=True)
            points_input[~output["mask"].cpu().numpy().astype(bool)] = np.nan

            args = parse_args()

            rotation, translation = camera_to_base_transform(camera_height=args.camera_height)
            esdf_result = pointcloud_to_esdf_pipeline(points_input, h_min=args.h_min, h_max=args.h_max,
                                                      R=rotation, t=translation,
                                                      x_min=args.x_min, x_max=args.x_max,
                                                      y_min=args.y_min, y_max=args.y_max,
                                                      )

            # esdf_surface = visualize_path(depth=depth, rgb=frame_rgb,
            #                               esdf_result=esdf_result, idx=frame_idx, args=args)
            dummy_path = Path(video_path)
            esdf_surface = save_debug_figure(pointcloud_path=dummy_path, rgb=frame_rgb,
                                             metadata=None, full_image_size=None,
                                             result=esdf_result, args=args)
            window_name = "esdf_surface"
            # Create a resizable window
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

            # Set fixed width x height (e.g., 640x480 or 1280x720)
            cv2.resizeWindow(window_name, 1280, 720)
            # Display FPS
            cv2.putText(esdf_surface,f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1,(0, 255, 0),2)
            cv2.imshow(
                window_name, cv2.cvtColor(esdf_surface, cv2.COLOR_RGB2BGR)
            )
            cv2.waitKey(1) # cv2.waitKey(1) if running in a real-time loop
            # Update running statistics
            frame_idx += 1
            frame_timestamps.append(time.time())
            if save_video_toggle:
                video_frames.append(cv2.cvtColor(esdf_surface, cv2.COLOR_RGB2BGR))
            current_peak_memory = torch.cuda.max_memory_allocated() / 1024 ** 3  # GB
            peak_memory = max(peak_memory, current_peak_memory)
            print(
                f"Processed frame {frame_idx}. "
                f"Current peak memory: {current_peak_memory:.2f} GB, "
                f"Overall peak memory: {peak_memory:.2f} GB.",
                end="\r",
            )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt received, stopping processing gracefully...")

    finally:
        # Source cleanup
        src.close()

        if save_video_toggle:
            if len(frame_timestamps) >= 2:
                elapsed = frame_timestamps[-1] - frame_timestamps[0]
                # Use average FPS over the whole run
                effective_fps = (len(frame_timestamps) - 1) / elapsed
            else:
                effective_fps = 30.0

            output_dir = f"{output_folder}/{stream_type}.mp4"
            save_depth_video_mp4(
                video=np.array(video_frames),
                path=output_dir,
                fps=4,
                # fps=effective_fps,
            )
            print(
                f"\nSaved video to {output_dir} at {effective_fps:.2f} FPS."
            )

        # Close any OpenCV windows
        if show_depth_img:
            cv2.destroyAllWindows()

        print(f"Processed {frame_idx} frames.")
        print(f"Peak GPU memory usage: {peak_memory:.2f} GB.")
if __name__ == "__main__":
    main()