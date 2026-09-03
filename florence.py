"""
see https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/depth_to_pointcloud.py
although not using the metric version of DA2


"""
import argparse
import os
import time
from pathlib import Path

import cv2
import torch
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

import open3d as o3d
from typing import Any, Dict, List, Optional, Sequence, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from moge.model.v2 import MoGeModel

from custom_utils.esdf_utils import parse_args, visualize_path_esdf, save_debug_figure
from custom_utils.stream_handler import FrameStatus, InputStreamHandler
from custom_utils.io_utils import colorize_pred, save_depth_video_mp4
from custom_utils.pointcloud_utils import camera_to_base_transform, pointcloud_to_esdf_pipeline


def plot_bbox(image, data, show_plot=True, return_img=False):
    # Create a figure and axes
    fig, ax = plt.subplots()

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for bbox, label in zip(data['bboxes'], data['labels']):
        # Unpack the bounding box coordinates
        x1, y1, x2, y2 = bbox
        # Create a Rectangle patch
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        # Add the rectangle to the Axes
        ax.add_patch(rect)
        # Annotate the label
        plt.text(x1, y1, label, color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

        # Remove the axis ticks and labels
    ax.axis('off')

    # Show the plot
    if show_plot:
        plt.show()
    if return_img:
        # Render the Matplotlib figure into an RGB NumPy array.
        fig.canvas.draw()
        image_rgb = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return image_rgb
    return None


def main():
    # Initialize predictor (single-GPU streaming)
    show_img = True
    save_video_toggle = False
    stream_type = "video" # ["yarp", "video", "webcam"]
    # video_path = "/home/jim/Projects/steernav/assets/Cars_and_Gasstation.mp4"
    video_path = "/home/jim/Projects/steernav/assets/jim_flownav_test.mp4"
    output_folder = "demo_video"
    webcam_index = 0
    yarp_port = "/sam3/rgbImage:i"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    # depth_model_name = "Ruicheng/moge-2-vitl-normal"
    vision_model_name = "microsoft/Florence-2-large"
    print("running", vision_model_name)
    # depth_model = MoGeModel.from_pretrained(model_name).to(device).eval()
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    obj_detect_model = AutoModelForCausalLM.from_pretrained(vision_model_name,
                                                            torch_dtype=torch_dtype,
                                                            attn_implementation="eager",
                                                            trust_remote_code=True).to(device)

    processor = AutoProcessor.from_pretrained(vision_model_name, trust_remote_code=True)
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

            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            text_input = "people"
            prompt = task_prompt + text_input
            pil_image = Image.fromarray(frame_rgb)
            inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device, torch_dtype)

            generated_ids = obj_detect_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=4096,
                num_beams=3,
                do_sample=False
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            parsed_answer = processor.post_process_generation(generated_text, task=task_prompt,
                                                              image_size=(pil_image.width, pil_image.height))

            # print(parsed_answer)
            pred_color = plot_bbox(pil_image, parsed_answer[task_prompt],
                                     show_plot=False, return_img=True)

            if show_img:
                # Display FPS
                cv2.putText(pred_color,f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1,(0, 255, 0),2)
                pred_color = cv2.resize(pred_color, (640, 480))
                cv2.imshow(
                    f"{stream_type}", cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR)
                )
                # cv2.imshow("Livestream", img_after, cv2.COLOR_RGB2BGR)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop_processing = True

            # moge_points = output['points'].cpu().numpy()
            # points_input = moge_points.astype(np.float32, copy=True)
            # points_input[~output["mask"].cpu().numpy().astype(bool)] = np.nan
            #
            # args = parse_args()
            #
            # rotation, translation = camera_to_base_transform(camera_height=args.camera_height)
            # esdf_result = pointcloud_to_esdf_pipeline(points_input, h_min=args.h_min, h_max=args.h_max,
            #                                           R=rotation, t=translation,
            #                                           x_min=args.x_min, x_max=args.x_max,
            #                                           y_min=args.y_min, y_max=args.y_max,
            #                                           )
            #
            # # esdf_surface = plot_esdf_surface(depth=depth, rgb=frame_rgb,
            # #                            result=esdf_result, idx=frame_idx, args=args)
            # dummy_path = Path(video_path)
            # esdf_surface = save_debug_figure(pointcloud_path=dummy_path, rgb=frame_rgb,
            #                                  metadata=None, full_image_size=None,
            #                                  result=esdf_result, args=args)
            # window_name = "esdf_surface"
            # # Create a resizable window
            # cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            #
            # # Set fixed width x height (e.g., 640x480 or 1280x720)
            # cv2.resizeWindow(window_name, 1280, 720)
            # # Display FPS
            # cv2.putText(esdf_surface,f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
            #     1,(0, 255, 0),2)
            # cv2.imshow(
            #     window_name, cv2.cvtColor(esdf_surface, cv2.COLOR_RGB2BGR)
            # )
            # cv2.waitKey(1) # cv2.waitKey(1) if running in a real-time loop
            # Update running statistics
            frame_idx += 1
            frame_timestamps.append(time.time())
            if save_video_toggle:
                video_frames.append(cv2.cvtColor(pred_color, cv2.COLOR_RGB2BGR))
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
        if show_img:
            cv2.destroyAllWindows()

        print(f"Processed {frame_idx} frames.")
        print(f"Peak GPU memory usage: {peak_memory:.2f} GB.")
if __name__ == "__main__":
    main()