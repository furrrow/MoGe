"""
see https://github.com/DepthAnything/Depth-Anything-V2/blob/main/metric_depth/depth_to_pointcloud.py
although not using the metric version of DA2


"""
import time
import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
from custom_utils.stream_handler import FrameStatus, InputStreamHandler
from custom_utils.io_utils import save_depth_video_mp4, plot_bbox
import supervision as sv
import argparse
from argparse import Namespace

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="argparse for bytetracker."
    )
    parser.add_argument("--var_x", type=float, default=1, help="some var.")
    parser.add_argument("--track-thresh", type=float, default=0.65, help="track thresh")
    parser.add_argument("--match-thresh", type=float, default=0.7, help="track thresh")
    parser.add_argument("--track-buffer", type=int, default=15, help="track thresh")
    parser.add_argument("--mot20", type=bool, default=True, help="mot20 benchmark, no confidence fusion if true")
    return parser.parse_args()


def filter_unwanted_results(bbox_result, img_w, img_h):
    total_img_area = img_w * img_h
    filtered_results = {
        'bboxes': [],
        'labels': []
    }
    for bbox, label in zip(bbox_result['bboxes'], bbox_result['labels']):
        x1, y1, x2, y2 = bbox
        box_area = (x2 - x1) * (y2 - y1)
        if (total_img_area * 0.01 ) < box_area < (total_img_area * 0.8 ):
            filtered_results['bboxes'].append(bbox)
            filtered_results['labels'].append(label)
    return filtered_results

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
    tracker = sv.ByteTrack()
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
            img_w, img_h = 640, 480
            frame_rgb = cv2.resize(frame_rgb, dsize=(img_w, img_h), interpolation=cv2.INTER_CUBIC)
            input_image = torch.tensor(frame_rgb / 255, dtype=torch.float32, device=device).permute(2, 0, 1)

            task_prompt = "<CAPTION_TO_PHRASE_GROUNDING>"
            text_prompt = "people"
            prompt = task_prompt + text_prompt
            pil_image = Image.fromarray(frame_rgb)
            obj_detect_inputs = processor(text=prompt, images=frame_rgb, return_tensors="pt").to(device, torch_dtype)

            generated_ids = obj_detect_model.generate(
                input_ids=obj_detect_inputs["input_ids"],
                pixel_values=obj_detect_inputs["pixel_values"],
                max_new_tokens=4096,
                num_beams=3,
                do_sample=False
            )
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            obj_detect_result = processor.post_process_generation(generated_text, task=task_prompt,
                                                              image_size=(pil_image.width, pil_image.height))
            bbox_result=obj_detect_result[task_prompt]
            bbox_result = filter_unwanted_results(bbox_result, img_w, img_h)
            # bbox_result format: dict_keys(['bboxes', 'labels'])
            # bboxes: [[x1, y1, x2, y2]...] labels: ['people' ...]
            bbox_only = [bbox for bbox, label in zip(bbox_result['bboxes'], bbox_result['labels'])]
            if len(bbox_only) > 0:
                dummy_confidence = np.ones(len(bbox_only)) * 0.7
                sv_detection = sv.Detections(xyxy=np.array(bbox_only), confidence=dummy_confidence)
                detections = tracker.update_with_detections(sv_detection)
                print(detections)
                pred_color = plot_bbox(frame_rgb, bbox_result, detections.tracker_id, show_plot=False, return_img=True)
            else:
                pred_color = frame_rgb
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