import copy
import os
from datetime import datetime

import gradio as gr

os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "0,1,2,3,4,5,6,7"
import tempfile

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from moviepy.editor import ImageSequenceClip
from PIL import Image
from sam2.build_sam import build_sam2_video_predictor

# Description
title = "<center><strong><font size='8'>Edge Track Anything (EdgeTAM)<font></strong></center>"

description_e = """This is a demo of [Edge Track Anything (EdgeTAM) Model](https://github.com/facebookresearch/EdgeTAM).
              """

description_p = """# Interactive Video Segmentation
                - Built our demo based on [SAM2-Video-Predictor](https://huggingface.co/spaces/fffiloni/SAM2-Video-Predictor). Thanks to Sylvain Filoni.
                - Instruction
                <ol>
                <li> Upload one video or click one example video</li>
                <li> Click 'include' point type, select the object to segment and track</li>
                <li> Click 'exclude' point type (optional), select the area you want to avoid segmenting and tracking</li>
                <li> Click the 'Track' button to obtain the masked video </li>
                </ol>
                - Github [link](https://github.com/facebookresearch/EdgeTAM)
              """

# examples
examples = [
    ["examples/01_dog.mp4"],
    ["examples/02_cups.mp4"],
    ["examples/03_blocks.mp4"],
    ["examples/04_coffee.mp4"],
    ["examples/05_default_juggle.mp4"],
    ["examples/01_breakdancer.mp4"],
    ["examples/02_hummingbird.mp4"],
    ["examples/03_skateboarder.mp4"],
    ["examples/04_octopus.mp4"],
    ["examples/05_landing_dog_soccer.mp4"],
    ["examples/06_pingpong.mp4"],
    ["examples/07_snowboarder.mp4"],
    ["examples/08_driving.mp4"],
    ["examples/09_birdcartoon.mp4"],
    ["examples/10_cloth_magic.mp4"],
    ["examples/11_polevault.mp4"],
    ["examples/12_hideandseek.mp4"],
    ["examples/13_butterfly.mp4"],
    ["examples/14_social_dog_training.mp4"],
    ["examples/15_cricket.mp4"],
    ["examples/16_robotarm.mp4"],
    ["examples/17_childrendancing.mp4"],
    ["examples/18_threedogs.mp4"],
    ["examples/19_cyclist.mp4"],
    ["examples/20_doughkneading.mp4"],
    ["examples/21_biker.mp4"],
    ["examples/22_dogskateboarder.mp4"],
    ["examples/23_racecar.mp4"],
    ["examples/24_clownfish.mp4"],
]

OBJ_ID = 0

DEVICE = "mps"
sam2_checkpoint = "checkpoints/edgetam.pt"
model_cfg = "edgetam.yaml"
predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=DEVICE)
print("PREDICTOR LOADED")

torch.autocast(device_type=DEVICE, dtype=torch.float16).__enter__()

# use bfloat16 for the entire notebook
# torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()
# if torch.cuda.get_device_properties(0).major >= 8:
#     # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True


def get_video_fps(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return None

    # Get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    return fps


def reset(session_state):
    session_state["input_points"] = []
    session_state["input_labels"] = []
    if session_state["inference_state"] is not None:
        predictor.reset_state(session_state["inference_state"])
    session_state["first_frame"] = None
    session_state["all_frames"] = None
    session_state["inference_state"] = None
    return (
        None,
        gr.update(open=True),
        None,
        None,
        gr.update(value=None, visible=False),
        session_state,
    )


def clear_points(session_state):
    session_state["input_points"] = []
    session_state["input_labels"] = []
    if session_state["inference_state"]["tracking_has_started"]:
        predictor.reset_state(session_state["inference_state"])
    return (
        session_state["first_frame"],
        None,
        gr.update(value=None, visible=False),
        session_state,
    )


def preprocess_video_in(video_path, session_state):
    if video_path is None:
        return (
            gr.update(open=True),  # video_in_drawer
            None,  # points_map
            None,  # output_image
            gr.update(value=None, visible=False),  # output_video
            session_state,
        )

    # Read the first frame
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return (
            gr.update(open=True),  # video_in_drawer
            None,  # points_map
            None,  # output_image
            gr.update(value=None, visible=False),  # output_video
            session_state,
        )

    frame_number = 0
    first_frame = None
    all_frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = np.array(frame)

        # Store the first frame
        if frame_number == 0:
            first_frame = frame
        all_frames.append(frame)

        frame_number += 1

    cap.release()
    session_state["first_frame"] = copy.deepcopy(first_frame)
    session_state["all_frames"] = all_frames

    session_state["inference_state"] = predictor.init_state(video_path=video_path)
    session_state["input_points"] = []
    session_state["input_labels"] = []

    return [
        gr.update(open=False),  # video_in_drawer
        first_frame,  # points_map
        None,  # output_image
        gr.update(value=None, visible=False),  # output_video
        session_state,
    ]


def segment_with_points(
    point_type,
    session_state,
    evt: gr.SelectData,
):
    session_state["input_points"].append(evt.index)
    print(f"TRACKING INPUT POINT: {session_state['input_points']}")

    if point_type == "include":
        session_state["input_labels"].append(1)
    elif point_type == "exclude":
        session_state["input_labels"].append(0)
    print(f"TRACKING INPUT LABEL: {session_state['input_labels']}")

    # Open the image and get its dimensions
    transparent_background = Image.fromarray(session_state["first_frame"]).convert(
        "RGBA"
    )
    w, h = transparent_background.size

    # Define the circle radius as a fraction of the smaller dimension
    fraction = 0.01  # You can adjust this value as needed
    radius = int(fraction * min(w, h))

    # Create a transparent layer to draw on
    transparent_layer = np.zeros((h, w, 4), dtype=np.uint8)

    for index, track in enumerate(session_state["input_points"]):
        if session_state["input_labels"][index] == 1:
            cv2.circle(transparent_layer, track, radius, (0, 255, 0, 255), -1)
        else:
            cv2.circle(transparent_layer, track, radius, (255, 0, 0, 255), -1)

    # Convert the transparent layer back to an image
    transparent_layer = Image.fromarray(transparent_layer, "RGBA")
    selected_point_map = Image.alpha_composite(
        transparent_background, transparent_layer
    )

    # Let's add a positive click at (x, y) = (210, 350) to get started
    points = np.array(session_state["input_points"], dtype=np.float32)
    # for labels, `1` means positive click and `0` means negative click
    labels = np.array(session_state["input_labels"], np.int32)
    _, _, out_mask_logits = predictor.add_new_points(
        inference_state=session_state["inference_state"],
        frame_idx=0,
        obj_id=OBJ_ID,
        points=points,
        labels=labels,
    )

    mask_image = show_mask((out_mask_logits[0] > 0.0).cpu().numpy())
    first_frame_output = Image.alpha_composite(transparent_background, mask_image)

    torch.cuda.empty_cache()
    return selected_point_map, first_frame_output, session_state


def show_mask(mask, obj_id=None, random_color=False, convert_to_image=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    mask = (mask * 255).astype(np.uint8)
    if convert_to_image:
        mask = Image.fromarray(mask, "RGBA")
    return mask


def propagate_to_all(
    video_in,
    session_state,
):
    if (
        len(session_state["input_points"]) == 0
        or video_in is None
        or session_state["inference_state"] is None
    ):
        return (
            None,
            session_state,
        )

    # run propagation throughout the video and collect the results in a dict
    video_segments = {}  # video_segments contains the per-frame segmentation results
    print("starting propagate_in_video")
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(
        session_state["inference_state"]
    ):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    # obtain the segmentation results every few frames
    vis_frame_stride = 1

    output_frames = []
    for out_frame_idx in range(0, len(video_segments), vis_frame_stride):
        transparent_background = Image.fromarray(
            session_state["all_frames"][out_frame_idx]
        ).convert("RGBA")
        out_mask = video_segments[out_frame_idx][OBJ_ID]
        mask_image = show_mask(out_mask)
        output_frame = Image.alpha_composite(transparent_background, mask_image)
        output_frame = np.array(output_frame)
        output_frames.append(output_frame)

    torch.cuda.empty_cache()

    # Create a video clip from the image sequence
    original_fps = get_video_fps(video_in)
    fps = original_fps  # Frames per second
    clip = ImageSequenceClip(output_frames, fps=fps)
    # Write the result to a file
    unique_id = datetime.now().strftime("%Y%m%d%H%M%S")
    final_vid_output_path = f"output_video_{unique_id}.mp4"
    final_vid_output_path = os.path.join(tempfile.gettempdir(), final_vid_output_path)

    # Write the result to a file
    clip.write_videofile(final_vid_output_path, codec="libx264")

    return (
        gr.update(value=final_vid_output_path),
        session_state,
    )


def update_ui():
    return gr.update(visible=True)


with gr.Blocks() as demo:
    session_state = gr.State(
        {
            "first_frame": None,
            "all_frames": None,
            "input_points": [],
            "input_labels": [],
            "inference_state": None,
        }
    )

    with gr.Column():
        # Title
        gr.Markdown(title)
        with gr.Row():

            with gr.Column():
                # Instructions
                gr.Markdown(description_p)

                with gr.Accordion("Input Video", open=True) as video_in_drawer:
                    video_in = gr.Video(label="Input Video", format="mp4")

                with gr.Row():
                    point_type = gr.Radio(
                        label="point type",
                        choices=["include", "exclude"],
                        value="include",
                        scale=2,
                    )
                    propagate_btn = gr.Button("Track", scale=1, variant="primary")
                    clear_points_btn = gr.Button("Clear Points", scale=1)
                    reset_btn = gr.Button("Reset", scale=1)

                points_map = gr.Image(
                    label="Frame with Point Prompt", type="numpy", interactive=False
                )

            with gr.Column():
                gr.Markdown("# Try some of the examples below ⬇️")
                gr.Examples(
                    examples=examples,
                    inputs=[
                        video_in,
                    ],
                    examples_per_page=8,
                )
                gr.Markdown("\n\n\n\n\n\n\n\n\n\n\n")
                gr.Markdown("\n\n\n\n\n\n\n\n\n\n\n")
                gr.Markdown("\n\n\n\n\n\n\n\n\n\n\n")
                output_image = gr.Image(label="Reference Mask")

                output_video = gr.Video(visible=False)

    # When new video is uploaded
    video_in.upload(
        fn=preprocess_video_in,
        inputs=[
            video_in,
            session_state,
        ],
        outputs=[
            video_in_drawer,  # Accordion to hide uploaded video player
            points_map,  # Image component where we add new tracking points
            output_image,
            output_video,
            session_state,
        ],
        queue=False,
    )

    video_in.change(
        fn=preprocess_video_in,
        inputs=[
            video_in,
            session_state,
        ],
        outputs=[
            video_in_drawer,  # Accordion to hide uploaded video player
            points_map,  # Image component where we add new tracking points
            output_image,
            output_video,
            session_state,
        ],
        queue=False,
    )

    # triggered when we click on image to add new points
    points_map.select(
        fn=segment_with_points,
        inputs=[
            point_type,  # "include" or "exclude"
            session_state,
        ],
        outputs=[
            points_map,  # updated image with points
            output_image,
            session_state,
        ],
        queue=False,
    )

    # Clear every points clicked and added to the map
    clear_points_btn.click(
        fn=clear_points,
        inputs=session_state,
        outputs=[
            points_map,
            output_image,
            output_video,
            session_state,
        ],
        queue=False,
    )

    reset_btn.click(
        fn=reset,
        inputs=session_state,
        outputs=[
            video_in,
            video_in_drawer,
            points_map,
            output_image,
            output_video,
            session_state,
        ],
        queue=False,
    )

    propagate_btn.click(
        fn=update_ui,
        inputs=[],
        outputs=output_video,
        queue=False,
    ).then(
        fn=propagate_to_all,
        inputs=[
            video_in,
            session_state,
        ],
        outputs=[
            output_video,
            session_state,
        ],
        concurrency_limit=10,
        queue=False,
    )

demo.queue()
demo.launch()
