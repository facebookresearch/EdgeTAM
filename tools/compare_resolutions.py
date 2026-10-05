# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Compare EdgeTAM at lower input sizes against 1024 on your own video.

Example (a folder of JPEG frames named 00000.jpg, 00001.jpg, ...):
    python tools/compare_resolutions.py --video_dir /data/clip01 \
        --box 120 80 260 300 --sizes 1024 768 512
"""

import argparse
import time

import numpy as np
import torch

from sam2.build_sam import build_sam2_video_predictor

CONFIGS = {
    1024: "configs/edgetam.yaml",
    768: "configs/edgetam_768.yaml",
    512: "configs/edgetam_512.yaml",
}


def track(cfg, args, prompts):
    predictor = build_sam2_video_predictor(cfg, args.checkpoint, device=args.device)
    cuda = args.device.startswith("cuda")
    # bf16 on GPU, as in the README examples
    with torch.inference_mode(), torch.autocast("cuda", torch.bfloat16, enabled=cuda):
        state = predictor.init_state(video_path=args.video_dir)
        for obj_id, kind, value in prompts:
            if kind == "box":
                predictor.add_new_points_or_box(
                    state, frame_idx=0, obj_id=obj_id, box=np.array(value, np.float32)
                )
            else:
                predictor.add_new_points_or_box(
                    state,
                    frame_idx=0,
                    obj_id=obj_id,
                    points=np.array([value], np.float32),
                    labels=np.array([1], np.int32),
                )
        n = min(state["num_frames"], args.max_frames)
        masks = np.zeros(
            (n, len(prompts), state["video_height"], state["video_width"]), bool
        )
        times = []
        t0 = time.perf_counter()
        for frame_idx, _, logits in predictor.propagate_in_video(
            state, max_frame_num_to_track=n - 1
        ):
            if cuda:
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            if frame_idx > 0:  # frame 0 was decoded when adding the prompts
                times.append(t1 - t0)
            masks[frame_idx] = (logits[:, 0] > 0.0).cpu().numpy()
            t0 = t1
    # skip warm-up frames while the memory bank fills
    return masks, 1000 * float(np.mean(times[6:] or times))


def iou(a, b):
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else np.logical_and(a, b).sum() / union


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video_dir", required=True, help="folder of JPEG frames")
    parser.add_argument("--checkpoint", default="checkpoints/edgetam.pt")
    parser.add_argument(
        "--box",
        nargs=4,
        type=float,
        action="append",
        default=[],
        metavar=("X0", "Y0", "X1", "Y1"),
        help="box prompt on frame 0 (repeatable)",
    )
    parser.add_argument(
        "--point",
        nargs=2,
        type=float,
        action="append",
        default=[],
        metavar=("X", "Y"),
        help="positive click on frame 0 (repeatable)",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, default=[1024, 512], choices=sorted(CONFIGS)
    )
    parser.add_argument("--max_frames", type=int, default=100)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    prompts = [(i + 1, "box", b) for i, b in enumerate(args.box)]
    prompts += [(len(prompts) + i + 1, "point", p) for i, p in enumerate(args.point)]
    if not prompts:
        parser.error("give at least one --box or --point")
    sizes = sorted(set(args.sizes) | {1024}, reverse=True)

    results = {s: track(CONFIGS[s], args, prompts) for s in sizes}
    ref, ref_ms = results[1024]
    print(
        f"\n{'size':>5s} {'ms/frame':>9s} {'speedup':>8s}  mean IoU vs 1024 per object"
    )
    for s in sizes:
        masks, ms = results[s]
        per_obj = [
            np.mean([iou(ref[t, j], masks[t, j]) for t in range(1, len(ref))])
            for j in range(len(prompts))
        ]
        print(
            f"{s:5d} {ms:9.0f} {ref_ms / ms:7.2f}x  "
            + "  ".join(f"{v:.3f}" for v in per_obj)
        )


if __name__ == "__main__":
    main()
