"""
Interactive Metric Depth Estimation Viewer using Apple's DepthPro.
Provides real-time 3D coordinate estimation, distance measurement, panning, and zooming.
"""

import os
import sys
import time
import tkinter as tk
from dataclasses import dataclass, field
from tkinter import filedialog
from typing import List, Tuple, Optional, Any

import cv2
import numpy as np
import torch
from depth_pro import create_model_and_transforms, load_rgb
from scripts.ckpoint import OUT, main as load_checkpoint

# --- Constants ---
WINDOW_NAME = "DepthPro Interactive Viewer"
DEFAULT_WIN_WIDTH = 1280
DEFAULT_WIN_HEIGHT = 720


# --- Architecture: GUI State ---
@dataclass
class AppState:
    """Maintains the interactive application state."""
    image_bgr: np.ndarray
    depth_colored_bgr: np.ndarray
    depth_map: np.ndarray
    focal_length: float
    view_mode: str = "RGB"  # "RGB" or "DEPTH"
    scale: float = 1.0
    tx: float = 0.0
    ty: float = 0.0
    points: List[Tuple[int, int, float]] = field(default_factory=list)  # (x, y, depth)
    hover_pos: Optional[Tuple[int, int]] = None  # (x, y) in image space
    is_panning: bool = False
    pan_start_w: Tuple[int, int] = (0, 0)
    pan_start_t: Tuple[float, float] = (0.0, 0.0)
    show_overlay: bool = True


# --- Architecture: Model Loading & Inference ---
def run_inference(image_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Loads the DepthPro model and handles the single-pass metric depth estimation loop on CPU."""
    print("Loading DepthPro model on CPU... (this may take a moment)")
    device = torch.device("cpu")
    model, transform = create_model_and_transforms(
        device=device,
        precision=torch.float32,
    )
    model.eval()

    print(f"Processing image: {image_path}")
    image_tensor, _, f_px = load_rgb(image_path)
    
    with torch.no_grad():
        prediction = model.infer(transform(image_tensor), f_px=f_px)
    
    depth = prediction["depth"].detach().cpu().numpy().squeeze()
    focal = float(prediction["focallength_px"])

    # Load original image using OpenCV for native BGR visualization rendering
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Could not read the image using OpenCV: {image_path}")

    h, w = img_bgr.shape[:2]
    # Ensure depth dimensions perfectly match the target visualization layout
    if depth.shape[-2:] != (h, w):
        depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

    # Compute inverse depth color mapping matching Apple's normalized presentation
    inv_depth = 1.0 / np.maximum(depth, 1e-5)
    d_min, d_max = np.min(inv_depth), np.max(inv_depth)
    if d_max > d_min:
        norm_depth = (inv_depth - d_min) / (d_max - d_min)
    else:
        norm_depth = np.zeros_like(inv_depth)
        
    depth_uint8 = (norm_depth * 255).astype(np.uint8)
    depth_colored_bgr = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)

    return img_bgr, depth_colored_bgr, depth, focal


# --- Architecture: Math Utilities ---
def compute_3d_coordinates(x: float, y: float, depth_val: float, w: int, h: int, focal: float) -> Tuple[float, float, float]:
    """Converts 2D pixel coordinates and depth values into real-world 3D metric coordinates."""
    cx = w / 2.0
    cy = h / 2.0
    X = (x - cx) * depth_val / focal
    Y = (y - cy) * depth_val / focal
    Z = depth_val
    return X, Y, Z


# --- Architecture: Mouse Callbacks ---
def mouse_callback(event: int, x: int, y: int, flags: int, param: Any) -> None:
    """Handles window-space coordinates transformation, clicks, pan events, and zoom steps."""
    state: AppState = param
    wx, wy = x, y

    # Compute reverse transformation matrix mapping window coordinates back to source canvas
    ix = int(round((wx - state.tx) / state.scale))
    iy = int(round((wy - state.ty) / state.scale))

    h, w = state.image_bgr.shape[:2]
    if 0 <= ix < w and 0 <= iy < h:
        state.hover_pos = (ix, iy)
    else:
        state.hover_pos = None

    if event == cv2.EVENT_LBUTTONDOWN:
        if 0 <= ix < w and 0 <= iy < h:
            depth_val = state.depth_map[iy, ix]
            state.points.append((ix, iy, depth_val))

    elif event == cv2.EVENT_MBUTTONDOWN:
        state.is_panning = True
        state.pan_start_w = (wx, wy)
        state.pan_start_t = (state.tx, state.ty)

    elif event == cv2.EVENT_MOUSEMOVE:
        if state.is_panning:
            dwx = wx - state.pan_start_w[0]
            dwy = wy - state.pan_start_w[1]
            state.tx = state.pan_start_t[0] + dwx
            state.ty = state.pan_start_t[1] + dwy

    elif event == cv2.EVENT_MBUTTONUP:
        state.is_panning = False

    elif event == cv2.EVENT_MOUSEWHEEL:
        # Determine movement scaling factor based on native hardware scroll steps
        zoom_factor = 1.1 if flags > 0 else 0.9
        old_scale = state.scale
        new_scale = max(0.05, min(old_scale * zoom_factor, 100.0))

        # Adjust anchor origins dynamically to track the pointer hotspot
        ix_f = (wx - state.tx) / old_scale
        iy_f = (wy - state.ty) / old_scale

        state.scale = new_scale
        state.tx = wx - ix_f * new_scale
        state.ty = wy - iy_f * new_scale


# --- Architecture: Rendering Functions ---
def draw_overlay_panel(frame: np.ndarray, state: AppState, w: int, h: int) -> None:
    """Draws a clean semi-transparent metrics and controls panel."""
    overlay = frame.copy()
    panel_w, panel_h = 350, 270
    cv2.rectangle(overlay, (10, 10), (panel_w, panel_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    hover_depth_str = "N/A"
    if state.hover_pos is not None:
        hx, hy = state.hover_pos
        hover_depth_str = f"{state.depth_map[hy, hx]:.3f} m"

    infos = [
        f"Resolution: {w}x{h}",
        f"Focal Length: {state.focal_length:.1f} px",
        f"Cursor Depth: {hover_depth_str}",
        f"Points Selected: {len(state.points)}",
        f"Mode: {state.view_mode} (TAB to switch)",
        "------------------------------------",
        "[L-Click] Place Metric Point",
        "[M-Drag] Pan Canvas  [Wheel] Zoom",
        "[u] Undo Last Point  [c] Clear All",
        "[s] Save Display Output  [h] Toggle Info",
        "[q] Exit Application"
    ]

    for i, text in enumerate(infos):
        color = (200, 255, 200) if "Mode:" in text or "Depth:" in text else (255, 255, 255)
        if "[" in text:
            color = (180, 180, 180)
        cv2.putText(frame, text, (20, 32 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, lineType=cv2.LINE_AA)


def render_scene(state: AppState, win_w: int, win_h: int) -> np.ndarray:
    """Assembles transformations, targets markers, links pairs, and prints dashboard elements."""
    base_frame = state.image_bgr if state.view_mode == "RGB" else state.depth_colored_bgr
    h, w = base_frame.shape[:2]

    # Generate affine pixel positioning grid matching active parameters
    M = np.float32([[state.scale, 0, state.tx], [0, state.scale, state.ty]])
    display_frame = cv2.warpAffine(
        base_frame, M, (win_w, win_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(40, 40, 40)
    )

    # Render persistent depth query markers on screen coordinates
    for idx, (ix, iy, d) in enumerate(state.points):
        wx = int(round(ix * state.scale + state.tx))
        wy = int(round(iy * state.scale + state.ty))
        
        if 0 <= wx < win_w and 0 <= wy < win_h:
            cv2.circle(display_frame, (wx, wy), 5, (0, 0, 255), -1, lineType=cv2.LINE_AA)
            cv2.circle(display_frame, (wx, wy), 7, (255, 255, 255), 1, lineType=cv2.LINE_AA)
            cv2.putText(display_frame, str(idx + 1), (wx + 10, wy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2, lineType=cv2.LINE_AA)
            cv2.putText(display_frame, str(idx + 1), (wx + 10, wy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Compute paired distance measurements if exactly two elements exist
    if len(state.points) == 2:
        p1, p2 = state.points[0], state.points[1]
        wx1 = int(round(p1[0] * state.scale + state.tx))
        wy1 = int(round(p1[1] * state.scale + state.ty))
        wx2 = int(round(p2[0] * state.scale + state.tx))
        wy2 = int(round(p2[1] * state.scale + state.ty))

        cv2.line(display_frame, (wx1, wy1), (wx2, wy2), (0, 255, 255), 2, lineType=cv2.LINE_AA)

        pix_dist = np.hypot(p1[0] - p2[0], p1[1] - p2[1])
        depth_diff = abs(p1[2] - p2[2])

        X1, Y1, Z1 = compute_3d_coordinates(p1[0], p1[1], p1[2], w, h, state.focal_length)
        X2, Y2, Z2 = compute_3d_coordinates(p2[0], p2[1], p2[2], w, h, state.focal_length)
        dist_3d = np.sqrt((X1 - X2)**2 + (Y1 - Y2)**2 + (Z1 - Z2)**2)

        mx, my = (wx1 + wx2) // 2, (wy1 + wy2) // 2
        metrics = [
            f"Pixel: {pix_dist:.1f} px",
            f"Depth: {depth_diff:.3f} m",
            f"3D: {dist_3d:.3f} m"
        ]

        for i, text in enumerate(metrics):
            tx_offset, ty_offset = mx + 15, my + (i * 18) - 10
            cv2.putText(display_frame, text, (tx_offset, ty_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, lineType=cv2.LINE_AA)
            cv2.putText(display_frame, text, (tx_offset, ty_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, lineType=cv2.LINE_AA)

    # Hover cursor localized readout configuration
    if state.hover_pos is not None:
        ix, iy = state.hover_pos
        d_val = state.depth_map[iy, ix]
        wx_m = int(round(ix * state.scale + state.tx))
        wy_m = int(round(iy * state.scale + state.ty))
        hover_text = f"({ix}, {iy}): {d_val:.3f}m"
        
        cv2.putText(display_frame, hover_text, (wx_m + 15, wy_m - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, lineType=cv2.LINE_AA)
        cv2.putText(display_frame, hover_text, (wx_m + 15, wy_m - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

    # Attach interactive text dashboard panel
    if state.show_overlay:
        draw_overlay_panel(display_frame, state, w, h)

    return display_frame


# --- Architecture: Application Loop ---
def main() -> None:
    """Orchestrates configuration checks, launches file managers, and executes UI render loop updates."""
    # Check command-line arguments for image input
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        # Launch fallback TK file dialogue window if arguments are missing
        root = tk.Tk()
        root.withdraw()
        image_path = filedialog.askopenfilename(
            title="Select Source Image for DepthPro Inference",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if not image_path:
            print("No image provided. Exiting.")
            sys.exit(0)

    if not os.path.exists(image_path):
        print(f"Error: Specified path target does not exist: {image_path}")
        sys.exit(1)

    if not OUT.exists():load_checkpoint()
    # Inference step running strictly on CPU
    try:
        img_bgr, depth_colored, depth_map, focal_length = run_inference(image_path)
    except Exception as e:
        print(f"Fatal implementation failure during inference: {e}")
        sys.exit(1)

    # Configure viewport initialization dimensions
    h, w = img_bgr.shape[:2]
    win_w = min(DEFAULT_WIN_WIDTH, w)
    win_h = min(DEFAULT_WIN_HEIGHT, h)

    # Match bounding scale presets centering asset mapping securely within workspace bounds
    init_scale = min(win_w / w, win_h / h)
    init_tx = (win_w - w * init_scale) / 2.0
    init_ty = (win_h - h * init_scale) / 2.0

    app_state = AppState(
        image_bgr=img_bgr,
        depth_colored_bgr=depth_colored,
        depth_map=depth_map,
        focal_length=focal_length,
        scale=init_scale,
        tx=init_tx,
        ty=init_ty
    )

    # Instantiate named platform layout window elements
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, win_w, win_h)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, app_state)

    print("\nViewer operational. Focus window and use instructions on-screen overlay.")

    while True:
        # Dynamically sample container transformations to secure layout aspects cleanly
        window_rect = cv2.getWindowImageRect(WINDOW_NAME)
        if window_rect is not None and window_rect[2] > 0 and window_rect[3] > 0:
            win_w, win_h = window_rect[2], window_rect[3]

        # Compute canvas visualization arrays
        display_buffer = render_scene(app_state, win_w, win_h)
        cv2.imshow(WINDOW_NAME, display_buffer)

        key = cv2.waitKey(1) & 0xFF

        # --- Architecture: Keyboard Shortcuts Handling ---
        if key == ord('q'):
            break
        elif key == 9:  # TAB Key code representation ASCII
            app_state.view_mode = "DEPTH" if app_state.view_mode == "RGB" else "RGB"
        elif key == ord('c'):
            app_state.points.clear()
        elif key == ord('u'):
            if app_state.points:
                app_state.points.pop()
        elif key == ord('h'):
            app_state.show_overlay = not app_state.show_overlay
        elif key == ord('s'):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            out_name = f"depthpro_snapshot_{timestamp}.png"
            cv2.imwrite(out_name, display_buffer)
            print(f"Captured screen frame saved: {out_name}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()