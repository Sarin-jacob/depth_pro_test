import cv2
import numpy as np
import torch
import tempfile
import os
import tkinter as tk
from tkinter import ttk
from PIL import Image
from transformers import pipeline

# ==========================================
# DEFAULT STARTUP CONFIGURATION
# ==========================================
DEFAULT_MODEL = "Depth Anything V2 Small" # "Depth Anything V2 Small", "Depth Anything V2 Base", "Depth Pro"
DEFAULT_FILTER = "CLAHE"                  # "None", "Grayscale", "CLAHE", "Red Compensation"
DEFAULT_VIEW_MODE = "Side-by-Side"        # "Side-by-Side", "Overlay", "RGB Only", "Depth Only"
DEFAULT_SCALE = 1.0                       # Real-time metric scale multiplier
DEFAULT_ALPHA = 0.5                       # Overlay opacity (0.0 to 1.0)
# ==========================================

try:
    import depth_pro
    DEPTH_PRO_AVAILABLE = True
except ImportError:
    DEPTH_PRO_AVAILABLE = False
    print("Warning: Apple 'depth_pro' module not found. It will be disabled in the dropdown.")

class UnderwaterDepthApp:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source}")

        # Retrieve original dimensions
        self.orig_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.orig_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Video Source Resolution: {self.orig_w}x{self.orig_h}")

        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Compute device: {self.device}")

        # Active State Variables
        self.current_model_name = ""
        self.filter_name = DEFAULT_FILTER
        self.view_mode = DEFAULT_VIEW_MODE
        self.scale_factor = DEFAULT_SCALE
        self.alpha = DEFAULT_ALPHA
        self.show_ui = True

        # Mouse Probe State
        self.mouse_x = 0
        self.mouse_y = 0
        self.metric_depth_map = None

        # Pipelines
        self.hf_pipe = None
        self.dp_model = None
        self.dp_transform = None

        # Store view options for toggling
        self.view_options = ["Side-by-Side", "Overlay", "RGB Only", "Depth Only"]

        # Build GUI Control Panel
        self.setup_control_panel()

        # Build OpenCV Display Window
        self.window_name = "Underwater Depth Estimation (Press 'H' to toggle text, 'V' to toggle view)"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        # Load Initial Model
        self.load_model(DEFAULT_MODEL)

    def setup_control_panel(self):
        """Creates a Tkinter Control Window with Dropdowns and Sliders."""
        self.root = tk.Tk()
        self.root.title("Depth App Controls")
        self.root.geometry("360x380")
        self.root.attributes('-topmost', True) # Keep controls on top

        padding = {'padx': 10, 'pady': 6}

        # 1. Model Dropdown
        tk.Label(self.root, text="Select Model:", font=('Helvetica', 10, 'bold')).pack(anchor='w', **padding)
        model_options = ["Depth Anything V2 Small", "Depth Anything V2 Base"]
        if DEPTH_PRO_AVAILABLE:
            model_options.append("Depth Pro")
        
        self.model_cb = ttk.Combobox(self.root, values=model_options, state="readonly")
        self.model_cb.set(DEFAULT_MODEL if DEFAULT_MODEL in model_options else model_options[0])
        self.model_cb.pack(fill='x', **padding)
        self.model_cb.bind("<<ComboboxSelected>>", lambda e: self.load_model(self.model_cb.get()))

        # 2. Filter Dropdown
        tk.Label(self.root, text="Preprocessing Filter:", font=('Helvetica', 10, 'bold')).pack(anchor='w', **padding)
        filter_options = ["None", "Grayscale", "CLAHE", "Red Compensation"]
        self.filter_cb = ttk.Combobox(self.root, values=filter_options, state="readonly")
        self.filter_cb.set(DEFAULT_FILTER)
        self.filter_cb.pack(fill='x', **padding)
        self.filter_cb.bind("<<ComboboxSelected>>", lambda e: setattr(self, 'filter_name', self.filter_cb.get()))

        # 3. View Mode Dropdown
        tk.Label(self.root, text="Display View Mode:", font=('Helvetica', 10, 'bold')).pack(anchor='w', **padding)
        self.view_cb = ttk.Combobox(self.root, values=self.view_options, state="readonly")
        self.view_cb.set(DEFAULT_VIEW_MODE)
        self.view_cb.pack(fill='x', **padding)
        self.view_cb.bind("<<ComboboxSelected>>", lambda e: setattr(self, 'view_mode', self.view_cb.get()))

        # 4. Overlay Opacity Slider
        tk.Label(self.root, text="Overlay Opacity (Alpha):", font=('Helvetica', 10, 'bold')).pack(anchor='w', **padding)
        self.alpha_slider = tk.Scale(self.root, from_=0.0, to=1.0, resolution=0.05, orient='horizontal',
                                     command=lambda v: setattr(self, 'alpha', float(v)))
        self.alpha_slider.set(DEFAULT_ALPHA)
        self.alpha_slider.pack(fill='x', **padding)

        # 5. Metric Scale Factor Slider
        tk.Label(self.root, text="Metric Scale Factor (Meters Multiplier):", font=('Helvetica', 10, 'bold')).pack(anchor='w', **padding)
        self.scale_slider = tk.Scale(self.root, from_=0.1, to=10.0, resolution=0.1, orient='horizontal',
                                     command=lambda v: setattr(self, 'scale_factor', float(v)))
        self.scale_slider.set(DEFAULT_SCALE)
        self.scale_slider.pack(fill='x', **padding)

    def load_model(self, model_name):
        if model_name == self.current_model_name:
            return

        print(f"\nLoading {model_name}... Please wait.")
        self.hf_pipe = None
        self.dp_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if "Depth Anything V2" in model_name:
            model_id = "depth-anything/Depth-Anything-V2-Small-hf" if "Small" in model_name else "depth-anything/Depth-Anything-V2-Base-hf"
            self.hf_pipe = pipeline(task="depth-estimation", model=model_id, device=self.device)
        elif model_name == "Depth Pro" and DEPTH_PRO_AVAILABLE:
            self.dp_model, self.dp_transform = depth_pro.create_model_and_transforms(
                device=self.device, precision=torch.float16
            )
            self.dp_model.eval()

        self.current_model_name = model_name
        print(f"Active model: {model_name}")

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y

    def apply_filter(self, frame):
        if self.filter_name == "None":
            return frame
        elif self.filter_name == "Grayscale":
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif self.filter_name == "CLAHE":
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif self.filter_name == "Red Compensation":
            img = frame.copy().astype(np.float32)
            avg_b, avg_g, avg_r = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
            avg = (avg_b + avg_g + avg_r) / 3.0
            img[:,:,0] *= (avg / (avg_b + 1e-5))
            img[:,:,1] *= (avg / (avg_g + 1e-5))
            img[:,:,2] *= (avg / (avg_r + 1e-5))
            return np.clip(img, 0, 255).astype(np.uint8)
        return frame

    def process_depth(self, frame_rgb):
        if "Depth Anything V2" in self.current_model_name:
            pil_img = Image.fromarray(frame_rgb)
            result = self.hf_pipe(pil_img)
            raw_disparity = np.array(result["depth"]).astype(np.float32)

            # Normalize disparity [0, 1] (higher = closer)
            d_min, d_max = raw_disparity.min(), raw_disparity.max()
            norm_disparity = (raw_disparity - d_min) / (d_max - d_min + 1e-6)

            # Invert to Metric Distance: d = scale / (disparity + eps)
            # Farther objects -> lower disparity -> LARGER distance value (meters)
            # Closer objects -> higher disparity -> SMALLER distance value (meters)
            self.metric_depth_map = self.scale_factor / (norm_disparity + 0.05)

            # Color mapping
            depth_vis = cv2.normalize(norm_disparity, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)

        elif self.current_model_name == "Depth Pro" and DEPTH_PRO_AVAILABLE:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                Image.fromarray(frame_rgb).save(tmp.name)
                tmp_path = tmp.name

            try:
                image, _, f_px = depth_pro.load_rgb(tmp_path)
                image = self.dp_transform(image).to(self.device)

                with torch.no_grad():
                    prediction = self.dp_model.infer(image, f_px=f_px)
                    raw_depth_meters = prediction["depth"].cpu().numpy()

                # Metric depth = physical meters * user scale factor
                self.metric_depth_map = raw_depth_meters * self.scale_factor

                # Color mapping
                inv_depth = 1.0 / (raw_depth_meters + 1e-6)
                depth_vis = cv2.normalize(inv_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                return cv2.applyColorMap(depth_vis, cv2.COLORMAP_INFERNO)
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    def run(self):
        print("Running application...")
        print("Press 'H' in the video window to hide/show on-screen text overlays for clean screenshots.")

        while True:
            # Process Tkinter Events
            self.root.update_idletasks()
            self.root.update()

            ret, frame = self.cap.read()
            if not ret:
                print("End of video stream or camera disconnected.")
                break

            # Frame stays at native resolution
            h, w = frame.shape[:2]

            filtered_frame = self.apply_filter(frame)
            frame_rgb = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)
            depth_colormap = self.process_depth(frame_rgb)

            # View Modes
            if self.view_mode == "Side-by-Side":
                display_img = np.hstack((filtered_frame, depth_colormap))
            elif self.view_mode == "Overlay":
                display_img = cv2.addWeighted(filtered_frame, 1.0 - self.alpha, depth_colormap, self.alpha, 0)
            elif self.view_mode == "RGB Only":
                display_img = filtered_frame.copy()
            elif self.view_mode == "Depth Only":
                display_img = depth_colormap.copy()

            # Pointer Depth Probe Logic
            px, py = self.mouse_x, self.mouse_y
            valid_probe = False

            if self.view_mode == "Side-by-Side":
                if px >= w: px -= w # Remap right pane x-coord back to image width
                if 0 <= px < w and 0 <= py < h: valid_probe = True
            else:
                if 0 <= px < w and 0 <= py < h: valid_probe = True

            # Draw Overlays (If UI Toggle is ON)
            if self.show_ui:
                # Mode label
                cv2.putText(display_img, f"Mode: {self.view_mode} | Filter: {self.filter_name}", 
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

                if valid_probe and self.metric_depth_map is not None:
                    depth_val = self.metric_depth_map[py, px]
                    text = f"Depth: {depth_val:.2f} m"

                    tx = min(self.mouse_x + 15, display_img.shape[1] - 180)
                    ty = max(30, self.mouse_y)

                    # Text outline for high visibility
                    cv2.putText(display_img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(display_img, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(self.window_name, display_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.show_ui = not self.show_ui
            elif key == ord('v'):
                # Cycle view mode
                current_index = self.view_options.index(self.view_mode)
                next_index = (current_index + 1) % len(self.view_options)
                self.view_mode = self.view_options[next_index]
                # Sync Tkinter dropdown
                self.view_cb.set(self.view_mode)

        self.cap.release()
        cv2.destroyAllWindows()
        self.root.destroy()

if __name__ == "__main__":
    app = UnderwaterDepthApp(source=r"C:\Users\sarin\base\save\share\underwater\Leakage video.mp4")
    app.run()