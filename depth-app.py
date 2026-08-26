import cv2
import numpy as np
import torch
import tempfile
import os
from PIL import Image
from transformers import pipeline

try:
    import depth_pro
    DEPTH_PRO_AVAILABLE = True
except ImportError:
    DEPTH_PRO_AVAILABLE = False
    print("Warning: Apple 'depth_pro' not found. It will be disabled in the UI.")

class UnderwaterDepthApp:
    def __init__(self, source=0):
        self.cap = cv2.VideoCapture(source)
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using compute device: {self.device}")
        
        # State variables
        self.filter_idx = 0 
        self.model_idx = -1
        self.view_mode = 0      # 0: Side-by-Side, 1: Overlay, 2: RGB Only, 3: Depth Only
        self.alpha = 50         # Overlay opacity (%)
        self.depth_scale = 10   # Multiplier (x0.1)
        self.show_ui = True     # Toggle for clean screenshots
        
        # Mouse probe state
        self.mouse_x = 0
        self.mouse_y = 0
        self.raw_depth = None
        
        self.hf_pipe = None
        self.dp_model = None
        self.dp_transform = None
        
        # UI Setup
        self.window_name = "Underwater Depth Estimation"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        
        # Trackbars
        cv2.createTrackbar("Filter", self.window_name, 0, 3, self.on_filter_change)
        max_model = 2 if DEPTH_PRO_AVAILABLE else 1
        cv2.createTrackbar("Model", self.window_name, 0, max_model, self.on_model_change)
        
        cv2.createTrackbar("View Mode", self.window_name, 0, 3, self.on_view_change)
        cv2.createTrackbar("Overlay %", self.window_name, 50, 100, self.on_alpha_change)
        cv2.createTrackbar("Scale(x0.1)", self.window_name, 10, 100, self.on_scale_change)
        
        # Initialize the first model
        self.on_model_change(0)

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            self.mouse_x = x
            self.mouse_y = y

    def on_filter_change(self, val): self.filter_idx = val
    def on_view_change(self, val): self.view_mode = val
    def on_alpha_change(self, val): self.alpha = val
    def on_scale_change(self, val): self.depth_scale = max(1, val) # Prevent zero

    def on_model_change(self, val):
        if val == self.model_idx:
            return
            
        print(f"\nSwapping to Model {val}... Please wait.")
        self.hf_pipe = None
        self.dp_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if val in [0, 1]:
            model_id = "depth-anything/Depth-Anything-V2-Small-hf" if val == 0 else "depth-anything/Depth-Anything-V2-Base-hf"
            self.hf_pipe = pipeline(task="depth-estimation", model=model_id, device=self.device)
        elif val == 2 and DEPTH_PRO_AVAILABLE:
            self.dp_model, self.dp_transform = depth_pro.create_model_and_transforms(
                device=self.device, precision=torch.float16
            )
            self.dp_model.eval()
            
        self.model_idx = val
        print("Model Loaded successfully!")

    def apply_filter(self, frame, f_idx):
        if f_idx == 0:
            return frame
        elif f_idx == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif f_idx == 2: # CLAHE
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif f_idx == 3: # Red Compensation
            img = frame.copy().astype(np.float32)
            avg_b, avg_g, avg_r = np.mean(img[:,:,0]), np.mean(img[:,:,1]), np.mean(img[:,:,2])
            avg = (avg_b + avg_g + avg_r) / 3
            img[:,:,0] *= (avg / (avg_b + 1e-5))
            img[:,:,1] *= (avg / (avg_g + 1e-5))
            img[:,:,2] *= (avg / (avg_r + 1e-5))
            return np.clip(img, 0, 255).astype(np.uint8)

    def process_depth(self, frame_rgb):
        if self.model_idx in [0, 1]:
            pil_img = Image.fromarray(frame_rgb)
            result = self.hf_pipe(pil_img)
            depth_arr = np.array(result["depth"]).astype(np.float32)
            self.raw_depth = depth_arr # DA-V2 outputs relative depth
            
            # Normalize for visualization
            depth_norm = cv2.normalize(depth_arr, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            
        elif self.model_idx == 2 and DEPTH_PRO_AVAILABLE:
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                Image.fromarray(frame_rgb).save(tmp.name)
                tmp_path = tmp.name
                
            try:
                image, _, f_px = depth_pro.load_rgb(tmp_path)
                image = self.dp_transform(image).to(self.device)
                
                with torch.no_grad():
                    prediction = self.dp_model.infer(image, f_px=f_px)
                    depth = prediction["depth"].cpu().numpy()
                
                self.raw_depth = depth # Depth Pro outputs absolute depth in meters
                
                inverse_depth = 1.0 / (depth + 1e-6)
                depth_norm = cv2.normalize(inverse_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            finally:
                os.remove(tmp_path)

    def run(self):
        print("Controls: Press 'Q' to quit, 'H' to toggle on-image UI text.")
        while True:
            ret, frame = self.cap.read()
            if not ret: break
                
            frame = cv2.resize(frame, (640, 480))
            h, w = frame.shape[:2]
            
            filtered_frame = self.apply_filter(frame, self.filter_idx)
            frame_rgb = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)
            depth_colormap = self.process_depth(frame_rgb)
            
            # Determine the display layout based on View Mode
            if self.view_mode == 0:
                display_img = np.hstack((filtered_frame, depth_colormap))
            elif self.view_mode == 1:
                alpha_val = self.alpha / 100.0
                display_img = cv2.addWeighted(filtered_frame, 1 - alpha_val, depth_colormap, alpha_val, 0)
            elif self.view_mode == 2:
                display_img = filtered_frame.copy()
            elif self.view_mode == 3:
                display_img = depth_colormap.copy()

            # Handle Depth Probing
            probe_x, probe_y = self.mouse_x, self.mouse_y
            valid_probe = False
            
            # Map mouse coordinates to the underlying image array
            if self.view_mode == 0:
                if probe_x >= w: probe_x -= w # Map to right-side image
                if 0 <= probe_x < w and 0 <= probe_y < h: valid_probe = True
            else:
                if 0 <= probe_x < w and 0 <= probe_y < h: valid_probe = True

            if self.show_ui:
                # Mode Labeling
                modes = ["Side-by-Side", "Overlay", "RGB Only", "Depth Only"]
                cv2.putText(display_img, modes[self.view_mode], (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Draw Depth Probe Text
                if valid_probe and self.raw_depth is not None:
                    scale = self.depth_scale / 10.0
                    d_val = self.raw_depth[probe_y, probe_x] * scale
                    
                    # Note: DA-V2 gives relative disparity (higher=closer). 
                    # Depth Pro gives metric depth (lower=closer). 
                    unit = "m" if self.model_idx == 2 else "units"
                    text = f"Depth: {d_val:.2f} {unit}"
                    
                    # Ensure text stays within bounds of the screen
                    text_x = min(self.mouse_x + 15, display_img.shape[1] - 150)
                    text_y = max(30, self.mouse_y)
                    
                    # Draw text with a black outline for readability against bright maps
                    cv2.putText(display_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                    cv2.putText(display_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow(self.window_name, display_img)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self.show_ui = not self.show_ui # Toggle clean frame
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = UnderwaterDepthApp(source=r"C:\Users\sarin\base\save\share\underwater\Leakage video.mp4")
    app.run()