import cv2
import numpy as np
import torch
import tempfile
import os
from PIL import Image
from transformers import pipeline

# Attempt to load Apple Depth Pro
try:
    import depth_pro
    DEPTH_PRO_AVAILABLE = True
except ImportError:
    DEPTH_PRO_AVAILABLE = False
    print("Warning: Apple 'depth_pro' not found. It will be disabled in the UI.")

class UnderwaterDepthApp:
    def __init__(self, source=0):
        # Initialize video feed (change 'source' to a file path like 'video.mp4' for recorded footage)
        self.cap = cv2.VideoCapture(source)
        self.device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        print(f"Using compute device: {self.device}")
        
        # State variables
        self.filter_idx = 0 
        self.model_idx = -1
        
        self.hf_pipe = None
        self.dp_model = None
        self.dp_transform = None
        
        # UI Setup
        cv2.namedWindow("Underwater Depth Estimation", cv2.WINDOW_NORMAL)
        
        # Filter Trackbar: 0=None, 1=Grayscale, 2=CLAHE, 3=Red Compensation
        cv2.createTrackbar("Filter", "Underwater Depth Estimation", 0, 3, self.on_filter_change)
        
        # Model Trackbar: 0=DA-Small, 1=DA-Base, 2=Depth Pro
        max_model = 2 if DEPTH_PRO_AVAILABLE else 1
        cv2.createTrackbar("Model", "Underwater Depth Estimation", 0, max_model, self.on_model_change)
        
        # Initialize the first model
        self.on_model_change(0)

    def on_filter_change(self, val):
        self.filter_idx = val

    def on_model_change(self, val):
        if val == self.model_idx:
            return
            
        print(f"\nSwapping to Model {val}... Please wait.")
        # Clear VRAM before loading the new model
        self.hf_pipe = None
        self.dp_model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        if val in [0, 1]:
            # Load Hugging Face Depth Anything V2
            model_id = "depth-anything/Depth-Anything-V2-Small-hf" if val == 0 else "depth-anything/Depth-Anything-V2-Base-hf"
            self.hf_pipe = pipeline(task="depth-estimation", model=model_id, device=self.device)
        elif val == 2 and DEPTH_PRO_AVAILABLE:
            # Load Apple Depth Pro
            self.dp_model, self.dp_transform = depth_pro.create_model_and_transforms(
                device=self.device, precision=torch.float16
            )
            self.dp_model.eval()
            
        self.model_idx = val
        print("Model Loaded successfully!")

    def apply_filter(self, frame, f_idx):
        """Applies preprocessing filters to the BGR frame."""
        if f_idx == 0:
            return frame
        elif f_idx == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        elif f_idx == 2:
            # CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # Highly effective for underwater scattering and haze
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            lab = cv2.merge((cl, a, b))
            return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif f_idx == 3:
            # Simple Underwater Red Compensation (Gray World Assumption)
            # Restores the red spectrum absorbed by water depths
            img = frame.copy().astype(np.float32)
            avg_b = np.mean(img[:,:,0])
            avg_g = np.mean(img[:,:,1])
            avg_r = np.mean(img[:,:,2])
            avg = (avg_b + avg_g + avg_r) / 3
            
            img[:,:,0] *= (avg / (avg_b + 1e-5))
            img[:,:,1] *= (avg / (avg_g + 1e-5))
            img[:,:,2] *= (avg / (avg_r + 1e-5))
            return np.clip(img, 0, 255).astype(np.uint8)

    def get_depth_map(self, frame_rgb):
        """Passes the processed frame through the active neural network."""
        if self.model_idx in [0, 1]:
            pil_img = Image.fromarray(frame_rgb)
            result = self.hf_pipe(pil_img)
            depth_img = np.array(result["depth"])
            
            # Normalize for OpenCV colormap
            depth_norm = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            
        elif self.model_idx == 2 and DEPTH_PRO_AVAILABLE:
            # Depth Pro uses a strict file-loading pipeline that requires a focal length estimation.
            # We bypass this by writing the frame to a temporary file in real-time.
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                Image.fromarray(frame_rgb).save(tmp.name)
                tmp_path = tmp.name
                
            try:
                image, _, f_px = depth_pro.load_rgb(tmp_path)
                image = self.dp_transform(image).to(self.device)
                
                with torch.no_grad():
                    prediction = self.dp_model.infer(image, f_px=f_px)
                    depth = prediction["depth"].cpu().numpy()
                
                # Convert metric depth to inverse depth for visualization (closer = brighter)
                inverse_depth = 1.0 / (depth + 1e-6)
                depth_norm = cv2.normalize(inverse_depth, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                return cv2.applyColorMap(depth_norm, cv2.COLORMAP_INFERNO)
            finally:
                os.remove(tmp_path)

    def run(self):
        print("Starting video feed. Press 'q' to quit.")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("End of stream or cannot read camera.")
                break
                
            # Resize for consistent processing speed
            frame = cv2.resize(frame, (640, 480))
            
            # 1. Preprocess
            filtered_frame = self.apply_filter(frame, self.filter_idx)
            
            # 2. Depth Estimation
            frame_rgb = cv2.cvtColor(filtered_frame, cv2.COLOR_BGR2RGB)
            depth_colormap = self.get_depth_map(frame_rgb)
            
            # 3. Display Side-by-Side
            combined = np.hstack((filtered_frame, depth_colormap))
            
            # Add UI labels
            cv2.putText(combined, "Preprocessed Input", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(combined, "Depth Map", (650, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Underwater Depth Estimation", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = UnderwaterDepthApp(source=r"C:\Users\sarin\base\save\share\underwater\Leakage video.mp4")
    app.run()