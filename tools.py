import matplotlib.pyplot as plt
import numpy as np
import torch
import sys
import tkinter as tk
import cv2

# CONFIGS
COLORS = [
    (255, 182, 193),  # light red (light pink)
    (144, 238, 144),  # light green (light green)
    (173, 216, 230),  # light blue
    (255, 255, 224),  # light yellow
    (255, 182, 255),  # light magenta (light violet)
    (224, 255, 255),  # light cyan
    (255, 160, 122),  # light dark red (light coral)
    (144, 238, 144),  # light dark green (light green)
    (173, 216, 230)   # light dark blue (light sky blue)
]
DPI = 100

class OutputWindow(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Output Console")
        self.geometry("600x400")
        
        # Text widget with scrollbar
        self.text = tk.Text(self, wrap=tk.WORD)
        scrollbar = tk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.text.config(state=tk.DISABLED)
        
        # Redirect stdout/stderr
        sys.stdout = self
        sys.stderr = self
    
    def write(self, message):
        self.text.config(state=tk.NORMAL)
        self.text.insert(tk.END, message)
        self.text.see(tk.END)
        self.text.config(state=tk.DISABLED)
        
    def flush(self):
        pass
    
    def close(self):
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        self.destroy()

def setup_device():
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
            
        if device.type == "cuda":
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
            if torch.cuda.get_device_properties(0).major >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
        return device
        
def show_points(coords, labels, ax, marker_size=100):
    for coord, label in zip(coords, labels):
        if coord == []:
            continue
        coord = np.array(coord)
        label = np.array(label)
        pos_points = coord[label == 1]
        neg_points = coord[label == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', 
                    s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', 
                    s=marker_size, edgecolor='white', linewidth=1.25)   

def show_boxes(boxes, ax):
    for box in boxes:
        if box == []:
            continue
        box = np.array(box)
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', 
                        facecolor=(0, 0, 0, 0), lw=2))    
    
def show_masks(masks, ax, scores=None, borders=True, alpha=0.4):
    if masks is None: 
        return
    
    rgba_colors = [(r/255, g/255, b/255, alpha) for r, g, b in COLORS]
    
    for i, mask in enumerate(masks):
        color = rgba_colors[i % len(rgba_colors)]  # Cycle through colors
        h, w = mask.shape
        mask_image = np.zeros((h, w, 4))
        mask_image[mask == 1] = color
        ax.imshow(mask_image)

        if borders:
            import skimage.measure
            contours = skimage.measure.find_contours(mask, 0.5)
            for contour in contours:
                ax.plot(contour[:, 1], contour[:, 0], color='white', linewidth=2)
                
        if scores is not None and i < len(scores):
            yx = np.argwhere(mask)
            if yx.size > 0:
                y_min, x_min = yx.min(axis=0)
                ax.text(x_min, y_min - 5, f"{scores[i]:.3f}", color='white',
                        fontsize=6, bbox=dict(facecolor='black', alpha=alpha, pad=1))
                
def overlay_masks_on_images(image_paths, video_segments, output_path, alpha, show_borders, border_thickness=2, fps=30):
    """
    Overlay masks on images and create output video with optional borders
    
    Args:
        image_paths: List of paths to original images (in order)
        video_segments: Dict[int, np.ndarray] - masks per frame index (N, H, W)
        output_path: Path to save the output video
        alpha: Blend factor for overlay (0.0~1.0)
        show_borders: Whether to draw borders around masks
        border_thickness: Thickness of border lines in pixels
    """
    # Check at least one frame has masks
    first_mask = next(iter(video_segments.values()))
    N, H, W = first_mask.shape  # Number of masks, Height, Width

    # Prepare video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    for idx, img_path in enumerate(image_paths):
        # Read original image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img_float = img.astype(float)

        # Start with original image
        blended = img_float.copy()

        if idx in video_segments:
            masks = video_segments[idx]
            # Create combined mask (1 where any mask exists, 0 elsewhere)
            combined_mask = np.zeros((H, W), dtype=bool)
            for mask_idx in range(masks.shape[0]):
                combined_mask = np.logical_or(combined_mask, masks[mask_idx] > 0)
            
            # Create overlay only where masks exist
            overlay = np.zeros_like(img_float)
            for mask_idx in range(masks.shape[0]):
                mask = masks[mask_idx]
                color = COLORS[mask_idx % len(COLORS)]
                colored_mask = np.zeros_like(img_float)
                colored_mask[mask > 0] = color
                overlay += colored_mask

            # Apply blending only to masked regions
            blended[combined_mask] = (img_float[combined_mask] * (1 - alpha) + 
                                     overlay[combined_mask] * alpha)

            # Draw borders if enabled
            if show_borders:
                # Convert to uint8 for contour detection
                blended_uint8 = blended.astype(np.uint8)
                for mask_idx in range(masks.shape[0]):
                    mask = (masks[mask_idx] > 0).astype(np.uint8)
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    color = [int(c) for c in COLORS[mask_idx % len(COLORS)]]  # Convert to BGR tuple
                    cv2.drawContours(blended_uint8, contours, -1, color, border_thickness)
                blended = blended_uint8.astype(float)

        # Convert and write
        blended = np.clip(blended, 0, 255).astype(np.uint8)
        out.write(blended)

    out.release()

def save_green_screen_masks(image_paths, video_segments, output_path, fps=30):
    """
    Save masks as video with green background (chroma key).
    
    Args:
        image_paths: List of paths to original images (in order)
        video_segments: Dict[int, np.ndarray] - masks per frame index (N, H, W)
        output_path: Path to save the output video
    """
    # Get size from first mask
    first_mask = next(iter(video_segments.values()))
    N, H, W = first_mask.shape

    # Prepare video writer (MP4, no alpha)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (W, H))

    for idx, img_path in enumerate(image_paths):
        # Read original image
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Start with green background (BGR: (0,255,0))
        green_bg = np.zeros((H, W, 3), dtype=np.uint8)
        green_bg[:, :] = (0, 255, 0)  

        if idx in video_segments:
            masks = video_segments[idx]

            # Combined mask (True = foreground)
            combined_mask = np.zeros((H, W), dtype=bool)
            for mask_idx in range(masks.shape[0]):
                combined_mask = np.logical_or(combined_mask, masks[mask_idx] > 0)

            # Copy object pixels where mask is true
            green_bg[combined_mask] = img_rgb[combined_mask]

            # # Optional: draw borders around masks
            # for mask_idx in range(masks.shape[0]):
            #     mask = (masks[mask_idx] > 0).astype(np.uint8)
            #     contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #     cv2.drawContours(green_bg, contours, -1, (255, 0, 0), border_thickness)

        # Convert back to BGR for saving
        out.write(cv2.cvtColor(green_bg, cv2.COLOR_RGB2BGR))

    out.release()

# def save_transparent_masks(image_paths, video_segments, output_path):
#     """
#     Save transparent masks as video where only segmented pixels are visible with alpha channel.
    
#     Args:
#         image_paths: List of paths to original images (in order)
#         video_segments: Dict[int, np.ndarray] - masks per frame index (N, H, W)
#         output_path: Path to save the output video
#         show_borders: Whether to draw borders around masks
#     """
#     # Check at least one frame has masks
#     first_mask = next(iter(video_segments.values()))
#     N, H, W = first_mask.shape  # Number of masks, Height, Width

#     # Prepare video writer for transparent video (with alpha channel)
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(str(output_path), fourcc, 30, (W, H))

#     for idx, img_path in enumerate(image_paths):
#         # Read original image
#         img = cv2.imread(str(img_path))
#         if img is None:
#             raise ValueError(f"Could not read image: {img_path}")
        
#         # Convert BGR to RGB
#         img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
#         # Create transparent image (RGBA)
#         transparent_img = np.zeros((H, W, 4), dtype=np.uint8)
        
#         # Copy RGB channels
#         transparent_img[:, :, 0:3] = img_rgb
        
#         # Set alpha channel to 0 initially (fully transparent)
#         transparent_img[:, :, 3] = 0
        
#         if idx in video_segments:
#             masks = video_segments[idx]
            
#             # Create combined mask (True where any mask exists, False elsewhere)
#             combined_mask = np.zeros((H, W), dtype=bool)
#             for mask_idx in range(masks.shape[0]):
#                 combined_mask = np.logical_or(combined_mask, masks[mask_idx] > 0)
            
#             # Set alpha to 255 (fully opaque) where masks exist
#             transparent_img[combined_mask, 3] = 255
            
#             # # Draw borders if enabled
#             # if show_borders:
#             #     for mask_idx in range(masks.shape[0]):
#             #         mask = (masks[mask_idx] > 0).astype(np.uint8)
#             #         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#             #         color = [int(c) for c in COLORS[mask_idx % len(COLORS)]]  # Get color for this mask
#             #         for contour in contours:
#             #             # Draw border on the alpha channel to make it visible
#             #             cv2.drawContours(transparent_img[:, :, 3], [contour], -1, 255, border_thickness)

#         # Convert to BGR for OpenCV (since OpenCV expects BGR)
#         transparent_bgr = cv2.cvtColor(transparent_img, cv2.COLOR_RGBA2BGRA)
        
#         # Write frame
#         out.write(transparent_bgr)

#     out.release()


def save_transparent_masks_img(original_image, masks, output_path):
    """
    Save transparent masks as image where only segmented pixels are visible with alpha channel.
    
    Args:
        original_image: Original image as numpy array (H, W, 3) in RGB format
        masks: Mask array (N, H, W) where N is number of masks
        output_path: Path to save the output image
    """
    H, W, C = original_image.shape
    
    # Create transparent image (RGBA)
    transparent_img = np.zeros((H, W, 4), dtype=np.uint8)
    
    # Copy RGB channels from original image
    transparent_img[:, :, 0:3] = original_image
    
    # Set alpha channel to 0 initially (fully transparent)
    transparent_img[:, :, 3] = 0
    
    if masks is not None:
        # Create combined mask (True where any mask exists, False elsewhere)
        combined_mask = np.zeros((H, W), dtype=bool)
        for mask_idx in range(masks.shape[0]):
            combined_mask = np.logical_or(combined_mask, masks[mask_idx] > 0)
        
        # Set alpha to 255 (fully opaque) where masks exist
        transparent_img[combined_mask, 3] = 255
        
        # # Draw borders if enabled
        # if show_borders:
        #     for mask_idx in range(masks.shape[0]):
        #         mask = (masks[mask_idx] > 0).astype(np.uint8)
        #         contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #         for contour in contours:
        #             # Draw border on the alpha channel
        #             cv2.drawContours(transparent_img[:, :, 3], [contour], -1, 255, border_thickness)
    
    # Save with Pillow which handles transparency better
    from PIL import Image
    img_pil = Image.fromarray(transparent_img, mode='RGBA')
    img_pil.save(output_path, "PNG")


### FOR DISPLAY CM TO LOCAL HOST
    
# from flask import Flask, Response, render_template_string
# import cv2

# app = Flask(__name__)
# camera = cv2.VideoCapture(0)  # Change to 1 if needed

# HTML_PAGE = """
# <html>
# <head><title>Live Webcam</title></head>
# <body>
# <h1>Live Stream</h1>
# <img src="{{ url_for('video_feed') }}">
# </body>
# </html>
# """

# def generate_frames():
#     while True:
#         success, frame = camera.read()
#         if not success:
#             break
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# @app.route('/')
# def index():
#     return render_template_string(HTML_PAGE)

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=8000)

