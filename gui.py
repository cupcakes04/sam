import os
import json
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor
from sam2.build_sam import build_sam2_camera_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
import cv2
import copy
import mss
from tkinter import scrolledtext
from tkinter import simpledialog

from tools import *

class SAM2SegmentationApp:
    def __init__(self, root, debug=False):
        self.root = root
        self.debug = debug
        self.root.title("Interactive Segmentation Tool")
        self.save_raw_data = False
        self.extraction_percent = 100   # 100% is lossless
        
        # Initialize SAM2 model
        self.sam2_checkpoint_dir = Path("./sam2/sam/checkpoints")
        self.sam2_checkpoint_dict = {1: 'sam2.1_hiera_tiny.pt', 2: 'sam2.1_hiera_small.pt', 3: 'sam2.1_hiera_large.pt', 4: 'sam2.1_hiera_base_plus.pt'}
        self.model_cfg_dir = Path("configs/sam2.1")
        self.model_cfg_dict = {1: 'sam2.1_hiera_t.yaml', 2: 'sam2.1_hiera_s.yaml', 3: 'sam2.1_hiera_l.yaml', 4: 'sam2.1_hiera_b+.yaml'}
        self.model_idx = 2
        self.sam2_img_model = None
        self.sam2_vid_model = None
        self.sam2_cam_model = None
        self.predictor = None
        
        # Initialize variables
        self.image = None
        self.file_path = None
        self.mode = "image"  # or "video", "live"
        self.prev_mode = self.mode
        self.video_path = None
        self.frame_folder = Path("./video_frames")
        self.frame_paths = []
        self.current_frame_idx = 0
        self.total_frames = 0
        self.video_playing = False
        self.video_segments = {}
        self.backup_video_segments = {}
        
        # Segmentation variables
        self.input_points = [[]]
        self.input_labels = [[]]
        self.input_boxes = [[]]
        self.masks = None
        self.scores = None
        self.redo_stack = []
        self.action_history = [None]
        self.current_label = 1
        self.mask_idx = 0
        self.total_masks = 1
        self.maintain_display = False
        self.borders = True
        self.alpha = 0.4
        
        # Live video variables
        self.delay = 10 # ms
        self.video_stoped = True
        self.live_propagation = False
        
        # Create GUI
        self.create_widgets()
        self.setup_bindings()
        
        # Device selection
        self.device = setup_device()
        self.log(f"Using device: {self.device}")
        self.insert_model(default_model=True)

    def log(self, message, debug_only=False):
        """
        Robust logging method that always works
        Args:
            message: Text to display
            debug_only: Only show if in debug mode
        """
        if self.debug:
            print(message)
        if debug_only:
            return
        
        # Update GUI log if not debug-only
        self.log_text.config(state=tk.NORMAL)
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.log_text.update_idletasks()  # Force immediate update
            
#============================================================================================================
#----------------------------------------------- UI INTERFACE -----------------------------------------------
#============================================================================================================

    def create_widgets(self):
        # Main frame
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Menu bar
        self.menu_bar = tk.Menu(self.root)
        self.file_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.file_menu.add_command(label="Open Image", command=self.open_image)
        self.file_menu.add_command(label="Open Video", command=self.open_video)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Save Masks Image", command=self.save_masks_img)
        self.file_menu.add_command(label="Save Masks Video", command=self.save_masks_vid)
        self.file_menu.add_command(label="Insert Model", command=self.insert_model)
        self.file_menu.add_command(label="Exit", command=self.root.quit)
        self.menu_bar.add_cascade(label="File", menu=self.file_menu)
        
        self.edit_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.edit_menu.add_command(label="Undo (Ctrl+Z)", command=self.undo)
        self.edit_menu.add_command(label="Redo (Ctrl+Y)", command=self.redo)
        self.menu_bar.add_cascade(label="Edit", menu=self.edit_menu)
        
        self.help_menu = tk.Menu(self.menu_bar, tearoff=0)
        self.help_menu.add_command(label="About", command=self.show_about)
        self.menu_bar.add_cascade(label="Help", menu=self.help_menu)
        
        self.root.config(menu=self.menu_bar)
        
        # Control panel
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Controls")
        self.control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        # Mode selection
        self.mode_var = tk.StringVar(value="image")
        ttk.Radiobutton(self.control_frame, text="Image Mode", variable=self.mode_var, 
                       value="image", command=self.set_mode).pack(anchor=tk.W)
        ttk.Radiobutton(self.control_frame, text="Video Mode", variable=self.mode_var, 
                       value="video", command=self.set_mode).pack(anchor=tk.W)
        ttk.Radiobutton(self.control_frame, text="Live Mode", variable=self.mode_var, 
                       value="live", command=self.set_mode).pack(anchor=tk.W)
        
        # Video controls (initially hidden)
        self.video_control_frame = ttk.Frame(self.control_frame)
        
        # Segmentation controls
        ttk.Label(self.control_frame, text="Point Label:").pack(anchor=tk.W)
        self.label_var = tk.StringVar(value="Positive (1)")
        ttk.Radiobutton(self.control_frame, text="Positive (1)", variable=self.label_var, 
                       value="Positive (1)", command=lambda: self.set_label(1)).pack(anchor=tk.W)
        ttk.Radiobutton(self.control_frame, text="Negative (0)", variable=self.label_var, 
                       value="Negative (0)", command=lambda: self.set_label(0)).pack(anchor=tk.W)
        
        ttk.Button(self.control_frame, text="Predict Mask (Enter)", command=self.predict_mask).pack(fill=tk.X, pady=5)
        ttk.Button(self.control_frame, text="Add Mask (A)", command=lambda: self.update_current_mask('add')).pack(fill=tk.X)
        ttk.Button(self.control_frame, text="Delete Mask (D)", command=lambda: self.update_current_mask('del')).pack(fill=tk.X)
        ttk.Button(self.control_frame, text="Reset Segmentations", command=self.reset_segmentation).pack(fill=tk.X)
        
        # Save raw mask details (deafult: False)
        def update_save_raw():
            self.save_raw_data = self.save_raw_data_var.get()
        self.save_raw_data_var = tk.BooleanVar(value=self.save_raw_data)  # Default: not ticked (False)
        ttk.Checkbutton(self.control_frame, text="Save Raw Data", variable=self.save_raw_data_var, command=update_save_raw).pack(anchor=tk.W)
        
        # Annotation control: borders & alpha
        def update_borders():
            self.borders = self.borders_var.get()
            self.update_display()  # Redraw to apply change
        self.borders_var = tk.BooleanVar(value=self.borders)  # Default: not ticked (False)
        ttk.Checkbutton(self.control_frame, text="Show Borders", variable=self.borders_var, command=update_borders).pack(anchor=tk.W)
        
        # Save transparent option
        def update_save_transparent():
            self.save_transparent = self.save_transparent_var.get()
        self.save_transparent_var = tk.BooleanVar(value=False)  # Default: not ticked (False)
        ttk.Checkbutton(self.control_frame, text="Save Transparent", variable=self.save_transparent_var, command=update_save_transparent).pack(anchor=tk.W)
        
        def update_alpha():
            self.alpha = self.alpha_var.get()
            self.update_display()
        ttk.Label(self.control_frame, text="Overlay Alpha:").pack(anchor=tk.W)
        self.alpha_var = tk.DoubleVar(value=self.alpha)  # Initial alpha value
        ttk.Scale(self.control_frame, from_=0.0, to=1.0, orient='horizontal', variable=self.alpha_var, command=lambda val: update_alpha()).pack(fill=tk.X)
        
        # Extraction percent
        def update_extract_percent():
            self.extraction_percent = self.extraction_percent_var.get()
            self.log(f"extraction_percent: {self.extraction_percent}%")
        ttk.Label(self.control_frame, text="Extraction percent:").pack(anchor=tk.W)
        self.extraction_percent_var = tk.DoubleVar(value=self.extraction_percent)  # Initial alpha value
        ttk.Scale(self.control_frame, from_=0, to=100, orient='horizontal', variable=self.extraction_percent_var, command=lambda val: update_extract_percent()).pack(fill=tk.X)
        
        # Dropdown to select current mask index
        def on_mask_idx_change(event=None):
            self.mask_idx = int(self.mask_idx_var.get())
            self.update_display()
            self.log(f"Selected mask index: {self.mask_idx}")
            
        ttk.Label(self.control_frame, text="Select Mask Index:").pack(anchor=tk.W)
        self.mask_idx_var = tk.StringVar()
        self.mask_selector = ttk.Combobox(
            self.control_frame,
            textvariable=self.mask_idx_var,
            state="readonly",
            values=[str(i) for i in range(self.total_masks)],
        )
        self.mask_selector.pack(fill=tk.X)
        self.mask_selector.bind("<<ComboboxSelected>>", on_mask_idx_change)

        # Log Frame - Status bar
        self.log_frame = ttk.LabelFrame(self.main_frame, text="Log Messages")
        self.log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        self.log_text = tk.Text(self.log_frame, height=5, wrap=tk.WORD)
        self.log_scroll = ttk.Scrollbar(self.log_frame)
        self.log_text.config(yscrollcommand=self.log_scroll.set)
        self.log_scroll.config(command=self.log_text.yview)
        self.log_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.log_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)

        # Matplotlib figure
        self.figure = Figure(figsize=(10, 10), dpi=100)
        self.ax = self.figure.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.figure, master=self.main_frame)
        self.canvas.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        
    def update_mask_selector(self):
        """Update the Combobox values to reflect current mask count"""
        if self.total_masks >= 1:
            new_values = [str(i) for i in range(self.total_masks)]
            self.mask_selector['values'] = new_values
            # Keep the current selection if it's still valid
            if self.mask_idx < self.total_masks:
                self.mask_idx_var.set(str(self.mask_idx))
            else:
                # Select the last mask if current selection is invalid
                self.mask_idx_var.set(str(self.total_masks - 1))
        else:
            self.mask_selector['values'] = []
            self.mask_idx_var.set('')
        
    def setup_bindings(self):
        self.root.bind('<Key>', self.on_key_press)
        self.canvas.mpl_connect('button_press_event', self.on_click)
        self.canvas.mpl_connect('button_press_event', self.on_hold)
        self.canvas.mpl_connect('button_release_event', self.on_release)
        
    def set_mode(self):
        self.mode = self.mode_var.get()
        if self.mode in ["video", "live"] and self.prev_mode != self.mode:
            self.create_video_controls()
            if self.mode == "live":
                self.initialize_recorder()
        elif self.mode == "image":
            self.remove_video_controls()
        self.prev_mode = self.mode
        self.log(f"Mode set to {self.mode}")
        
    def set_label(self, label):
        self.current_label = label
        self.update_display()
        
#============================================================================================================
#----------------------------------------------- INITIALISE IMG VID -----------------------------------------
#============================================================================================================

    def insert_model(self, default_model=False):
        
        # Prompt user to choose from available list
        if not default_model:
            selected_model = int(simpledialog.askstring(
                "Select Model",
                f"Choose:\n\ntiny = 1\nsmall = 2\nlarge = 3\nb+ = 4"
            ))
            if selected_model in self.model_cfg_dict:
                self.model_idx = selected_model
            
        self.config_file = str(self.model_cfg_dir / self.model_cfg_dict[self.model_idx])
        self.checkpoint_file = str(self.sam2_checkpoint_dir / self.sam2_checkpoint_dict[self.model_idx])

        # Build the model based on current mode
        try:
            self.sam2_img_model = build_sam2(self.config_file, self.checkpoint_file, device=self.device)
            self.sam2_vid_model = build_sam2_video_predictor(self.config_file, self.checkpoint_file, device=self.device)
            self.sam2_cam_model = build_sam2_camera_predictor(self.config_file, self.checkpoint_file, device=self.device)
            self.log(f"Loaded model:\n  cfg: {self.config_file}\n  ckpt: {self.checkpoint_file}")
        except Exception as e:
            self.log(f"Model loading failed: {e}")

    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All Files", "*.*")]
        )
        if file_path:
            self.file_path = Path(file_path)
            self.reset_segmentation()
            self.image = np.array(Image.open(file_path).convert("RGB"))
            self.video_path = None
            self.mode = "image"
            self.mode_var.set("image")
            
            # Initialize SAM2 predictor
            self.predictor = SAM2ImagePredictor(self.sam2_img_model)
            self.predictor.set_image(self.image)
            
            self.update_display()
            self.log(f"Loaded image: {os.path.basename(file_path)}")
            
    def open_video(self):
        file_path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv"), ("All Files", "*.*")]
        )
        if file_path:
            self.log(f"Loading video...")
            self.reset_segmentation()
            self.video_path = Path(file_path)
            self.mode = "video"
            self.mode_var.set("video")
            
            # Extract frames if not already done
            self.frame_folder = Path(filedialog.askdirectory(title="Select Output Directory"))
            self.frame_folder.mkdir(exist_ok=True)
            self.video_dir = self.frame_folder / self.video_path.stem
            self.video_dir.mkdir(exist_ok=True)
            self.extract_video_frames()
            
            # Initialize predictor
            self.predictor = self.sam2_vid_model
            self.inference_state = self.predictor.init_state(video_path=str(self.video_dir))
            self.reset_vid_predictor()
            
            # Initialize video display
            self.show_current_frame()
            
            # Set Mode
            self.set_mode()
            
            # Save paths
            self.pkl_path = self.video_dir / "raw_folder" / f"{self.video_dir.stem}.pkl"
            self.inf_path = self.video_dir / "raw_folder" / f"{self.video_dir.stem}.pth"
            self.vid_path = self.video_dir / "raw_folder" / f"{self.video_dir.stem}.mp4"
            self.load_saved_data()
            
    def initialize_recorder(self):
        self.predictor = self.sam2_cam_model
        self.inference_state = None
        self.reset_vid_predictor()
            
    def extract_video_frames(self):
        """Extract video frames with percentage control
        
        Args:
            extraction_percent: 100 = all frames, 50 = every other frame, etc.
        """
        if not 0 < self.extraction_percent <= 100:
            raise ValueError("extraction_percent must be between 0 and 100")
        
        raw_folder = self.video_dir / "raw_folder"
        if not raw_folder.is_dir():
            self.log(f"Extracting video: {self.video_path}")
            cap = cv2.VideoCapture(str(self.video_path))
            
            i = 0
            saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_interval = max(1, round(100 / self.extraction_percent))
                if i % frame_interval == 0:
                    frame_file = self.video_dir / f"{saved_count:05d}.jpg"
                    cv2.imwrite(str(frame_file), frame)
                    saved_count += 1
                    if saved_count % 50 == 0:
                        self.log(f"Saved frame {saved_count} (original frame {i})")
                i += 1
                
            cap.release()
            raw_folder.mkdir(exist_ok=True)
            self.log(f"Extraction complete: {saved_count} frames saved ({self.extraction_percent}% of original)")
        else:
            self.log(f"Video: {self.video_path} already extracted")
        
        self.frame_paths = sorted([
            p for p in self.video_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg"]
        ])
        self.total_frames = len(self.frame_paths)

    def show_current_frame(self):
        """Display the current frame with any existing annotations"""
        idx = self.current_frame_idx
        if 0 <= idx < len(self.frame_paths):
            frame_path = self.frame_paths[idx]
            self.image = np.array(Image.open(frame_path))
            if self.video_segments and idx in self.video_segments: self.masks = self.video_segments[idx]
            self.reset_segmentation(reset_masks=False)
            self.update_display()
            self.log(f"Frame {idx+1}/{len(self.frame_paths)}")

    def play_video(self):
        """Handle video playback"""
        if self.video_playing and self.current_frame_idx < len(self.frame_paths) - 1:
            self.current_frame_idx += 1
            self.frame_slider.set(self.current_frame_idx)
            self.show_current_frame()
            self.root.after(33, self.play_video)  # ~0.001/1.000fps, but the inference is so long it doesnt matter

    def toggle_video_playback(self):
        """Toggle between play and pause"""
        self.video_playing = not self.video_playing
        if self.video_playing:
            self.play_video()
        self.update_controls()

    def update_controls(self):
        """Update control buttons based on current state"""
        if hasattr(self, 'play_pause_btn'):
            self.play_pause_btn.config(
                text="Pause" if self.video_playing else "Play",
                command=self.toggle_video_playback
            )

    def create_video_controls(self):
        """Create video-specific control elements"""
        # First remove any existing video controls
        self.remove_video_controls()
        
        # Create fresh controls based on current mode
        self.video_control_frame.pack(fill=tk.X, pady=5)
        
        # Navigation controls
        if self.mode == "live":
            def update_fps():
                self.delay = round(self.delay_var.get() * 100) + 10
                self.update_display()
                self.log(f"current delay: {self.delay}ms")
                
            # Live mode specific controls
            ttk.Label(self.control_frame, text="Delay (10~110ms):").pack(anchor=tk.W)
            self.delay_var = tk.DoubleVar(value=self.delay)
            self.delay_slider = ttk.Scale(
                self.control_frame, from_=0.0, to=1.0, orient='horizontal',
                variable=self.delay_var, command=lambda val: update_fps()
            )
            self.delay_slider.pack(fill=tk.X)

            ttk.Label(self.control_frame, text="Select Video Source:").pack(anchor=tk.W)
            self.video_source_var = tk.StringVar(value="none")
            
            def select_video_source():
                selected = self.video_source_var.get()
                self.log(f"Selected source: {selected}")
                
            ttk.Radiobutton(
                self.control_frame, text="Camera", variable=self.video_source_var,
                value="camera", command=select_video_source
            ).pack(anchor=tk.W)
            ttk.Radiobutton(
                self.control_frame, text="Screen", variable=self.video_source_var,
                value="screen", command=select_video_source
            ).pack(anchor=tk.W)
            ttk.Radiobutton(
                self.control_frame, text="Video File", variable=self.video_source_var,
                value="file", command=select_video_source
            ).pack(anchor=tk.W)

            self.start_stop_btn = ttk.Button(
                self.control_frame, text="Start", 
                command=self.toggle_start_stop_vid
            )
            self.start_stop_btn.pack(fill=tk.X, pady=2)
            
        elif self.mode == "video":
            # Video mode specific controls
            ttk.Button(self.video_control_frame, text="<<", 
                    command=lambda: self.navigate_frame(-10)).pack(side=tk.LEFT)
            ttk.Button(self.video_control_frame, text="<", 
                    command=lambda: self.navigate_frame(-1)).pack(side=tk.LEFT)
            
            self.play_pause_btn = ttk.Button(
                self.video_control_frame, text="Play", 
                command=self.toggle_video_playback
            )
            self.play_pause_btn.pack(side=tk.LEFT)
            
            ttk.Button(self.video_control_frame, text=">", 
                    command=lambda: self.navigate_frame(1)).pack(side=tk.LEFT)
            ttk.Button(self.video_control_frame, text=">>", 
                    command=lambda: self.navigate_frame(10)).pack(side=tk.LEFT)
            
            self.frame_slider = ttk.Scale(
                self.video_control_frame, from_=0, 
                to=max(1, len(self.frame_paths)-1), 
                command=self.on_slider_move
            )
            self.frame_slider.pack(fill=tk.X, pady=5)
        
        # Universal controls for both video modes
        ttk.Button(
            self.control_frame, text="Propagate Annotations", 
            command=self.propagate_annotations
        ).pack(fill=tk.X, pady=2)
        ttk.Button(
            self.control_frame, text="Reset Propagation", 
            command=self.reset_vid_predictor
        ).pack(fill=tk.X, pady=2)

    def remove_video_controls(self):
        """Remove all video-related controls while preserving other controls"""
        # List of video-specific widget texts to remove
        video_specific_widgets = [
            "Propagate Annotations", "Reset Propagation",
            "Delay (10~110ms):", "Start", "Stop", "Camera", 
            "Screen", "Video File", "Select Video Source:",
            "<<", "<", "Play", "Pause", ">", ">>"
        ]
        
        # Destroy all widgets in video control frame
        for widget in self.video_control_frame.winfo_children():
            widget.destroy()
        self.video_control_frame.pack_forget()
        
        # Destroy specific video-related widgets in main control frame
        for widget in self.control_frame.winfo_children():
            widget_class = widget.winfo_class()
            
            # Handle widgets that might not have 'text' property
            try:
                widget_text = widget.cget("text") if widget_class in ["TButton", "TLabel", "TRadiobutton"] else None
            except:
                widget_text = None
            
            # Remove video-specific widgets
            if (widget_text in video_specific_widgets or 
                (widget_class == "TScale" and getattr(self, "delay_slider", None) == widget) or
                (widget_class == "TScale" and getattr(self, "frame_slider", None) == widget)):
                widget.destroy()
        
        # Clean up slider references if they exist
        if hasattr(self, "delay_slider"):
            del self.delay_slider
        if hasattr(self, "frame_slider"):
            del self.frame_slider


    def navigate_frame(self, delta):
        """Navigate through frames by delta"""
        self.video_playing = False
        new_idx = max(0, min(self.current_frame_idx + delta, len(self.frame_paths) - 1))
        if new_idx != self.current_frame_idx:
            self.current_frame_idx = new_idx
            self.frame_slider.set(self.current_frame_idx)
            self.show_current_frame()

    def on_slider_move(self, value):
        """Handle frame slider movement"""
        new_idx = int(float(value))
        if new_idx != self.current_frame_idx:
            self.current_frame_idx = new_idx
            self.show_current_frame()
   
   
#========================================================================================================
#----------------------------------------------- LIVE VIDEO -----------------------------------------------
#========================================================================================================
        
    def stop_video_stream(self):
        self.log("Turned off local cam and stoping live propagation")
        self.live_propagation = False
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        if hasattr(self, 'canvas_update_job'):
            self.canvas.get_tk_widget().after_cancel(self.canvas_update_job)
            del self.canvas_update_job
            
    def toggle_start_stop_vid(self):
        # Update button text accordingly
        self.video_stoped = not self.video_stoped
        if hasattr(self, 'start_stop_btn'):
            self.start_stop_btn.config(
                text="Start" if self.video_stoped else "Stop",
                command=self.toggle_start_stop_vid
            )
        
        # Start/Stop video stream
        if not self.video_stoped:
            self.update_video_frame()  # kick off the loop
        else:
            self.stop_video_stream()
            

    def update_video_frame(self):
        source = self.video_source_var.get()

        if source == "screen":
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                sct_img = sct.grab(monitor)
                frame = np.array(sct_img)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        elif source == "camera":
            if not hasattr(self, 'cap') or not self.cap.isOpened():
                self.log("Turning on local cam...")
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    self.log("Failed to open webcam")
                    return
            ret, frame = self.cap.read()
            if not ret:
                self.log("No frame read from camera")
                return

        # elif source == "file":
        #     if not hasattr(self, 'cap') or not self.cap.isOpened():
        #         return
        #     ret, frame = self.cap.read()
        #     if not ret:
        #         self.log("End of video file or failed read")
        #         return

        self.live_frame = frame
        self.image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Start live propagation if enabled
        if self.live_propagation:
            out_obj_ids, out_mask_logits = self.predictor.track(frame)

            # Assign current masks
            mask_list = []
            for i, out_obj_id in enumerate(out_obj_ids):
                mask_list.append((out_mask_logits[i] > 0.0).cpu().numpy())
            self.masks = np.concatenate(mask_list, axis=0)

        self.reset_segmentation(reset_masks=False)
        self.update_display()

        # Schedule the next update (e.g., every 'delay' ms)
        self.canvas_update_job = self.canvas.get_tk_widget().after(self.delay, self.update_video_frame)


#========================================================================================================
#----------------------------------------------- CONTROLS -----------------------------------------------
#========================================================================================================

    def reset_segmentation(self, reset_masks=True):
        self.input_points = [[]]
        self.input_labels = [[]]
        self.input_boxes = [[]]
        if reset_masks: 
            self.masks = None
            self.total_masks = 1
        self.scores = None
        self.redo_stack = []
        self.action_history = [None]
        self.mask_idx = 0
        self.current_label = 1
        self.update_display()
        self.update_mask_selector()
        
    def update_display(self):
        if self.maintain_display: return
        self.ax.clear()
        if self.image is not None:
            self.ax.imshow(self.image)
        show_points(self.input_points, self.input_labels, self.ax)
        show_boxes(self.input_boxes, self.ax)
        show_masks(self.masks, self.ax, self.scores, borders=self.borders, alpha=self.alpha)
        
        title = f"Mask: {self.mask_idx} | Label: {'Positive' if self.current_label==1 else 'Negative'} | Points: {sum(len(s) for s in self.input_labels)}"
        self.ax.set_title(title)
        self.canvas.draw()
     
    
    def on_click(self, event):
        if not self.video_playing and event.inaxes and (not hasattr(event, 'key') or event.key != 'control'):
            self.input_points[self.mask_idx].append([event.xdata, event.ydata])
            self.input_labels[self.mask_idx].append(self.current_label)
            self.update_display()
            
            self.action_history.append('point')
            self.redo_stack.clear()
            
    def on_hold(self, event):
        if not self.video_playing and event.inaxes and hasattr(event, 'key') and event.key == 'control':
            self.box_start = (event.xdata, event.ydata)
            ax = event.inaxes
            self.start_marker = ax.plot(event.xdata, event.ydata, 'bx', markersize=25)[0]
            self.canvas.draw()
            self.maintain_display = True
            
    def on_release(self, event):
        if not self.video_playing and hasattr(self, 'box_start') and self.box_start is not None:
            x1, y1 = self.box_start
            x2, y2 = event.xdata, event.ydata
            x_min, x_max = sorted([x1, x2])
            y_min, y_max = sorted([y1, y2])
            self.input_boxes[self.mask_idx] = [x_min, y_min, x_max, y_max]
            if hasattr(self, 'start_marker'):
                self.start_marker.remove()
                del self.start_marker
            del self.box_start
            self.maintain_display = False
            self.update_display()
            self.action_history.append('box')
            self.redo_stack.clear()
            
    def on_key_press(self, event):
        key = event.keysym.lower()
        if key == 'return':
            self.predict_mask()
        elif key == 'a':
            self.update_current_mask('add')
        elif key == 'd':
            self.update_current_mask('del')
        elif key == 'shift_l':
            self.current_label = 1 - self.current_label  # Toggle between 0 and 1
            self.label_var.set("Positive (1)" if self.current_label == 1 else "Negative (0)")
        elif key == 'z' and event.state & 0x4:  # Ctrl+Z
            self.undo()
        elif key == 'y' and event.state & 0x4:  # Ctrl+Y
            self.redo()
        elif key == 's' and event.state & 0x4:  # Ctrl+S
            if self.mode == "image":
                self.save_masks_img()
            if self.mode == "video":
                self.save_masks_vid()
        elif key == 'space' and self.mode == "video":
            self.toggle_video_playback()
            
        self.update_display()
        
    def update_current_mask(self, mode, undo_redo=False):
        if mode in ["add", "a"]:
            self.input_points.append([])
            self.input_labels.append([])
            self.input_boxes.append([])
            self.mask_idx += 1
            self.total_masks += 1
        elif mode in ["del", "d"]:
            self.input_points.pop()
            self.input_labels.pop()
            self.input_boxes.pop()
            if self.mask_idx > 0: 
                self.mask_idx -= 1
                self.total_masks -= 1
            
        # To always preserve the [[]] structure:
        if not self.input_points:
            self.input_points.append([])
            self.input_labels.append([])
            self.input_boxes.append([])
            
        if not undo_redo: 
            self.action_history.append(mode)
            self.update_display()
            
        self.update_mask_selector()
            
    def undo(self):
        if not self.action_history:
            self.log('nothing to undo')
            return
            
        last_action = self.action_history[-1]
        
        if last_action == 'point':
            if self.input_points[self.mask_idx]:
                point = self.input_points[self.mask_idx].pop()
                label = self.input_labels[self.mask_idx].pop()
                self.redo_stack.append(('point', point, label))
                
        elif last_action == 'box':
            if self.input_boxes[self.mask_idx]:
                box = self.input_boxes[self.mask_idx]
                self.input_boxes[self.mask_idx] = []
                self.redo_stack.append(('box', box))
                
        elif last_action == 'predict':
            if self.masks is not None and len(self.masks) > 0:
                popped_mask = self.masks[-1:, :, :]
                self.masks = self.masks[:-1, :, :]
                self.redo_stack.append(('predict', popped_mask))
                
        elif last_action == 'add':
            self.update_current_mask('del', undo_redo=True)
            self.redo_stack.append(('add', None))
        elif last_action == 'del':
            self.update_current_mask('add', undo_redo=True)
            self.redo_stack.append(('del', None))
            
        self.action_history.pop()
        self.update_display()
        
    def redo(self):
        if not self.redo_stack:
            self.log('nothing to redo')
            return
            
        action, *data = self.redo_stack.pop()
        
        if action == 'point':
            point, label = data
            self.input_points[self.mask_idx].append(point)
            self.input_labels[self.mask_idx].append(label)
            
        elif action == 'box':
            box = data[0]
            self.input_boxes[self.mask_idx] = box
            
        elif action == 'predict':
            mask = data[0]
            if self.masks is None:
                self.masks = mask
            else:
                self.masks = np.concatenate([self.masks, mask], axis=0)
                
        elif action == 'add':
            self.update_current_mask('add', undo_redo=True)
        elif action == 'del':
            self.update_current_mask('del', undo_redo=True)
            
        self.action_history.append(action)
        self.update_display()
        
#============================================================================================================
#----------------------------------------------- AI INFERENCE -----------------------------------------------
#============================================================================================================

    def predict_mask(self):
        if not self.input_points and not self.input_boxes:
            messagebox.showwarning("Warning", "Nothing to predict. Add points or boxes first.")
            return
        
        def process_anns(points, labels, boxes):
            def _process(ann): 
                return np.array([ann]) if ann != [] else None
            points = _process(points)
            labels = _process(labels)
            boxes = _process(boxes)
            return points, labels, boxes
        
        mask_list, scores = [], []
            
        if self.mode == 'image':
            for points, labels, boxes in zip(self.input_points, self.input_labels, self.input_boxes):
                
                points, labels, boxes = process_anns(points, labels, boxes)
                if all(x is None for x in [points, labels, boxes]):
                    continue
                
                self.log((points, labels, boxes), debug_only=True)
                mask, score, _ = self.predictor.predict(
                    point_coords=points,
                    point_labels=labels,
                    box=boxes,
                    multimask_output=False
                )
                mask_list.append(mask)
                scores.append(round(float(score[0]), 3))
                
            self.masks = np.concatenate(mask_list, axis=0)
            self.scores = scores
            
        if self.mode == 'video':
            for ann_obj_id, (points, labels, boxes) in enumerate(zip(self.input_points, self.input_labels, self.input_boxes)):
                
                points, labels, boxes = process_anns(points, labels, boxes)
                self.log(("id:", ann_obj_id, "pt:", points, "lb:", labels, "bx:", boxes), debug_only=True)
                
                if all(x is None for x in [points, labels, boxes]):
                    continue
                
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.current_frame_idx,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                    box=boxes,
                )
                
            # Assign current masks
            for i, out_obj_id in enumerate(out_obj_ids):
                self.log((out_mask_logits[i] > 0.0).cpu().numpy().shape, debug_only=True)
                mask_list.append((out_mask_logits[i] > 0.0).cpu().numpy())
            self.masks = np.concatenate(mask_list, axis=0)
            
            if self.video_segments: 
                self.video_segments[self.current_frame_idx] = self.masks
        
        if self.mode == "live":
            
            self.predictor.load_first_frame(self.live_frame)
            for ann_obj_id, (points, labels, boxes) in enumerate(zip(self.input_points, self.input_labels, self.input_boxes)):
                points, labels, boxes = process_anns(points, labels, boxes)
                self.log(("id:", ann_obj_id, "pt:", points, "lb:", labels, "bx:", boxes), debug_only=True)
                
                if all(x is None for x in [points, labels, boxes]):
                    continue
                
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=ann_obj_id,
                    points=points,
                    labels=labels,
                    bbox=boxes,
                )
                
            # Assign current masks
            for i, out_obj_id in enumerate(out_obj_ids):
                self.log((out_mask_logits[i] > 0.0).cpu().numpy().shape, debug_only=True)
                mask_list.append((out_mask_logits[i] > 0.0).cpu().numpy())
            self.masks = np.concatenate(mask_list, axis=0)
            
        # Save history states
        self.action_history.append('predict')
        self.update_display()
        
    def propagate_annotations(self):
        """Propagate current annotations to subsequent frames"""
        if self.masks is None:
            self.log("No mask or segmentation created, cannot propagate")
            return
        
        if self.mode == "video":
            self.backup_video_segments = copy.deepcopy(self.video_segments)
            
            # run propagation throughout the video and collect the results in a (N, H, W)
            self.log("Propogating Video ...")
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
                mask_list = []
                for i, out_obj_id in enumerate(out_obj_ids):
                    mask_list.append((out_mask_logits[i] > 0.0).cpu().numpy())
                self.video_segments[out_frame_idx] = np.concatenate(mask_list, axis=0)
                
        elif self.mode == "live":
            self.live_propagation = True
            
        messagebox.showinfo("Propagation", "Finished, play the video to see results to edit more or add.")
    
    # def undo_inference_state(self):
    #     if self.backup_inference_state:
    #         self.log("Inference state back to previous state")
    #         self.inference_state = self.backup_inference_state
    #         self.reset_segmentation()
    #         self.video_segments = copy.deepcopy(self.backup_video_segments)
    #     else:
    #         self.reset_vid_predictor()
        
    def reset_vid_predictor(self):
        self.log("Inference state entirely reseted")
        
        # Live video
        if self.mode in ["live"]:
            try:
                self.predictor.reset_state()
            except:
                self.log("[reset_state] skipped", debug_only=True)
            self.live_propagation = False
            
        # Standard fixed image/video
        elif self.mode in ["image", 'video']:
            self.predictor.reset_state(self.inference_state)
            
        self.reset_segmentation()
        self.video_segments = {}
        self.backup_video_segments = {}
        
    def save_masks_img(self):
        
        # Process input variables first
        if self.masks is None:
            messagebox.showwarning("Warning", "No masks to save.")
            return
        
        self.log("Saving image data...")
        masks_np = self.masks
            
        output_dir = filedialog.askdirectory(title="Select Output Directory")
        if not output_dir: return

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
            
        # Create Images
        num_masks, height, width = masks_np.shape
        grayscale_img = np.zeros((height, width), dtype=np.uint8)
        color_img = np.zeros((height, width, 3), dtype=np.uint8)
        
        for i in range(num_masks):
            mask_value = i + 1
            mask = masks_np[i]
            grayscale_img[mask > 0] = mask_value
            
            color = COLORS[i % len(COLORS)]
            for channel in range(3):
                color_img[mask > 0, channel] = color[channel]
                
        self.canvas.draw()
        overlay_img = np.array(self.canvas.renderer.buffer_rgba())
        
        # Determine base filename
        if self.file_path:
            base_name = self.file_path.stem
        
        # Save grayscale, colored, overlay images
        def save_data(img, format): 
            (output_dir / format).mkdir(exist_ok=True)
            if format == "npy":
                path = output_dir / format / f"{base_name}.npy"
                np.save(path, masks_np) # (N, H, W)
            else:
                path = output_dir / format / f"{base_name}.png"
                Image.fromarray(img).save(path)
            return path
            
        grayscale_path = save_data(grayscale_img, "binary")
        color_path = save_data(color_img, "color", )
        overlay_path = save_data(overlay_img, "overlay")
        npy_path = save_data(masks_np, "npy")
        
        # Save transparent masks if the option is checked
        if self.save_transparent_var.get():
            from tools import save_transparent_masks_img
            (output_dir / "transparent").mkdir(exist_ok=True)
            transparent_path = output_dir / "transparent" / f"{base_name}_transparent.png"
            save_transparent_masks_img(self.image, masks_np, transparent_path)
            messagebox.showinfo("Success", f"Masks saved to:\n\n{grayscale_path}\n\n{color_path}\n\n{overlay_path}\n\n{npy_path}\n\n{transparent_path}")
        else:
            messagebox.showinfo("Success", f"Masks saved to:\n\n{grayscale_path}\n\n{color_path}\n\n{overlay_path}\n\n{npy_path}")
        
    def save_masks_vid(self):
        
        if self.video_segments == {}:
            messagebox.showwarning("Warning", "No masks to save.")
            return
        
        self.log("Saving video data...")
        image_paths = [self.frame_paths[idx] for idx in range(self.total_frames)]
        
        # Save transparent video if the option is checked
        if self.save_transparent_var.get():
            from tools import save_green_screen_masks
            transparent_vid_path = self.video_dir / "raw_folder" / f"{self.video_dir.stem}_transparent.mp4"
            save_green_screen_masks(image_paths, self.video_segments, transparent_vid_path)
            if self.save_raw_data:
                self.log(f"saving to {self.video_dir}")
                torch.save(self.inference_state, self.inf_path)
                with open(self.pkl_path, 'wb') as f: 
                    pickle.dump(self.video_segments, f)
                messagebox.showinfo("Success", f"Video saved to:\n\n{self.vid_path}\n\n{transparent_vid_path}\n\n{self.pkl_path}\n\n{self.inf_path}")
            else:
                messagebox.showinfo("Success", f"Video saved to:\n\n{self.vid_path}\n\n{transparent_vid_path}")
        else:
            overlay_masks_on_images(image_paths, self.video_segments, self.vid_path, alpha=self.alpha, show_borders=self.borders)
            if self.save_raw_data:
                self.log(f"saving to {self.video_dir}")
                torch.save(self.inference_state, self.inf_path)
                with open(self.pkl_path, 'wb') as f: 
                    pickle.dump(self.video_segments, f)
                messagebox.showinfo("Success", f"Video saved to:\n\n{self.vid_path}\n\n{self.pkl_path}\n\n{self.inf_path}")
            else:
                messagebox.showinfo("Success", f"Video saved to:\n\n{self.vid_path}")
        
    def load_saved_data(self):
        if self.inf_path.is_file():
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.inference_state = torch.load(self.inf_path)
                self.log("loaded inference state")
        if self.pkl_path.is_file():
            self.log("loaded video segments")
            with open(self.pkl_path, 'rb') as f:
                self.video_segments = pickle.load(f)
        self.update_display()
        self.show_current_frame()
            
    def show_about(self):
        about_text = """Interactive Segmentation Tool\n
A GUI application for interactive image and video segmentation.\n
Features:
- Interactive point and box selection
- Multiple mask creation
- Undo/Redo functionality
- Video frame-by-frame annotation
- Annotation propagation between frames
- Save masks as grayscale or color images\n
Keyboard Shortcuts:
- Left Click: Add point
- Ctrl+Left Click+Drag: Add box
- Enter: Predict mask
- a: Add new mask
- d: Delete current mask
- Shift: Toggle point label
- Ctrl+z: Undo
- Ctrl+y: Redo
- Ctrl+s: Save masks
- Space: Play/Pause video"""
        messagebox.showinfo("About SAM2 Segmentation Tool", about_text)


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent)
    root = tk.Tk()
    app = SAM2SegmentationApp(root, debug=True)
    root.mainloop()