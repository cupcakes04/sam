import os
import json
import pickle
import base64
import shutil
from io import BytesIO
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import copy

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
from werkzeug.utils import secure_filename

from sam2.build_sam import build_sam2, build_sam2_video_predictor, build_sam2_camera_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tools import setup_device, COLORS, overlay_masks_on_images, save_green_screen_masks

app = Flask(__name__, 
            static_folder='webapp/static',
            template_folder='webapp/templates')
app.secret_key = 'your-secret-key-here-change-in-production'
CORS(app)

# Configuration
UPLOAD_FOLDER = Path('./webapp/uploads')
VIDEO_FOLDER = Path('./webapp/videos')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VIDEO_FOLDER'] = VIDEO_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max for videos

# Global state (in production, use Redis or database)
sessions = {}

class SAM2Session:
    def __init__(self, session_id, device):
        self.session_id = session_id
        self.device = device
        
        # Model paths
        self.sam2_checkpoint_dir = Path("./sam2/sam/checkpoints")
        self.model_cfg_dir = Path("configs/sam2.1")
        self.model_configs = {
            'tiny': ('sam2.1_hiera_t.yaml', 'sam2.1_hiera_tiny.pt'),
            'small': ('sam2.1_hiera_s.yaml', 'sam2.1_hiera_small.pt'),
            'large': ('sam2.1_hiera_l.yaml', 'sam2.1_hiera_large.pt'),
            'base_plus': ('sam2.1_hiera_b+.yaml', 'sam2.1_hiera_base_plus.pt')
        }
        
        # Initialize with small model by default
        self.current_model = 'small'
        self.load_model('small')
        
        # Session state
        self.image = None
        self.mode = 'image'  # 'image', 'video', or 'webcam'
        self.input_points = [[]]
        self.input_labels = [[]]
        self.input_boxes = [[]]
        self.masks = None
        self.scores = None
        self.mask_idx = 0
        self.total_masks = 1
        self.alpha = 0.4
        self.borders = True
        
        # Webcam state
        self.webcam_active = False
        self.webcam_frame = None
        self.live_propagation = False
        
        # Video state
        self.video_path = None
        self.video_dir = None
        self.frame_paths = []
        self.current_frame_idx = 0
        self.total_frames = 0
        self.video_segments = {}  # {frame_idx: masks}
        self.backup_video_segments = {}
        self.inference_state = None
        self.extraction_percent = 100
        
        # Webcam predictor (for live tracking)
        self.webcam_predictor = None
        
    def load_model(self, model_name):
        """Load SAM2 model"""
        if model_name not in self.model_configs:
            raise ValueError(f"Unknown model: {model_name}")
        
        cfg_name, ckpt_name = self.model_configs[model_name]
        config_file = str(self.model_cfg_dir / cfg_name)
        checkpoint_file = str(self.sam2_checkpoint_dir / ckpt_name)
        
        self.sam2_img_model = build_sam2(config_file, checkpoint_file, device=self.device)
        self.sam2_vid_model = build_sam2_video_predictor(config_file, checkpoint_file, device=self.device)
        self.sam2_cam_model = build_sam2_camera_predictor(config_file, checkpoint_file, device=self.device)
        self.predictor = None
        self.current_model = model_name
        
    def reset_segmentation(self, reset_masks=True):
        """Reset segmentation state"""
        self.input_points = [[]]
        self.input_labels = [[]]
        self.input_boxes = [[]]
        if reset_masks:
            self.masks = None
            self.total_masks = 1
        self.scores = None
        self.mask_idx = 0
        
    def extract_video_frames(self, extraction_percent=100):
        """Extract video frames with percentage control"""
        if not 0 < extraction_percent <= 100:
            raise ValueError("extraction_percent must be between 0 and 100")
        
        raw_folder = self.video_dir / "raw_folder"
        if not raw_folder.is_dir():
            cap = cv2.VideoCapture(str(self.video_path))
            
            i = 0
            saved_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_interval = max(1, round(100 / extraction_percent))
                if i % frame_interval == 0:
                    frame_file = self.video_dir / f"{saved_count:05d}.jpg"
                    cv2.imwrite(str(frame_file), frame)
                    saved_count += 1
                i += 1
                
            cap.release()
            raw_folder.mkdir(exist_ok=True)
        
        self.frame_paths = sorted([
            p for p in self.video_dir.iterdir()
            if p.suffix.lower() in [".jpg", ".jpeg"]
        ])
        self.total_frames = len(self.frame_paths)
        
    def get_current_frame_image(self):
        """Get current frame as numpy array"""
        if 0 <= self.current_frame_idx < len(self.frame_paths):
            frame_path = self.frame_paths[self.current_frame_idx]
            return np.array(Image.open(frame_path).convert('RGB'))
        return None
        
    def reset_video_predictor(self):
        """Reset video predictor state"""
        if self.mode == 'video' and self.inference_state is not None:
            self.predictor.reset_state(self.inference_state)
        self.reset_segmentation()
        self.video_segments = {}
        self.backup_video_segments = {}
    
    def predict_mask(self):
        
        if not self.input_points and not self.input_boxes:
            return jsonify({'error': 'No annotations provided'}), 400
        
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
                points_np, labels_np, box_np = process_anns(points, labels, boxes)
                
                if points_np is None and box_np is None:
                    continue
                
                # Predict
                mask, score, _ = self.predictor.predict(
                    point_coords=points_np,
                    point_labels=labels_np,
                    box=box_np,
                    multimask_output=False
                )
                mask_list.append(mask)
                scores.append(float(score[0]))
            
            if mask_list:
                self.masks = np.concatenate(mask_list, axis=0)
                self.scores = scores
        
        elif self.mode == 'video':
            for ann_obj_id, (points, labels, boxes) in enumerate(zip(self.input_points, self.input_labels, self.input_boxes)):
                points_np, labels_np, box_np = process_anns(points, labels, boxes)
                print("id:", ann_obj_id, "pt:", points_np, "lb:", labels_np, "bx:", box_np)
                
                if all(x is None for x in [points_np, labels_np, box_np]):
                    continue
                
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                    inference_state=self.inference_state,
                    frame_idx=self.current_frame_idx,
                    obj_id=ann_obj_id,
                    points=points_np,
                    labels=labels_np,
                    box=box_np,
                )
            
            # Assign current masks
            for i, out_obj_id in enumerate(out_obj_ids):
                mask_list.append((out_mask_logits[i] > 0.0).cpu().numpy())
            
            if mask_list:
                self.masks = np.concatenate(mask_list, axis=0)
                # Store masks for current frame
                self.video_segments[self.current_frame_idx] = self.masks
        
        if self.mode == "webcam":
            
            self.predictor.load_first_frame(self.webcam_frame)
            for ann_obj_id, (points, labels, boxes) in enumerate(zip(self.input_points, self.input_labels, self.input_boxes)):
                points_np, labels_np, box_np = process_anns(points, labels, boxes)
                print("id:", ann_obj_id, "pt:", points_np, "lb:", labels_np, "bx:", box_np)
                
                if all(x is None for x in [points_np, labels_np, box_np]):
                    continue
                
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                    frame_idx=0,
                    obj_id=ann_obj_id,
                    points=points_np,
                    labels=labels_np,
                    bbox=box_np,
                )
                
            # Assign current masks
            for i, out_obj_id in enumerate(out_obj_ids):
                mask_list.append((out_mask_logits[i] > 0.0).cpu().numpy())
                
            if mask_list:
                self.masks = np.concatenate(mask_list, axis=0)
                
        self.scores = scores
    
    def propagate_annotations(self):
        # Backup current segments
        self.backup_video_segments = copy.deepcopy(self.video_segments)
    
        # Run propagation throughout the video
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            mask_list = []
            for i, out_obj_id in enumerate(out_obj_ids):
                mask_list.append((out_mask_logits[i] > 0.0).cpu().numpy())
            if mask_list:
                self.video_segments[out_frame_idx] = np.concatenate(mask_list, axis=0)
    
    def initialize_webcam(self):
        """Initialize webcam mode"""
        self.mode = 'webcam'
        
        self.predictor = self.sam2_cam_model
        self.inference_state = None
        self.reset_video_predictor()
        self.webcam_active = True
        self.live_propagation = False
        
        # Frame processing state
        self.processing_frame = False
        self.latest_frame_data = None
        self.frame_counter = 0
        
    def stop_webcam(self):
        """Stop webcam mode"""
        self.webcam_active = False
        self.live_propagation = False
        self.webcam_frame = None
        
    def queue_webcam_frame(self, frame_data):
        """Queue the latest frame for processing (non-blocking)"""
        self.frame_counter += 1
        self.latest_frame_data = frame_data
        
        # If not currently processing, start processing the latest frame
        if not self.processing_frame:
            return self.process_latest_frame()
        else:
            # Return the last processed frame if still processing
            return self.webcam_frame if self.webcam_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
    
    def process_latest_frame(self):
        """Process the latest queued frame"""
        if self.latest_frame_data is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
            
        try:
            self.processing_frame = True
            frame_data = self.latest_frame_data
            
            # Decode base64 image
            if ',' in frame_data:
                img_data = base64.b64decode(frame_data.split(',')[1])
            else:
                img_data = base64.b64decode(frame_data)
            
            img = Image.open(BytesIO(img_data)).convert('RGB')
            frame = np.array(img)
            
            # Store frame
            self.webcam_frame = frame
            self.image = frame
            
            # If live propagation is enabled and we have masks
            if self.live_propagation and self.masks is not None:
                try:
                    # Use camera predictor for live tracking
                    out_obj_ids, out_mask_logits = self.predictor.track(frame)

                    # Assign current masks
                    mask_list = []
                    for i, out_obj_id in enumerate(out_obj_ids):
                        mask_list.append((out_mask_logits[i] > 0.0).cpu().numpy())
                    
                    if mask_list:
                        self.masks = np.concatenate(mask_list, axis=0)
                    
                except Exception as e:
                    print(f"Tracking error: {e}")
            
            return frame
            
        except Exception as e:
            print(f"Frame processing error: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((480, 640, 3), dtype=np.uint8)
        finally:
            self.processing_frame = False
        
    def start_webcam_tracking(self):
        """Start live tracking on webcam"""
        if self.mode == 'webcam' and self.webcam_frame is not None and self.masks is not None:
            self.live_propagation = True
            return True
        return False
    
    def stop_webcam_tracking(self):
        """Stop live tracking but keep webcam active"""
        self.live_propagation = False
        # Keep masks and webcam_frame so user can continue annotating

def get_session(session_id):
    """Get or create session"""
    if session_id not in sessions:
        device = setup_device()
        sessions[session_id] = SAM2Session(session_id, device)
    return sessions[session_id]

def numpy_to_base64(img_array):
    """Convert numpy array to base64 string"""
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    img = Image.fromarray(img_array)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def create_overlay_image(image, masks, points, labels, boxes, alpha=0.4, borders=True):
    """Create visualization with masks, points, and boxes"""
    h, w = image.shape[:2]
    overlay = image.copy()
    
    # Draw masks
    if masks is not None:
        for i, mask in enumerate(masks):
            color = COLORS[i % len(COLORS)]
            mask_bool = mask > 0
            overlay[mask_bool] = (overlay[mask_bool] * (1 - alpha) + 
                                 np.array(color) * alpha).astype(np.uint8)
            
            # Draw borders
            if borders:
                mask_uint8 = (mask > 0).astype(np.uint8)
                contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, 
                                              cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)
    
    # Draw points
    for coords, labs in zip(points, labels):
        if not coords:
            continue
        for coord, label in zip(coords, labs):
            x, y = int(coord[0]), int(coord[1])
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(overlay, (x, y), 8, color, -1)
            cv2.circle(overlay, (x, y), 8, (255, 255, 255), 2)
    
    # Draw boxes
    for box in boxes:
        if not box:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return overlay

@app.route('/')
def index():
    """Main page"""
    if 'session_id' not in session:
        session['session_id'] = os.urandom(16).hex()
    return render_template('index.html')

@app.route('/api/upload_image', methods=['POST'])
def upload_image():
    """Upload and process image"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save and load image
    filename = secure_filename(file.filename)
    filepath = app.config['UPLOAD_FOLDER'] / filename
    file.save(filepath)
    
    sess.image = np.array(Image.open(filepath).convert('RGB'))
    sess.mode = 'image'
    sess.reset_segmentation()
    
    # Initialize predictor
    sess.predictor = SAM2ImagePredictor(sess.sam2_img_model)
    sess.predictor.set_image(sess.image)
    
    # Return image
    img_base64 = numpy_to_base64(sess.image)
    return jsonify({
        'success': True,
        'image': img_base64,
        'width': sess.image.shape[1],
        'height': sess.image.shape[0]
    })

@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Upload and process video"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    # Save video file
    filename = secure_filename(file.filename)
    video_path = app.config['VIDEO_FOLDER'] / filename
    file.save(video_path)
    
    # Setup video session
    sess.video_path = video_path
    sess.mode = 'video'
    sess.video_dir = app.config['VIDEO_FOLDER'] / video_path.stem
    print(sess.video_dir)
    sess.video_dir.mkdir(exist_ok=True)
    sess.current_frame_idx = 0
    sess.reset_segmentation()
    
    # Extract frames
    extraction_percent = request.form.get('extraction_percent', 100)
    sess.extraction_percent = float(extraction_percent)
    sess.extract_video_frames(sess.extraction_percent)
    
    # Initialize video predictor
    sess.predictor = sess.sam2_vid_model
    sess.inference_state = sess.predictor.init_state(video_path=str(sess.video_dir))
    sess.reset_video_predictor()
    
    # Get first frame
    sess.image = sess.get_current_frame_image()
    
    if sess.image is not None:
        img_base64 = numpy_to_base64(sess.image)
        return jsonify({
            'success': True,
            'image': img_base64,
            'width': sess.image.shape[1],
            'height': sess.image.shape[0],
            'total_frames': sess.total_frames,
            'current_frame': sess.current_frame_idx + 1
        })
    else:
        return jsonify({'error': 'Failed to extract video frames'}), 400

@app.route('/api/add_point', methods=['POST'])
def add_point():
    """Add annotation point"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    data = request.json
    x, y = data['x'], data['y']
    label = data['label']  # 1 for positive, 0 for negative
    
    sess.input_points[sess.mask_idx].append([x, y])
    sess.input_labels[sess.mask_idx].append(label)
    
    # Create overlay
    overlay = create_overlay_image(
        sess.image, sess.masks, sess.input_points, 
        sess.input_labels, sess.input_boxes, sess.alpha, sess.borders
    )
    
    return jsonify({
        'success': True,
        'image': numpy_to_base64(overlay)
    })

@app.route('/api/add_box', methods=['POST'])
def add_box():
    """Add bounding box"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    data = request.json
    x1, y1, x2, y2 = data['x1'], data['y1'], data['x2'], data['y2']
    
    sess.input_boxes[sess.mask_idx] = [
        min(x1, x2), min(y1, y2), 
        max(x1, x2), max(y1, y2)
    ]
    
    # Create overlay
    overlay = create_overlay_image(
        sess.image, sess.masks, sess.input_points, 
        sess.input_labels, sess.input_boxes, sess.alpha, sess.borders
    )
    
    return jsonify({
        'success': True,
        'image': numpy_to_base64(overlay)
    })

@app.route('/api/navigate_frame', methods=['POST'])
def navigate_frame():
    """Navigate to specific frame in video"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    if sess.mode != 'video':
        return jsonify({'error': 'Not in video mode'}), 400
    
    data = request.json
    frame_idx = data.get('frame_idx', 0)
    
    # Clamp frame index
    frame_idx = max(0, min(frame_idx, sess.total_frames - 1))
    sess.current_frame_idx = frame_idx
    
    # Get frame image
    sess.image = sess.get_current_frame_image()
    
    # Check if this frame has existing masks
    if frame_idx in sess.video_segments:
        sess.masks = sess.video_segments[frame_idx]
    else:
        sess.masks = None
    
    # Reset input annotations for clean display (similar to GUI behavior)
    sess.input_points = [[]]
    sess.input_labels = [[]]
    sess.input_boxes = [[]]
    
    # Create overlay
    if sess.image is not None:
        overlay = create_overlay_image(
            sess.image, sess.masks, sess.input_points, 
            sess.input_labels, sess.input_boxes, sess.alpha, sess.borders
        )
        
        return jsonify({
            'success': True,
            'image': numpy_to_base64(overlay),
            'current_frame': sess.current_frame_idx + 1,
            'total_frames': sess.total_frames,
            'has_masks': sess.masks is not None
        })
    
    return jsonify({'error': 'Failed to load frame'}), 400

@app.route('/api/propagate_masks', methods=['POST'])
def propagate():
    """Propagate masks through video frames"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    if sess.mode != 'video':
        return jsonify({'error': 'Not in video mode'}), 400
    
    if sess.masks is None:
        return jsonify({'error': 'No masks to propagate'}), 400
    
    # try:
    sess.propagate_annotations()
    
    return jsonify({
        'success': True,
        'message': 'Masks propagated successfully',
        'frames_processed': len(sess.video_segments)
    })
    
    # except Exception as e:
    #     return jsonify({'error': f'Propagation failed: {str(e)}'}), 500

@app.route('/api/reset_video', methods=['POST'])
def reset_video():
    """Reset video predictor state"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    if sess.mode != 'video':
        return jsonify({'error': 'Not in video mode'}), 400
    
    sess.reset_video_predictor()
    
    # Reinitialize video predictor
    if sess.video_dir:
        sess.inference_state = sess.predictor.init_state(video_path=str(sess.video_dir))
    
    # Get current frame
    sess.image = sess.get_current_frame_image()
    
    if sess.image is not None:
        img_base64 = numpy_to_base64(sess.image)
        return jsonify({
            'success': True,
            'image': img_base64,
            'message': 'Video predictor reset'
        })
    
    return jsonify({'error': 'Failed to reset video'}), 400

@app.route('/api/predict', methods=['POST'])
def predict():
    """Run mask prediction"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    sess.predict_mask()
    
    # Create overlay
    overlay = create_overlay_image(
        sess.image, sess.masks, sess.input_points, 
        sess.input_labels, sess.input_boxes, sess.alpha, sess.borders
    )
    
    return jsonify({
        'success': True,
        'image': numpy_to_base64(overlay),
        'scores': sess.scores if sess.mode == 'image' else [],
        'current_frame': sess.current_frame_idx + 1 if sess.mode == 'video' else None
    })

@app.route('/api/add_mask', methods=['POST'])
def add_mask():
    """Add new mask slot"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    sess.input_points.append([])
    sess.input_labels.append([])
    sess.input_boxes.append([])
    sess.mask_idx += 1
    sess.total_masks += 1
    
    return jsonify({
        'success': True,
        'mask_idx': sess.mask_idx,
        'total_masks': sess.total_masks
    })

@app.route('/api/delete_mask', methods=['POST'])
def delete_mask():
    """Delete current mask"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    if sess.total_masks > 1:
        sess.input_points.pop()
        sess.input_labels.pop()
        sess.input_boxes.pop()
        sess.mask_idx = max(0, sess.mask_idx - 1)
        sess.total_masks -= 1
    
    return jsonify({
        'success': True,
        'mask_idx': sess.mask_idx,
        'total_masks': sess.total_masks
    })

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset all annotations"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    if sess.mode == 'video':
        sess.reset_video_predictor()
        # Reinitialize video predictor
        if sess.video_dir:
            sess.inference_state = sess.predictor.init_state(video_path=str(sess.video_dir))
    else:
        sess.reset_segmentation()
    
    if sess.image is not None:
        img_base64 = numpy_to_base64(sess.image)
        return jsonify({
            'success': True,
            'image': img_base64
        })
    
    return jsonify({'success': True})

@app.route('/api/update_settings', methods=['POST'])
def update_settings():
    """Update visualization settings"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    data = request.json
    if 'alpha' in data:
        sess.alpha = float(data['alpha'])
    if 'borders' in data:
        sess.borders = bool(data['borders'])
    if 'extraction_percent' in data:
        sess.extraction_percent = float(data['extraction_percent'])
    
    # Refresh display
    if sess.image is not None:
        overlay = create_overlay_image(
            sess.image, sess.masks, sess.input_points, 
            sess.input_labels, sess.input_boxes, sess.alpha, sess.borders
        )
        
        return jsonify({
            'success': True,
            'image': numpy_to_base64(overlay)
        })
    
    return jsonify({'success': True})

@app.route('/api/get_session_info', methods=['GET'])
def get_session_info():
    """Get current session information"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    return jsonify({
        'mode': sess.mode,
        'current_frame': sess.current_frame_idx + 1 if sess.mode == 'video' else None,
        'total_frames': sess.total_frames if sess.mode == 'video' else None,
        'has_masks': sess.masks is not None,
        'mask_count': len(sess.masks) if sess.masks is not None else 0,
        'video_segments_count': len(sess.video_segments) if sess.mode == 'video' else None
    })

@app.route('/api/start_webcam', methods=['POST'])
def start_webcam():
    """Initialize webcam mode"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    sess.initialize_webcam()
    
    return jsonify({
        'success': True,
        'message': 'Webcam mode initialized'
    })

@app.route('/api/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop webcam mode"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    sess.stop_webcam()
    
    return jsonify({
        'success': True,
        'message': 'Webcam stopped'
    })

@app.route('/api/process_webcam_frame', methods=['POST'])
def process_webcam_frame():
    """Process a frame from webcam"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    if sess.mode != 'webcam':
        return jsonify({'error': 'Not in webcam mode'}), 400
    
    data = request.json
    frame_data = data.get('frame')
    
    if not frame_data:
        return jsonify({'error': 'No frame data provided'}), 400
    
    try:
        # Queue the frame for processing (non-blocking)
        frame = sess.queue_webcam_frame(frame_data)
        # print(f"Processed frame shape: {frame.shape}, dtype: {frame.dtype}, min: {frame.min()}, max: {frame.max()}")
        
        # Create overlay with current annotations and masks
        overlay = create_overlay_image(
            frame, sess.masks, sess.input_points, 
            sess.input_labels, sess.input_boxes, sess.alpha, sess.borders
        )
        # print(f"Overlay shape: {overlay.shape}, dtype: {overlay.dtype}, min: {overlay.min()}, max: {overlay.max()}")
        
        return jsonify({
            'success': True,
            'image': numpy_to_base64(overlay),
            'width': frame.shape[1],
            'height': frame.shape[0],
            'live_tracking': sess.live_propagation
        })
        
    except Exception as e:
        print(f"Webcam frame processing error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Failed to process frame: {str(e)}'}), 500

@app.route('/api/start_webcam_tracking', methods=['POST'])
def start_webcam_tracking():
    """Start live tracking on webcam"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    if sess.mode != 'webcam':
        return jsonify({'error': 'Not in webcam mode'}), 400
    
    if sess.masks is None:
        return jsonify({'error': 'No masks to track. Please predict masks first.'}), 400
    
    try:
        sess.start_webcam_tracking()
        
        return jsonify({
            'success': True,
            'message': 'Live tracking started',
            'live_tracking': sess.live_propagation
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start tracking: {str(e)}'}), 500

@app.route('/api/stop_webcam_tracking', methods=['POST'])
def stop_webcam_tracking():
    """Stop live tracking on webcam"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    if sess.mode != 'webcam':
        return jsonify({'error': 'Not in webcam mode'}), 400
    
    try:
        sess.stop_webcam_tracking()
        
        return jsonify({
            'success': True,
            'message': 'Live tracking stopped - webcam continues',
            'live_tracking': sess.live_propagation
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to stop tracking: {str(e)}'}), 500

@app.route('/api/save_masks', methods=['POST'])
def save_masks():
    """Save masks as image or video"""
    session_id = session.get('session_id')
    sess = get_session(session_id)
    
    data = request.json
    save_type = data.get('type', 'overlay')  # overlay, binary, color, transparent
    
    if sess.mode == 'image':
        if sess.masks is None:
            return jsonify({'error': 'No masks to save'}), 400
        
        masks_np = sess.masks
        N, H, W = masks_np.shape
        
        if save_type == 'binary':
            # Grayscale mask
            output = np.zeros((H, W), dtype=np.uint8)
            for i in range(N):
                output[masks_np[i] > 0] = i + 1
            img = Image.fromarray(output)
        
        elif save_type == 'color':
            # Colored mask
            output = np.zeros((H, W, 3), dtype=np.uint8)
            for i in range(N):
                mask = masks_np[i]
                color = COLORS[i % len(COLORS)]
                for c in range(3):
                    output[mask > 0, c] = color[c]
            img = Image.fromarray(output)
        
        elif save_type == 'transparent':
            # Transparent PNG
            output = np.zeros((H, W, 4), dtype=np.uint8)
            output[:, :, :3] = sess.image
            combined_mask = np.zeros((H, W), dtype=bool)
            for i in range(N):
                combined_mask = np.logical_or(combined_mask, masks_np[i] > 0)
            output[combined_mask, 3] = 255
            img = Image.fromarray(output, mode='RGBA')
        
        else:  # overlay
            overlay = create_overlay_image(
                sess.image, sess.masks, [], [], [], sess.alpha, sess.borders
            )
            img = Image.fromarray(overlay)
        
        # Save to bytes
        buffered = BytesIO()
        img.save(buffered, format='PNG')
        buffered.seek(0)
        
        return send_file(buffered, mimetype='image/png', 
                        as_attachment=True, 
                        download_name=f'mask_{save_type}.png')
    
    elif sess.mode == 'video':
        if not sess.video_segments:
            return jsonify({'error': 'No video masks to save'}), 400
        
        try:
            # Prepare image paths
            image_paths = [sess.frame_paths[idx] for idx in range(sess.total_frames)]
            
            # Generate output path
            output_path = sess.video_dir / "raw_folder" / f"{sess.video_dir.stem}_{save_type}.mp4"
            output_path.parent.mkdir(exist_ok=True)
            
            if save_type == 'transparent':
                save_green_screen_masks(image_paths, sess.video_segments, output_path)
            else:
                overlay_masks_on_images(
                    image_paths, sess.video_segments, output_path, 
                    alpha=sess.alpha, show_borders=sess.borders
                )
            
            return send_file(output_path, as_attachment=True, 
                           download_name=f'video_{save_type}.mp4')
        
        except Exception as e:
            return jsonify({'error': f'Failed to save video: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid mode or no data to save'}), 400

if __name__ == '__main__':
    os.chdir(Path(__file__).resolve().parent)
    app.run(debug=True, host='0.0.0.0', port=5000)