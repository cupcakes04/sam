# SAM2 Interactive Segmentation Tool

This repository contains an interactive segmentation tool built using Meta's Segment Anything Model 2 (SAM2). It provides a graphical user interface for performing segmentation on images, videos, and live camera feeds.

**Please refer to this repo to set up sam2 modules first**
https://github.com/facebookresearch/sam2

insert the .pt files inside `sam2/sam/checkpoints`

## Features

- **Interactive Segmentation**: Click on images to add positive/negative points for segmentation
- **Box Selection**: Control + click and drag to create bounding boxes for segmentation
- **Multi-Object Support**: Create and manage multiple segmentation masks simultaneously
- **Image Segmentation**: Process single images with point and box prompts
- **Video Segmentation**: Annotate videos frame-by-frame with ability to propagate annotations across frames
- **Live Camera Segmentation**: Real-time segmentation from camera feed, screen capture, or video files
- **Export Options**: Save segmentation masks in multiple formats (binary, color, overlay, numpy arrays)
- **Video Export**: Create annotated videos with segmentation overlays
- **Transparent Export**: Save transparent masks where only segmented pixels remain and the rest are alpha = 0. Enabled via a "Save Transparent" checkbox in the GUI.
- **Model Selection**: Support for different SAM2 model variants (tiny, small, large, base+)
- **Frame Extraction Control**: Configure percentage of frames to extract from videos
- **Undo/Redo**: Comprehensive history with undo/redo functionality
- **Keyboard Shortcuts**: Various shortcuts for efficient operation

## Requirements

- Python 3.8+
- PyTorch
- SAM2 models and checkpoints
- OpenCV
- Tkinter
- Matplotlib
- NumPy
- PIL/Pillow
- Hydra
- OmegaConf
- mss (for screen capture)

## Directory Structure

```
SAM/
├── gui.py              # Main GUI application
├── tools.py            # Utility functions for visualization and processing
├── workspace.ipynb     # Jupyter notebook for experimentation
├── sam2/               # SAM2 core modules and models
├── input/              # Input images and videos (subdirectories: img/, vid/)
├── output/             # Output results (subdirectories: img/, vid/)
├── sample_data/        # Sample input data for testing
└── configs/            # Configuration files for different SAM2 models
```

## Setup

1. Install required dependencies
2. Download SAM2 model checkpoints to `./sam2/sam/checkpoints/`
3. Run the application:

```bash
python gui.py
```

## Usage

### Image Mode
1. Select "Image Mode" from the controls
2. Open an image file using "File > Open Image"
3. Click to add positive points (green stars)
4. Hold Shift to toggle to negative points (red stars)
5. Use Ctrl+click+drag to create bounding boxes
6. Press Enter or click "Predict Mask" to generate segmentation
7. Use "A" to add new mask or "D" to delete current mask
8. Adjust overlay alpha and borders as needed
9. Save results via "File > Save Masks Image"

### Video Mode
1. Select "Video Mode" from the controls
2. Open a video file using "File > Open Video"
3. Choose an output directory for extracted frames
4. Add annotations to the current frame (similar to Image Mode)
5. Use propagation to extend annotations across the video
6. Navigate frames with the slider or navigation buttons
7. Choose save modes (*Save Transparent* mode creates green screen mp4 files)
8. Save results via "File > Save Masks Video" (video results will be in *video_name/raw_folder/*)

### Live Mode
1. Select "Live Mode" from the controls
2. Choose video source (Camera, Screen, or Video File, curently only camera supported)
3. Add initial annotations
4. Start the stream and optionally enable live propagation
5. Adjust delay for processing speed

## Keyboard Shortcuts

- Left Click: Add positive point
- Ctrl + Left Click + Drag: Create bounding box
- Enter: Predict mask
- A: Add new mask
- D: Delete current mask
- Shift: Toggle point label (positive/negative)
- Ctrl+Z: Undo
- Ctrl+Y: Redo
- Ctrl+S: Save masks
- Space: Play/Pause video (in video mode)

## Model Selection

The application supports multiple SAM2 model variants:
- SAM2 Hiera Tiny (fastest, smallest)
- SAM2 Hiera Small (good balance of speed and accuracy)
- SAM2 Hiera Base+ (better accuracy)
- SAM2 Hiera Large (highest accuracy)

Choose your preferred model in "File > Insert Model".

## Output Formats

When saving masks:
- **Binary**: Grayscale mask with different values for each segmented object
- **Color**: Colored mask with different colors for each object
- **Overlay**: Original image with segmentation overlay
- **Numpy**: Raw segmentation data in numpy array format
- **Transparent**: Images that have the background set to transparent

## Note to self:
1) Make changes (edit code, add files, delete stuff, etc).

2) Stage changes (tell Git which changes to include):

    - `git add .`   (commit all)
    - `git add app.py`  (one file)
    - `git add app.py README.md utils/helpers.py`   (3 files)
    - `git add src/`    (everything inside folder)
    - `git add *.py`    (certain files only)
    - `git reset`   (remove everything from staging)
    - `git reset HEAD app.py` (remove one from staging)

3) Commit (save a snapshot in history):

    `git commit -m "Describe what you changed"`

4) Push to GitHub (sync your commits):

    `git push`
    
Repeat steps 1 → 4 every time you’re done with a piece of work.