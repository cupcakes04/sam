// Global state
let canvas, ctx;
let currentImage = null;
let currentLabel = 1; // 1 = positive, 0 = negative
let isDrawingBox = false;
let boxStart = null;
let pointCount = 0;
let currentMaskIdx = 0;
let totalMasks = 1;
let imageScale = 1;
let imageOffsetX = 0;
let imageOffsetY = 0;

// Video state
let currentMode = 'image'; // 'image', 'video', or 'webcam'
let currentFrame = 1;
let totalFrames = 1;
let extractionPercent = 100;
let isPlaying = false;
let playbackSpeed = 1;
let playbackInterval = null;

// Webcam state
let webcamStream = null;
let webcamVideo = null;
let webcamActive = false;
let webcamPaused = false;
let webcamFrameInterval = null;
let liveTracking = false;

// Frame processing state
let processingFrame = false;
let frameProcessingTime = 100; // Initial estimate in ms
let adaptiveFrameRate = 100; // Start at 100ms (10 FPS)
let lastFrameTime = 0;

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext('2d');
    
    setupEventListeners();
    updateMaskInfo();
});

// Setup all event listeners
function setupEventListeners() {
    // Mode selection
    document.querySelectorAll('input[name="mode"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            currentMode = e.target.value;
            toggleModeUI();
        });
    });
    
    // File upload
    document.getElementById('uploadBtn').addEventListener('click', uploadImage);
    document.getElementById('uploadVideoBtn').addEventListener('click', uploadVideo);
    
    // Webcam controls
    document.getElementById('startWebcamBtn').addEventListener('click', startWebcam);
    document.getElementById('stopWebcamBtn').addEventListener('click', stopWebcam);
    document.getElementById('pauseWebcamBtn').addEventListener('click', pauseWebcamFeed);
    document.getElementById('resumeWebcamBtn').addEventListener('click', resumeWebcamFeed);
    document.getElementById('startTrackingBtn').addEventListener('click', startLiveTracking);
    document.getElementById('stopTrackingBtn').addEventListener('click', stopLiveTracking);
    
    // Extraction percent slider
    document.getElementById('extractionSlider').addEventListener('input', (e) => {
        extractionPercent = parseInt(e.target.value);
        document.getElementById('extractionValue').textContent = extractionPercent;
    });
    
    // Label selection
    document.querySelectorAll('input[name="label"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            currentLabel = parseInt(e.target.value);
        });
    });
    
    // Action buttons
    document.getElementById('predictBtn').addEventListener('click', predictMask);
    document.getElementById('addMaskBtn').addEventListener('click', addMask);
    document.getElementById('deleteMaskBtn').addEventListener('click', deleteMask);
    document.getElementById('resetBtn').addEventListener('click', resetAnnotations);
    
    // Video action buttons
    document.getElementById('propagateBtn').addEventListener('click', propagateMasks);
    document.getElementById('resetVideoBtn').addEventListener('click', resetVideo);
    
    // Video navigation
    document.getElementById('playPauseBtn').addEventListener('click', togglePlayback);
    document.getElementById('prevFrameBtn').addEventListener('click', () => navigateFrame(-1));
    document.getElementById('nextFrameBtn').addEventListener('click', () => navigateFrame(1));
    document.getElementById('prevTenBtn').addEventListener('click', () => navigateFrame(-10));
    document.getElementById('nextTenBtn').addEventListener('click', () => navigateFrame(10));
    document.getElementById('frameSlider').addEventListener('input', (e) => {
        navigateToFrame(parseInt(e.target.value) - 1);
    });
    
    // Playback speed
    document.getElementById('speedSlider').addEventListener('input', (e) => {
        playbackSpeed = parseFloat(e.target.value);
        document.getElementById('speedValue').textContent = playbackSpeed + 'x';
        if (isPlaying) {
            stopPlayback();
            startPlayback();
        }
    });
    
    // Settings
    document.getElementById('bordersCheck').addEventListener('change', updateSettings);
    document.getElementById('alphaSlider').addEventListener('input', (e) => {
        document.getElementById('alphaValue').textContent = e.target.value;
        updateSettings();
    });
    
    // Save buttons - Image
    document.getElementById('saveOverlay').addEventListener('click', () => saveMasks('overlay'));
    document.getElementById('saveBinary').addEventListener('click', () => saveMasks('binary'));
    document.getElementById('saveColor').addEventListener('click', () => saveMasks('color'));
    document.getElementById('saveTransparent').addEventListener('click', () => saveMasks('transparent'));
    
    // Save buttons - Video
    document.getElementById('saveVideoOverlay').addEventListener('click', () => saveMasks('overlay'));
    document.getElementById('saveVideoTransparent').addEventListener('click', () => saveMasks('transparent'));
    
    // Canvas interactions
    canvas.addEventListener('mousedown', onCanvasMouseDown);
    canvas.addEventListener('mousemove', onCanvasMouseMove);
    canvas.addEventListener('mouseup', onCanvasMouseUp);
    
    // Keyboard shortcuts
    document.addEventListener('keydown', onKeyDown);
}

// Toggle UI based on mode
function toggleModeUI() {
    const imageControls = document.getElementById('imageControls');
    const videoControls = document.getElementById('videoControls');
    const webcamControls = document.getElementById('webcamControls');
    const videoActions = document.getElementById('videoActions');
    const webcamActions = document.getElementById('webcamActions');
    const videoNavigation = document.getElementById('videoNavigation');
    const imageSaveControls = document.getElementById('imageSaveControls');
    const videoSaveControls = document.getElementById('videoSaveControls');
    
    // Hide all controls first
    imageControls.style.display = 'none';
    videoControls.style.display = 'none';
    webcamControls.style.display = 'none';
    videoActions.style.display = 'none';
    webcamActions.style.display = 'none';
    videoNavigation.style.display = 'none';
    imageSaveControls.style.display = 'none';
    videoSaveControls.style.display = 'none';
    
    if (currentMode === 'video') {
        videoControls.style.display = 'block';
        videoActions.style.display = 'block';
        videoNavigation.style.display = 'block';
        videoSaveControls.style.display = 'block';
    } else if (currentMode === 'webcam') {
        webcamControls.style.display = 'block';
        webcamActions.style.display = 'block';
        imageSaveControls.style.display = 'block';
    } else { // image mode
        imageControls.style.display = 'block';
        imageSaveControls.style.display = 'block';
    }
}

// Upload image to server
async function uploadImage() {
    const fileInput = document.getElementById('imageUpload');
    const file = fileInput.files[0];
    
    if (!file) {
        showStatus('Please select an image first', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/upload_image', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentImage = data.image;
            displayImage(data.image, data.width, data.height);
            showStatus('Image loaded successfully!', 'success');
            resetState();
        } else {
            showStatus('Error uploading image', 'error');
        }
    } catch (error) {
        console.error('Upload error:', error);
        showStatus('Failed to upload image', 'error');
    } finally {
        showLoading(false);
    }
}

// Upload video to server
async function uploadVideo() {
    const fileInput = document.getElementById('videoUpload');
    const file = fileInput.files[0];
    
    if (!file) {
        showStatus('Please select a video first', 'error');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    formData.append('extraction_percent', extractionPercent);
    
    showLoading(true);
    showStatus('Extracting video frames...', 'info');
    
    try {
        const response = await fetch('/api/upload_video', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentImage = data.image;
            displayImage(data.image, data.width, data.height);
            
            // Update video state
            currentFrame = data.current_frame;
            totalFrames = data.total_frames;
            
            // Update UI
            updateVideoControls();
            resetState();
            
            showStatus(`Video loaded! ${totalFrames} frames extracted.`, 'success');
        } else {
            showStatus(data.error || 'Error uploading video', 'error');
        }
    } catch (error) {
        console.error('Video upload error:', error);
        showStatus('Failed to upload video', 'error');
    } finally {
        showLoading(false);
    }
}

// Display image on canvas
function displayImage(base64Image, width, height) {
    const img = new Image();
    img.onload = () => {
        // Calculate scaling to fit canvas
        const maxWidth = canvas.parentElement.clientWidth - 40;
        const maxHeight = canvas.parentElement.clientHeight - 40;
        
        imageScale = Math.min(maxWidth / width, maxHeight / height, 1);
        
        canvas.width = width * imageScale;
        canvas.height = height * imageScale;
        
        imageOffsetX = 0;
        imageOffsetY = 0;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
    };
    img.src = 'data:image/png;base64,' + base64Image;
}

// Get mouse position relative to image
function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
    };
}

// Canvas mouse down handler
function onCanvasMouseDown(e) {
    if (!currentImage) return;
    
    const pos = getMousePos(e);
    
    if (e.ctrlKey || e.metaKey) {
        // Start drawing box
        isDrawingBox = true;
        boxStart = pos;
    } else {
        // Add point
        addPoint(pos.x, pos.y, currentLabel);
    }
}

// Canvas mouse move handler
function onCanvasMouseMove(e) {
    if (!isDrawingBox || !boxStart) return;
    
    const pos = getMousePos(e);
    
    // Redraw image and show temporary box
    const img = new Image();
    img.onload = () => {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        
        // Draw temporary box
        ctx.strokeStyle = '#00ff00';
        ctx.lineWidth = 2;
        ctx.strokeRect(
            boxStart.x,
            boxStart.y,
            pos.x - boxStart.x,
            pos.y - boxStart.y
        );
    };
    img.src = 'data:image/png;base64,' + currentImage;
}

// Canvas mouse up handler
function onCanvasMouseUp(e) {
    if (!isDrawingBox || !boxStart) return;
    
    const pos = getMousePos(e);
    
    // Add box
    addBox(boxStart.x, boxStart.y, pos.x, pos.y);
    
    isDrawingBox = false;
    boxStart = null;
}

// Add annotation point
async function addPoint(x, y, label) {
    showLoading(true);
    
    // Convert to original image coordinates
    const originalX = x / imageScale;
    const originalY = y / imageScale;
    
    try {
        const response = await fetch('/api/add_point', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ x: originalX, y: originalY, label: label })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentImage = data.image;
            displayImage(data.image, canvas.width / imageScale, canvas.height / imageScale);
            pointCount++;
            updateMaskInfo();
        }
    } catch (error) {
        console.error('Add point error:', error);
        showStatus('Failed to add point', 'error');
    } finally {
        showLoading(false);
    }
}

// Add bounding box
async function addBox(x1, y1, x2, y2) {
    showLoading(true);
    
    // Convert to original image coordinates
    const originalX1 = x1 / imageScale;
    const originalY1 = y1 / imageScale;
    const originalX2 = x2 / imageScale;
    const originalY2 = y2 / imageScale;
    
    try {
        const response = await fetch('/api/add_box', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                x1: originalX1,
                y1: originalY1,
                x2: originalX2,
                y2: originalY2
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentImage = data.image;
            displayImage(data.image, canvas.width / imageScale, canvas.height / imageScale);
            showStatus('Box added', 'success');
        }
    } catch (error) {
        console.error('Add box error:', error);
        showStatus('Failed to add box', 'error');
    } finally {
        showLoading(false);
    }
}

// Predict mask
async function predictMask() {
    if (pointCount === 0) {
        showStatus('Add some points or boxes first!', 'error');
        return;
    }
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentImage = data.image;
            displayImage(data.image, canvas.width / imageScale, canvas.height / imageScale);
            showStatus('Mask predicted successfully!', 'success');
            
            if (data.scores) {
                console.log('Mask scores:', data.scores);
            }
        } else {
            showStatus(data.error || 'Prediction failed', 'error');
        }
    } catch (error) {
        console.error('Predict error:', error);
        showStatus('Failed to predict mask', 'error');
    } finally {
        showLoading(false);
    }
}

// Add new mask slot
async function addMask() {
    try {
        const response = await fetch('/api/add_mask', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentMaskIdx = data.mask_idx;
            totalMasks = data.total_masks;
            pointCount = 0;
            updateMaskInfo();
            showStatus('New mask slot added', 'success');
        }
    } catch (error) {
        console.error('Add mask error:', error);
        showStatus('Failed to add mask', 'error');
    }
}

// Delete current mask
async function deleteMask() {
    try {
        const response = await fetch('/api/delete_mask', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentMaskIdx = data.mask_idx;
            totalMasks = data.total_masks;
            pointCount = 0;
            updateMaskInfo();
            showStatus('Mask deleted', 'success');
        }
    } catch (error) {
        console.error('Delete mask error:', error);
        showStatus('Failed to delete mask', 'error');
    }
}

// Reset all annotations
async function resetAnnotations() {
    if (!confirm('Reset all annotations?')) return;
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/reset', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.image) {
                currentImage = data.image;
                displayImage(data.image, canvas.width / imageScale, canvas.height / imageScale);
            }
            resetState();
            showStatus('Annotations reset', 'success');
        }
    } catch (error) {
        console.error('Reset error:', error);
        showStatus('Failed to reset', 'error');
    } finally {
        showLoading(false);
    }
}

// Update display settings
async function updateSettings() {
    const borders = document.getElementById('bordersCheck').checked;
    const alpha = parseFloat(document.getElementById('alphaSlider').value);
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/update_settings', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ borders, alpha })
        });
        
        const data = await response.json();
        
        if (data.success && data.image) {
            currentImage = data.image;
            displayImage(data.image, canvas.width / imageScale, canvas.height / imageScale);
        }
    } catch (error) {
        console.error('Update settings error:', error);
    } finally {
        showLoading(false);
    }
}

// Save masks
async function saveMasks(type) {
    showLoading(true);
    
    try {
        const response = await fetch('/api/save_masks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ type })
        });
        
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            
            // Set correct file extension based on mode
            if (currentMode === 'video') {
                a.download = `video_${type}.mp4`;
            } else {
                a.download = `mask_${type}.png`;
            }
            
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            const fileType = currentMode === 'video' ? 'video' : 'mask';
            showStatus(`${type} ${fileType} saved!`, 'success');
        } else {
            const data = await response.json();
            showStatus(data.error || 'Failed to save mask', 'error');
        }
    } catch (error) {
        console.error('Save error:', error);
        showStatus('Failed to save mask', 'error');
    } finally {
        showLoading(false);
    }
}

// Keyboard shortcuts
function onKeyDown(e) {
    const key = e.key.toLowerCase();
    
    if (key === 'enter') {
        e.preventDefault();
        predictMask();
    } else if (key === 'a') {
        e.preventDefault();
        addMask();
    } else if (key === 'd') {
        e.preventDefault();
        deleteMask();
    } else if (key === 'r') {
        e.preventDefault();
        resetAnnotations();
    } else if (key === ' ' || key === 'spacebar') {
        // Space bar for play/pause (video mode only)
        if (currentMode === 'video') {
            e.preventDefault();
            togglePlayback();
        }
    } else if (key === 'arrowleft') {
        // Left arrow for previous frame (video mode only)
        if (currentMode === 'video') {
            e.preventDefault();
            navigateFrame(-1);
        }
    } else if (key === 'arrowright') {
        // Right arrow for next frame (video mode only)
        if (currentMode === 'video') {
            e.preventDefault();
            navigateFrame(1);
        }
    } else if (key === 'shift') {
        // Toggle label
        currentLabel = currentLabel === 1 ? 0 : 1;
        document.querySelector(`input[name="label"][value="${currentLabel}"]`).checked = true;
    }
}

// Video navigation functions
function updateVideoControls() {
    const frameSlider = document.getElementById('frameSlider');
    const currentFrameDisplay = document.getElementById('currentFrameDisplay');
    const totalFramesDisplay = document.getElementById('totalFramesDisplay');
    
    frameSlider.max = totalFrames;
    frameSlider.value = currentFrame;
    frameSlider.disabled = false;
    
    currentFrameDisplay.textContent = currentFrame;
    totalFramesDisplay.textContent = totalFrames;
}

// Video playback functions
function togglePlayback() {
    if (currentMode !== 'video') return;
    
    if (isPlaying) {
        stopPlayback();
    } else {
        startPlayback();
    }
}

function startPlayback() {
    if (currentMode !== 'video' || isPlaying) return;
    
    isPlaying = true;
    updatePlayPauseButton();
    
    const frameDelay = Math.max(100, 1000 / (30 * playbackSpeed)); // Max 30fps, adjustable by speed
    
    playbackInterval = setInterval(async () => {
        if (currentFrame >= totalFrames) {
            stopPlayback();
            return;
        }
        
        await navigateFrame(1);
    }, frameDelay);
}

function stopPlayback() {
    if (!isPlaying) return;
    
    isPlaying = false;
    updatePlayPauseButton();
    
    if (playbackInterval) {
        clearInterval(playbackInterval);
        playbackInterval = null;
    }
}

function updatePlayPauseButton() {
    const btn = document.getElementById('playPauseBtn');
    if (isPlaying) {
        btn.textContent = '⏸ Pause';
        btn.className = 'btn btn-warning';
    } else {
        btn.textContent = '▶ Play';
        btn.className = 'btn btn-primary';
    }
}

async function navigateFrame(delta) {
    const newFrame = Math.max(1, Math.min(currentFrame + delta, totalFrames));
    await navigateToFrame(newFrame - 1);
}

async function navigateToFrame(frameIdx) {
    if (currentMode !== 'video') return;
    
    // Stop playback when manually navigating (except during auto-playback)
    if (isPlaying && !playbackInterval) {
        stopPlayback();
    }
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/navigate_frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ frame_idx: frameIdx })
        });
        
        const data = await response.json();
        
        if (data.success) {
            currentImage = data.image;
            displayImage(data.image, canvas.width / imageScale, canvas.height / imageScale);
            
            currentFrame = data.current_frame;
            totalFrames = data.total_frames;
            
            updateVideoControls();
            
            // Reset segmentation similar to GUI version when playing video
            if (isPlaying) {
                resetSegmentationState();
            } else {
                resetState();
            }
            
            if (!isPlaying) { // Only show status if not auto-playing
                showStatus(`Frame ${currentFrame}/${totalFrames}${data.has_masks ? ' (has masks)' : ''}`, 'info');
            }
        } else {
            showStatus(data.error || 'Failed to navigate frame', 'error');
        }
    } catch (error) {
        console.error('Frame navigation error:', error);
        showStatus('Failed to navigate frame', 'error');
    } finally {
        showLoading(false);
    }
}

// Reset segmentation state similar to GUI version
function resetSegmentationState() {
    // Clear current annotations but preserve mask info
    pointCount = 0;
    updateMaskInfo();
    
    // Clear canvas annotations (points and boxes) but keep the image and existing masks
    if (currentImage) {
        const img = new Image();
        img.onload = () => {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            ctx.drawImage(img, imageOffsetX, imageOffsetY, 
                         img.width * imageScale, img.height * imageScale);
        };
        img.src = 'data:image/png;base64,' + currentImage;
    }
}

async function propagateMasks() {
    if (currentMode !== 'video') return;
    
    showLoading(true);
    showStatus('Propagating masks through video...', 'info');
    
    try {
        const response = await fetch('/api/propagate_masks', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            showStatus(`Masks propagated to ${data.frames_processed} frames!`, 'success');
        } else {
            showStatus(data.error || 'Propagation failed', 'error');
        }
    } catch (error) {
        console.error('Propagation error:', error);
        showStatus('Failed to propagate masks', 'error');
    } finally {
        showLoading(false);
    }
}

async function resetVideo() {
    if (currentMode !== 'video') return;
    
    if (!confirm('Reset all video annotations?')) return;
    
    showLoading(true);
    
    try {
        const response = await fetch('/api/reset_video', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            if (data.image) {
                currentImage = data.image;
                displayImage(data.image, canvas.width / imageScale, canvas.height / imageScale);
            }
            resetState();
            showStatus('Video predictor reset', 'success');
        } else {
            showStatus(data.error || 'Failed to reset video', 'error');
        }
    } catch (error) {
        console.error('Reset video error:', error);
        showStatus('Failed to reset video', 'error');
    } finally {
        showLoading(false);
    }
}

// Helper functions
function updateMaskInfo() {
    document.getElementById('currentMask').textContent = currentMaskIdx + 1;
    document.getElementById('totalMasks').textContent = totalMasks;
    document.getElementById('pointCount').textContent = pointCount;
}

function resetState() {
    pointCount = 0;
    currentMaskIdx = 0;
    totalMasks = 1;
    updateMaskInfo();
}

function showLoading(show) {
    const loading = document.getElementById('loading');
    if (show) {
        loading.classList.add('active');
    } else {
        loading.classList.remove('active');
    }
}

function showStatus(message, type = 'info') {
    const status = document.getElementById('status');
    status.textContent = message;
    status.style.background = type === 'error' ? 'rgba(220, 53, 69, 0.9)' : 
                              type === 'success' ? 'rgba(40, 167, 69, 0.9)' : 
                              'rgba(0, 0, 0, 0.8)';
    status.classList.add('active');
    
    setTimeout(() => {
        status.classList.remove('active');
    }, 3000);
}

// Webcam Functions
async function startWebcam() {
    try {
        showLoading(true);
        
        // Request camera permission and get stream
        webcamStream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 640 }, 
                height: { ideal: 480 },
                facingMode: 'user'
            } 
        });
        
        // Create hidden video element to capture frames
        webcamVideo = document.createElement('video');
        webcamVideo.srcObject = webcamStream;
        webcamVideo.autoplay = true;
        webcamVideo.muted = true;
        webcamVideo.style.display = 'none';
        document.body.appendChild(webcamVideo);
        
        // Wait for video to be ready and playing
        await new Promise((resolve) => {
            webcamVideo.onloadedmetadata = () => {
                webcamVideo.play().then(resolve);
            };
        });
        
        // Wait a bit more for the first frame to be available
        await new Promise(resolve => setTimeout(resolve, 500));
        
        // Initialize webcam mode on server
        const response = await fetch('/api/start_webcam', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            webcamActive = true;
            
            // Update UI
            document.getElementById('startWebcamBtn').style.display = 'none';
            document.getElementById('stopWebcamBtn').style.display = 'inline-block';
            document.getElementById('pauseWebcamBtn').style.display = 'inline-block';
            document.getElementById('resumeWebcamBtn').style.display = 'none';
            document.getElementById('webcamStatus').textContent = '📹 Webcam Active';
            
            // Start frame capture loop
            startFrameCapture();
            
            showStatus('Webcam started successfully', 'success');
        } else {
            throw new Error(data.error || 'Failed to start webcam');
        }
        
    } catch (error) {
        console.error('Webcam error:', error);
        
        if (error.name === 'NotAllowedError') {
            showStatus('Camera permission denied. Please allow camera access and try again.', 'error');
        } else if (error.name === 'NotFoundError') {
            showStatus('No camera found. Please connect a camera and try again.', 'error');
        } else {
            showStatus(`Failed to start webcam: ${error.message}`, 'error');
        }
        
        // Cleanup on error
        stopWebcam();
    } finally {
        showLoading(false);
    }
}

async function stopWebcam() {
    try {
        webcamActive = false;
        webcamPaused = false;
        liveTracking = false;
        processingFrame = false;
        
        // Reset adaptive frame rate
        adaptiveFrameRate = 100;
        
        // Stop frame capture
        if (webcamFrameInterval) {
            clearInterval(webcamFrameInterval);
            webcamFrameInterval = null;
        }
        
        // Stop webcam stream
        if (webcamStream) {
            webcamStream.getTracks().forEach(track => track.stop());
            webcamStream = null;
        }
        
        // Remove video element
        if (webcamVideo) {
            webcamVideo.remove();
            webcamVideo = null;
        }
        
        // Stop webcam on server
        await fetch('/api/stop_webcam', {
            method: 'POST'
        });
        
        // Update UI
        document.getElementById('startWebcamBtn').style.display = 'inline-block';
        document.getElementById('stopWebcamBtn').style.display = 'none';
        document.getElementById('pauseWebcamBtn').style.display = 'none';
        document.getElementById('resumeWebcamBtn').style.display = 'none';
        document.getElementById('startTrackingBtn').style.display = 'inline-block';
        document.getElementById('stopTrackingBtn').style.display = 'none';
        document.getElementById('webcamStatus').textContent = '';
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        currentImage = null;
        
        showStatus('Webcam stopped', 'success');
        
    } catch (error) {
        console.error('Stop webcam error:', error);
        showStatus('Error stopping webcam', 'error');
    }
}

function startFrameCapture() {
    if (!webcamActive || !webcamVideo) return;
    
    // Start adaptive frame capture
    captureNextFrame();
}

async function captureNextFrame() {
    if (!webcamActive || !webcamVideo || webcamPaused || processingFrame) {
        // Schedule next frame if still active
        if (webcamActive && !webcamPaused) {
            setTimeout(captureNextFrame, adaptiveFrameRate);
        }
        return;
    }
    
    try {
        processingFrame = true;
        const startTime = performance.now();
        
        // Create canvas to capture frame
        const tempCanvas = document.createElement('canvas');
        const tempCtx = tempCanvas.getContext('2d');
        
        tempCanvas.width = webcamVideo.videoWidth;
        tempCanvas.height = webcamVideo.videoHeight;
        
        // Draw video frame to canvas
        tempCtx.drawImage(webcamVideo, 0, 0);
        
        // Check if canvas has content
        const imageData = tempCtx.getImageData(0, 0, tempCanvas.width, tempCanvas.height);
        const pixelSum = imageData.data.reduce((sum, val) => sum + val, 0);
        
        // Get frame as base64
        const frameData = tempCanvas.toDataURL('image/jpeg', 0.8);
        
        // Only send if we have actual image data
        if (pixelSum === 0) {
            console.warn('Canvas is empty, skipping frame');
            return;
        }
        
        // Send frame to server for processing
        const response = await fetch('/api/process_webcam_frame', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                frame: frameData
            })
        });
        
        const data = await response.json();
        
        if (data.success && data.image) {
            currentImage = data.image;
            
            // Display processed frame (displayImage handles scaling automatically)
            displayImage(data.image, data.width, data.height);
            
            // Update tracking status
            liveTracking = data.live_tracking;
            updateTrackingUI();
        }
        
        // Calculate processing time and adjust frame rate
        const endTime = performance.now();
        frameProcessingTime = endTime - startTime;
        
        // Adaptive frame rate: ensure we don't queue faster than we can process
        // Add 50ms buffer to prevent overwhelming the server
        adaptiveFrameRate = Math.max(frameProcessingTime + 50, 100); // Min 10 FPS, max depends on processing speed
        
        // Update status with FPS info occasionally
        if (Math.random() < 0.05) { // 5% of frames
            const currentFPS = (1000/adaptiveFrameRate).toFixed(1);
            console.log(`Processing time: ${frameProcessingTime.toFixed(1)}ms, FPS: ${currentFPS}`);
            
            // Update webcam status with FPS
            const statusElement = document.getElementById('webcamStatus');
            if (statusElement && webcamActive) {
                const baseStatus = liveTracking ? '📹 Webcam Active - 🎯 Live Tracking' : '📹 Webcam Active - Ready for Annotations';
                statusElement.textContent = `${baseStatus} (${currentFPS} FPS)`;
            }
        }
        
    } catch (error) {
        console.error('Frame capture error:', error);
        // On error, slow down the frame rate
        adaptiveFrameRate = Math.min(adaptiveFrameRate * 1.5, 1000); // Max 1 FPS on errors
    } finally {
        processingFrame = false;
        
        // Schedule next frame capture
        if (webcamActive && !webcamPaused) {
            setTimeout(captureNextFrame, adaptiveFrameRate);
        }
    }
}

function pauseWebcamFeed() {
    webcamPaused = true;
    
    // Update UI
    document.getElementById('pauseWebcamBtn').style.display = 'none';
    document.getElementById('resumeWebcamBtn').style.display = 'inline-block';
    document.getElementById('webcamStatus').textContent = '📹 Webcam Paused - Ready for Annotations';
    
    showStatus('Webcam feed paused - frame frozen for annotation', 'success');
}

function resumeWebcamFeed() {
    webcamPaused = false;
    
    // Update UI
    document.getElementById('pauseWebcamBtn').style.display = 'inline-block';
    document.getElementById('resumeWebcamBtn').style.display = 'none';
    
    // Update status based on tracking state
    if (liveTracking) {
        document.getElementById('webcamStatus').textContent = '📹 Webcam Active - 🎯 Live Tracking';
    } else {
        document.getElementById('webcamStatus').textContent = '📹 Webcam Active - Ready for Annotations';
    }
    
    // Restart adaptive frame capture
    if (webcamActive && !processingFrame) {
        captureNextFrame();
    }
    
    showStatus('Webcam feed resumed', 'success');
}

async function startLiveTracking() {
    try {
        showLoading(true);
        
        const response = await fetch('/api/start_webcam_tracking', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            liveTracking = data.live_tracking;
            updateTrackingUI();
            showStatus('Live tracking started', 'success');
        } else {
            showStatus(data.error || 'Failed to start tracking', 'error');
        }
        
    } catch (error) {
        console.error('Start tracking error:', error);
        showStatus('Failed to start tracking', 'error');
    } finally {
        showLoading(false);
    }
}

async function stopLiveTracking() {
    try {
        showLoading(true);
        
        const response = await fetch('/api/stop_webcam_tracking', {
            method: 'POST'
        });
        
        const data = await response.json();
        
        if (data.success) {
            liveTracking = data.live_tracking;
            updateTrackingUI();
            showStatus('Live tracking stopped - webcam feed continues', 'success');
        } else {
            showStatus(data.error || 'Failed to stop tracking', 'error');
        }
        
    } catch (error) {
        console.error('Stop tracking error:', error);
        showStatus('Failed to stop tracking', 'error');
    } finally {
        showLoading(false);
    }
}

function updateTrackingUI() {
    const startBtn = document.getElementById('startTrackingBtn');
    const stopBtn = document.getElementById('stopTrackingBtn');
    
    if (liveTracking) {
        startBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
        if (!webcamPaused) {
            document.getElementById('webcamStatus').textContent = '📹 Webcam Active - 🎯 Live Tracking';
        } else {
            document.getElementById('webcamStatus').textContent = '📹 Webcam Paused - 🎯 Tracking Ready';
        }
    } else {
        startBtn.style.display = 'inline-block';
        stopBtn.style.display = 'none';
        if (webcamActive) {
            if (webcamPaused) {
                document.getElementById('webcamStatus').textContent = '📹 Webcam Paused - Ready for Annotations';
            } else {
                document.getElementById('webcamStatus').textContent = '📹 Webcam Active - Ready for Annotations';
            }
        } else {
            document.getElementById('webcamStatus').textContent = '';
        }
    }
}