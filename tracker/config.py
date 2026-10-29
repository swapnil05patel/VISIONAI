"""
config.py
---------
Central configuration for all tunable tracking parameters.
Modify values here to adjust behaviour without touching algorithm code.
"""

import torch

# ---- Device ----
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---- Detection thresholds ----
HIGH_CONF_THR = 0.45      # Detections below this are low-confidence detections
LOW_CONF_THR  = 0.12      # YOLO confidence floor for initial prediction

# ---- ReID Backbone & Embedding Options ----
# REID_BACKBONE: "osnet_x0_25", "osnet_x1_0", or "resnet50"
REID_BACKBONE       = "osnet_x0_25"
USE_PART_EMBEDDING  = False    # True = 3-part strips; False = full-body embedding
NUM_PARTS           = 3        # Number of horizontal strips per person crop (when USE_PART_EMBEDDING=True)
EMBED_CACHE_SIZE    = 10       # Sliding window — max embeddings stored per active track

# ---- Matching cost weights ----
MATCH_WEIGHT_IOU       = 0.3
MATCH_WEIGHT_REID      = 0.5
MATCH_WEIGHT_DIRECTION = 0.2    # Weight for motion direction consistency
DIRECTION_SPEED_THRESH = 3.0    # Minimum speed (px/frame) to start applying direction cost

# ---- Active track matching thresholds ----
IOU_THR                   = 0.30   # Minimum IoU to accept a match directly
REID_STRICT_THRESH        = 0.40   # Max cosine distance for ReID-based re-attachment (calibrated for OSNet)
REID_REATTACH_MAX_AGE     = 30     # Track must have been lost within this many frames
REID_CENTER_MAX_DIST_BASE = 160    # Base pixel distance for search radius
VELOCITY_GATE_MULTIPLIER  = 1.5    # How much predicted distance expands the search radius
REID_AREA_RATIO_MAX       = 2.5    # Max area ratio between track box and detection box

# ---- ByteTrack Second-Stage Low-Confidence Matching ----
ENABLE_BYTETRACK_STAGE2   = True   # Match remaining tracks against low-conf detections using IoU
STAGE2_IOU_THR            = 0.40   # Higher IoU threshold required for low-conf detections

# ---- Fallback IoU attach ----
# Prevents premature new ID creation when overlap is moderate
FALLBACK_IOU_FOR_NEW          = 0.18
FALLBACK_CENTER_MAX_DIST_BASE = 160
MAX_FALLBACK_MISSES           = 8    # Only consider tracks missed <= this many frames

# ---- Track lifecycle ----
MIN_HITS_TO_CONFIRM = 2     # Detections needed before a track is marked confirmed
MAX_AGE             = 150   # Frames without update before retirement

# ---- Adaptive exponential smoothing ----
SMOOTH_ALPHA_SLOW = 0.70    # High smoothing for slow-moving tracks (stable)
SMOOTH_ALPHA_FAST = 0.18    # Low smoothing for fast-moving tracks (responsive)

# ---- Kalman filter noise ----
KALMAN_PROCESS_NOISE = 1e-2
KALMAN_MEAS_NOISE    = 1e-1

# ---- Exit / frame boundary detection ----
EXIT_MARGIN         = 40    # Pixels outside frame boundary before counting as exit
EXIT_MIN_MISSES     = 3     # Consecutive outside-boundary predictions to retire track
HARD_RETIRE_ON_EXIT = False # True = retire immediately on first outside-boundary prediction

# ---- Static object filtering ----
# Suppress tracks for non-moving objects (posters, signs, mannequins, etc.)
ENABLE_STATIC_PRUNING      = True  # Configurable flag to enable/disable static track pruning
STATIC_VELOCITY_THRESH     = 2.0   # Max Kalman velocity (px/frame) to consider "nearly zero"
STATIC_DISPLACEMENT_THRESH = 8.0   # Max displacement (px) over the observation window
STATIC_FRAMES_REQUIRED     = 45    # Consecutive static frames before classifying as static

# ---- Track Gallery (persistent re-identification) ----
ENABLE_GALLERY             = True   # Master switch for track gallery
GALLERY_COSINE_THRESH      = 0.35   # Max cosine distance for gallery re-attachment (calibrated for OSNet)
GALLERY_TTL_SECONDS        = 600    # Gallery entry expires after this many seconds (10 min)
GALLERY_MAX_SIZE           = 1000   # Maximum entries stored in gallery at any time
GALLERY_MIN_HITS           = 2      # Track must have at least this many hits to be saved
GALLERY_EVICT_INTERVAL     = 300    # Run TTL eviction every N frames

# Gallery spatial gating & representation
GALLERY_USE_SPATIAL_GATE   = True   # Set False for cross-camera re-identification where positions change
GALLERY_SPATIAL_MAX_DIST   = 160    # Max pixel distance between last track position and query detection
GALLERY_AREA_RATIO_MAX     = 2.5    # Max area ratio between last track box and query detection
GALLERY_REPRESENTATION     = "mean" # "mean" or "ema" or "exemplars"
GALLERY_MAX_EXEMPLARS      = 5      # Maximum exemplar embeddings stored per gallery entry
GALLERY_EMA_ALPHA          = 0.9    # EMA decay when GALLERY_REPRESENTATION="ema"
GALLERY_EXEMPLAR_MIN_DIST  = 0.15   # Min cosine distance to add a new exemplar
REVIVAL_CONFIRMATION_FRAMES = 2     # Consecutive matches required before gallery-revived track is fully confirmed
