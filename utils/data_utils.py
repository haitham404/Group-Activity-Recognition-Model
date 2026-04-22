from collections import defaultdict
import os
import cv2


# =========================
# Dataset splits definition
# =========================
# Each number represents a video ID
splits_for_firstTry = {
    "train": [1],
    "val": [0],
    "test": [4],
}

splits = {
    "train": [1, 3, 6, 7, 10, 13, 15, 16, 18, 22, 23, 31, 32, 36, 38, 39, 40, 41, 42, 48, 50, 52, 53, 54],
    "val": [0, 2, 8, 12, 17, 19, 24, 26, 27, 28, 30, 33, 46, 49, 51],
    "test": [4, 5, 9, 11, 14, 20, 21, 25, 29, 34, 35, 37, 43, 44, 45, 47],
}


# =========================
# Bounding box structure
# =========================
class Box:
    """
    Represents a single player's bounding box in a frame.
    """

    def __init__(self, x: int, y: int, w: int, h: int, frame_id: str, action: str):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.frame_id = frame_id
        self.action = action


# =========================
# Clip-level annotations
# =========================
class ClipAnnotation:
    """
    Stores annotations for a single clip.
    Each clip contains multiple frames with player bounding boxes.
    """

    def __init__(self, clip_id: str, frame_annotations: dict[str, list[Box]]):
        self.clip_id = clip_id
        self.frame_annotations = frame_annotations


# =========================
# Video-level annotations
# =========================
class VideoAnnotation:
    """
    Stores full video annotations:
    - clip activities
    - frame-level bounding boxes
    """

    def __init__(
        self,
        video_id: str,
        clip_activities: dict[str, str],
        clip_annotations: dict[str, ClipAnnotation],
    ):
        self.video_id = video_id
        self.clip_activities = clip_activities
        self.clip_annotations = clip_annotations


# =========================
# Load clip-level boxes
# =========================
def _load_clip_annotation(clip_id: str, tracking_annotations_path: str) -> ClipAnnotation:
    """
    Loads bounding box annotations for a single clip.
    """

    player_boxes = defaultdict(list)

    # Read tracking file
    with open(tracking_annotations_path) as f:
        for line in f:
            data = line.split()

            # Parse values from annotation file
            player_id, x, y, w, h = map(int, data[:5])
            frame_id = data[5]
            action = data[-1]

            player_boxes[player_id].append(Box(x, y, w, h, frame_id, action))

    # Group boxes by frame
    frame_annotations = defaultdict(list)

    for boxes in player_boxes.values():
        # Take a subset of frames (middle frames)
        for box in boxes[6:14]:
            frame_annotations[box.frame_id].append(box)

    return ClipAnnotation(clip_id, frame_annotations)


# =========================
# Load clip activities
# =========================
def _load_clip_activities(annotation_path: str):
    """
    Loads clip-level activity labels.
    """

    clip_activities = {}

    try:
        with open(annotation_path) as f:
            for line in f:
                data = line.split()

                # Clip ID without extension
                clip_id = data[0].split(".")[0]

                # Activity label
                clip_activities[clip_id] = data[1]

    except FileNotFoundError:
        print(f"Warning: {annotation_path} not found.")

    return clip_activities


# =========================
# Load full dataset annotations
# =========================
def load_annotations(videos_dir: str, annotations_dir: str) -> dict[str, VideoAnnotation]:
    """
    Loads all videos, clips, and annotations into memory.
    """

    annotations = {}

    # Iterate over all videos
    for video_id in sorted(os.listdir(videos_dir)):

        video_dir = os.path.join(videos_dir, video_id)

        # Skip non-directory files
        if not os.path.isdir(video_dir):
            continue

        # Load clip activities
        clip_activities = _load_clip_activities(
            os.path.join(video_dir, "annotations.txt")
        )

        clip_annotations = {}

        # Iterate over clips
        for clip_id in os.listdir(video_dir):

            clip_dir = os.path.join(annotations_dir, video_id, clip_id)

            if not os.path.isdir(clip_dir):
                continue

            annotation_path = os.path.join(clip_dir, f"{clip_id}.txt")

            # Load clip annotation
            if os.path.exists(annotation_path):
                clip_annotation = _load_clip_annotation(clip_id, annotation_path)
                clip_annotations[clip_id] = clip_annotation

        # Store full video annotation
        annotations[video_id] = VideoAnnotation(
            video_id, clip_activities, clip_annotations
        )

    return annotations


# =========================
# Visualization function
# =========================
def visualize_clip(
    videos_dir: str,
    video_id: int,
    clip_id: int,
    annotations: dict[str, VideoAnnotation],
):
    """
    Visualizes bounding boxes for a given clip.
    """

    annotation = annotations[str(video_id)]

    for frame_id, boxes in annotation.clip_annotations[str(clip_id)].frame_annotations.items():

        # Load image frame
        image_path = os.path.join(videos_dir, str(video_id), str(clip_id), f"{frame_id}.jpg")
        image = cv2.imread(image_path)

        for box in boxes:

            # Draw bounding box (correct format: x1,y1 → x2,y2)
            cv2.rectangle(
                image,
                (box.x, box.y),
                (box.x + box.w, box.y + box.h),
                (0, 255, 0),
                2,
            )

            # Draw action label
            cv2.putText(
                image,
                box.action,
                (box.x, box.y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Show frame
        cv2.imshow(f"Video {video_id} Clip {clip_id}", image)
        cv2.waitKey(0)

    cv2.destroyAllWindows()