import os
from edit_video import process_video

if __name__ == "__main__":
    video_path = os.getenv('VIDEO_PATH')
    process_video(video_path)