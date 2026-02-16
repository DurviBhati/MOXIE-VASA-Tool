import os

def standardize_mp4(video_path):
    print(f"[PreProcess] Detected .mp4 file. Applying MP4 standardizations to {os.path.basename(video_path)}...")
    ## Writing the proprocessing steps such as resizing, change frm rate and more
    return video_path

def standardize_avi(video_path):
    print(f"[PreProcess] Detected .avi file. Applying AVI standardizations to {os.path.basename(video_path)}...")
    ## Writing the preprocessing steps
    return video_path

def route_and_preprocess(video_path):
    """
    Routes the video to the correct preprocessing function based upon the file type/extension.
    """
    _, ext = os.path.splitext(video_path)
    ext = ext.lower()

    if ext == '.mp4':
        return standardize_mp4(video_path)
    elif ext == '.avi':
        return standardize_avi(video_path)
    else:
        raise ValueError(f"Unsupported file format: {ext}")
        