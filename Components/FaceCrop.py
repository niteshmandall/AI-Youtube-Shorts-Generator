import cv2
import numpy as np
from moviepy.editor import *
from Components.Speaker import detect_faces_and_speakers, Frames
global Fps

def crop_to_vertical(input_video_path, output_video_path):
    print("🔍 Detecting faces and speakers...")
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(input_video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        print("❌ Error: Could not open video.")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    vertical_height = int(original_height)
    vertical_width = int(vertical_height * 9 / 16)
    print(f"📏 Original: {original_width}x{original_height}, Target: {vertical_width}x{vertical_height}")

    if original_width < vertical_width:
        print("❌ Error: Original video width is less than the desired vertical width.")
        return

    # Initialize crop window
    x_start = (original_width - vertical_width) // 2
    x_end = x_start + vertical_width
    half_width = vertical_width // 2

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (vertical_width, vertical_height))
    global Fps
    Fps = fps

    print(f"🎞️  Processing {total_frames} frames...")
    
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("\n✅ Reached end of video")
            break
            
        # Show progress
        if count % 30 == 0:  # Update progress every 30 frames
            print(f"\r🔄 Processing frame {count}/{total_frames} ({(count/total_frames)*100:.1f}%)", end="")
        
        # Process frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            try:
                # Get face position from speaker detection
                if count < len(Frames):
                    (X, Y, W, H) = Frames[count] if isinstance(Frames[count], (list, tuple)) else Frames[count][0]
                    
                    # Find the face that matches our speaker
                    for (x1, y1, w1, h1) in faces:
                        center = x1 + w1 // 2
                        if X <= center <= X + W and Y <= y1 <= Y + H:
                            # Calculate new crop window
                            centerX = x1 + w1 // 2
                            new_x_start = max(0, min(centerX - half_width, original_width - vertical_width))
                            new_x_end = new_x_start + vertical_width
                            
                            # Smooth transition between frames
                            if count > 0 and abs(new_x_start - x_start) > 10:  # Only move if significant change
                                x_start = int(x_start * 0.7 + new_x_start * 0.3)
                                x_end = x_start + vertical_width
                            else:
                                x_start, x_end = new_x_start, new_x_end
                            break
            except Exception as e:
                print(f"\n⚠️  Error processing frame {count}: {e}")
                # Fall back to center crop if face detection fails
                x_start = (original_width - vertical_width) // 2
                x_end = x_start + vertical_width
        
        # Ensure crop window is within bounds
        x_start = max(0, min(x_start, original_width - vertical_width))
        x_end = x_start + vertical_width
        
        # Crop and write frame
        try:
            cropped_frame = frame[:, x_start:x_end]
            if cropped_frame.size > 0:
                out.write(cropped_frame)
            else:
                print(f"\n⚠️  Empty frame at {count}, using center crop")
                center_crop = frame[:, (original_width - vertical_width) // 2:(original_width + vertical_width) // 2]
                out.write(center_crop)
        except Exception as e:
            print(f"\n⚠️  Error writing frame {count}: {e}")
        
        count += 1

    # Cleanup
    cap.release()
    out.release()
    print(f"\n✅ Cropping complete. Saved to {output_video_path}")



def combine_videos(video_with_audio, video_without_audio, output_filename):
    try:
        # Load video clips
        clip_with_audio = VideoFileClip(video_with_audio)
        clip_without_audio = VideoFileClip(video_without_audio)

        audio = clip_with_audio.audio

        combined_clip = clip_without_audio.set_audio(audio)

        global Fps
        combined_clip.write_videofile(output_filename, codec='libx264', audio_codec='aac', fps=Fps, preset='medium', bitrate='3000k')
        print(f"Combined video saved successfully as {output_filename}")
    
    except Exception as e:
        print(f"Error combining video and audio: {str(e)}")



if __name__ == "__main__":
    input_video_path = r'Out.mp4'
    output_video_path = 'Croped_output_video.mp4'
    final_video_path = 'final_video_with_audio.mp4'
    detect_faces_and_speakers(input_video_path, "DecOut.mp4")
    crop_to_vertical(input_video_path, output_video_path)
    combine_videos(input_video_path, output_video_path, final_video_path)



