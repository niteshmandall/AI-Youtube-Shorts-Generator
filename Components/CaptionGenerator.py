import os
import cv2
import numpy as np
import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip, ColorClip, concatenate_videoclips
from moviepy.video.VideoClip import VideoClip
from pydub import AudioSegment
from faster_whisper import WhisperModel
import math
from PIL import Image, ImageDraw, ImageFont
import os

def generate_captions(audio_path, model_size="base"):
    """
    Generate captions from audio using Whisper
    Returns a list of (text, start_time, end_time) tuples
    """
    print("🔤 Generating captions...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    
    # Transcribe audio with word timestamps
    segments, _ = model.transcribe(
        audio_path,
        word_timestamps=True,
        language="en",
        vad_filter=True
    )
    
    captions = []
    for segment in segments:
        for word in segment.words:
            captions.append((word.word, word.start, word.end))
    
    return captions

def create_caption_text_clip(text, start, end, video_size, fontsize=24, color='white'):
    """Create a text clip for the given text and time range using OpenCV"""
    width, height = video_size
    
    # Split text into multiple lines if needed
    max_chars = 40  # Max characters per line
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        if len(' '.join(current_line + [word])) <= max_chars:
            current_line.append(word)
        else:
            lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate total height needed for all lines
    line_height = fontsize + 10
    total_height = len(lines) * line_height
    
    # Create a transparent image for the text
    img = np.zeros((total_height, width, 4), dtype=np.uint8)
    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    
    # Load a font (using default font if Arial is not available)
    try:
        font = ImageFont.truetype("Arial.ttf", fontsize)
    except IOError:
        font = ImageFont.load_default()
    
    # Draw each line of text
    for i, line in enumerate(lines):
        # Calculate text size and position (centered)
        text_width = draw.textlength(line, font=font)
        x = (width - text_width) / 2
        y = i * line_height
        
        # Draw text with black stroke (outline)
        for dx in [-1, 1]:
            for dy in [-1, 1]:
                draw.text((x + dx, y + dy), line, font=font, fill=(0, 0, 0, 255))
        # Draw main text
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))
    
    # Convert back to numpy array and ensure it's in RGB format
    img = np.array(pil_img)
    
    # Create a function that returns the image for each frame
    def make_frame(t):
        # Convert RGBA to RGB by blending with a white background
        if img.shape[2] == 4:  # If image has alpha channel
            alpha = img[:, :, 3:] / 255.0
            rgb = img[:, :, :3]
            bg = np.ones_like(rgb) * 255  # White background
            result = (rgb * alpha + bg * (1 - alpha)).astype(np.uint8)
            return result
        return img
    
    # Create a video clip with the text
    text_clip = VideoClip(make_frame, duration=end - start)
    
    # Position the text at the bottom of the video
    text_clip = text_clip.set_position(('center', height - total_height - 20))
    text_clip = text_clip.set_start(start).set_end(end)
    
    return [text_clip]

def add_captions_to_video(video_path, output_path, captions):
    """Add captions to the video at the specified timestamps"""
    print("📝 Adding captions to video...")
    
    # Load the video
    video = VideoFileClip(video_path)
    
    # Create a list to hold all caption clips
    caption_clips = []
    
    # Group words into sentences/phrases for better readability
    current_phrase = []
    current_start = 0
    
    for i, (word, start, end) in enumerate(captions):
        if not current_phrase:
            current_start = start
            current_phrase = [word]
        else:
            # If the gap between words is small, add to current phrase
            if start - current_start < 2.0:  # 2 seconds max phrase duration
                current_phrase.append(word)
            else:
                # Add the completed phrase
                phrase = ' '.join(current_phrase)
                text_clips = create_caption_text_clip(
                    phrase, current_start, start, video.size
                )
                if isinstance(text_clips, list):
                    caption_clips.extend(text_clips)
                else:
                    caption_clips.append(text_clips)
                current_phrase = [word]
                current_start = start
    
    # Add the last phrase
    if current_phrase:
        phrase = ' '.join(current_phrase)
        text_clips = create_caption_text_clip(
            phrase, current_start, captions[-1][2], video.size
        )
        if isinstance(text_clips, list):
            caption_clips.extend(text_clips)
        else:
            caption_clips.append(text_clips)
    
    # Composite the video with captions
    final_video = CompositeVideoClip([video] + caption_clips)
    
    # Write the result to a file
    final_video.write_videofile(
        output_path,
        codec='libx264',
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        threads=4,  # Use multiple threads for faster processing
        remove_temp=True,  # Remove temporary files
        ffmpeg_params=[
            '-preset', 'fast',  # Faster encoding with minimal quality loss
            '-crf', '23'        # Constant Rate Factor (lower = better quality, 18-28 is a good range)
        ],
        fps=30
    )
    
    # Close the clips
    video.close()
    final_video.close()
    
    return output_path

def process_video_with_captions(video_path, audio_path, output_path="Final_With_Captions.mp4"):
    """Main function to process video and add captions"""
    # Generate captions
    captions = generate_captions(audio_path)
    
    # Add captions to video
    output = add_captions_to_video(video_path, output_path, captions)
    
    return output
