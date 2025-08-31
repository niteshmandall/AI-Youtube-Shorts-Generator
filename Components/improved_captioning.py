import os
import json
import subprocess
import srt
from datetime import timedelta
import numpy as np
from faster_whisper import WhisperModel
import ffmpeg
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip, ColorClip, VideoClip, AudioFileClip
from moviepy.config import change_settings
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import tempfile
import shutil
from indic_transliteration import sanscript
from indic_transliteration import sanscript as sanscript_translit

# Set imageio to use ffmpeg explicitly
change_settings({"IMAGEMAGICK_BINARY": "/usr/bin/convert"})

def transcribe_audio(audio_path, model_size="base", language=None, device="cuda", compute_type="float16"):
    """Transcribe audio using Whisper with word-level timestamps"""
    print("🔠 Loading Whisper model...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    
    print("🎤 Transcribing audio...")
    segments, info = model.transcribe(
        audio_path,
        language=language,
        word_timestamps=True,
        vad_filter=True,
        beam_size=5,
        best_of=5,
        temperature=0.0,
        condition_on_previous_text=False
    )
    
    # Language to script mapping for transliteration
    script_map = {
        'hi': sanscript.DEVANAGARI,  # Hindi
        'mr': sanscript.DEVANAGARI,  # Marathi
        'bn': sanscript.BENGALI,     # Bengali
        'ta': sanscript.TAMIL,       # Tamil
        'te': sanscript.TELUGU,      # Telugu
        'kn': sanscript.KANNADA,     # Kannada
        'gu': sanscript.GUJARATI,    # Gujarati
        'pa': sanscript.GURMUKHI,    # Punjabi (Gurmukhi)
        'or': sanscript.ORIYA,       # Odia
        'ml': sanscript.MALAYALAM,   # Malayalam
        'sa': sanscript.DEVANAGARI,  # Sanskrit
    }
    
    # Convert to list and extract word-level segments
    segments = list(segments)
    words = []
    for segment in segments:
        for word in segment.words:
            text = word.word
            # If the detected language is in our script map, transliterate to Latin
            if info.language in script_map and text.strip():
                try:
                    text = sanscript_translit.transliterate(text, script_map[info.language], sanscript.ITRANS)
                except Exception as e:
                    print(f"⚠️ Could not transliterate: {text} - {str(e)}")
            
            words.append({
                'text': text,
                'start': word.start,
                'end': word.end,
                'confidence': word.probability
            })
    
    return words, info.language

def create_srt(segments, output_path):
    """Create SRT file from word segments"""
    subs = []
    for i, segment in enumerate(segments):
        start_time = timedelta(seconds=segment['start'])
        end_time = timedelta(seconds=segment['end'])
        subs.append(srt.Subtitle(
            index=i+1,
            start=start_time,
            end=end_time,
            content=segment['text']
        ))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(srt.compose(subs))
    
    return output_path

def create_caption_text_clip(text, start, end, video_size, fontsize=48, 
                           color='white', stroke_color='black', stroke_width=2, padding=10):
    """Create a styled text clip with background"""
    width, height = video_size
    
    # Split text into lines that fit the video width
    max_width = int(width * 0.9)  # 90% of video width
    # Font is already loaded at the beginning of the function
    
    # Try to load a default font, fall back to default if not found
    try:
        # Try common system fonts
        for font_name in ['Arial', 'DejaVuSans', 'FreeSans', 'LiberationSans']:
            try:
                font = ImageFont.truetype(font_name, fontsize)
                break
            except IOError:
                continue
        else:
            # If no font found, use default
            font = ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()
    
    # Simple word wrapping
    words = text.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        test_width = font.getlength(test_line)
        
        if test_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate total height needed
    line_height = int(fontsize * 1.2)  # 1.2 for line spacing
    total_height = len(lines) * line_height + 2 * padding
    
    # Create a transparent image for the text
    img = Image.new('RGBA', (width, total_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw each line of text
    for i, line in enumerate(lines):
        # Calculate text width and position (centered)
        text_width = font.getlength(line)
        x = (width - text_width) / 2
        y = i * line_height + 10
        
        # Draw text with stroke (outline) for better visibility
        for dx in [-stroke_width, 0, stroke_width]:
            for dy in [-stroke_width, 0, stroke_width]:
                if dx != 0 or dy != 0:  # Skip the center position
                    draw.text((x + dx, y + dy), line, font=font, fill=stroke_color, stroke_width=stroke_width)
        
        # Draw main text
        draw.text((x, y), line, font=font, fill=color, stroke_width=0)
    
    # Convert to numpy array and ensure RGB format
    img_np = np.array(img)
    
    # If image has alpha channel, composite it with black background
    if img_np.shape[2] == 4:
        alpha = img_np[:, :, 3:4] / 255.0
        rgb = img_np[:, :, :3]
        img_np = (rgb * alpha + (1 - alpha) * 0).astype(np.uint8)
    
    # Create a function that returns the image for each frame
    def make_frame(t):
        return img_np
    
    # Create a video clip with the text
    text_clip = VideoClip(make_frame, duration=end - start)
    
    # Position the text at the bottom of the video
    text_clip = text_clip.set_position(('center', height - total_height - 20))
    text_clip = text_clip.set_start(start).set_end(end)
    
    return text_clip

def add_captions_to_video(video_path, audio_path, output_path):
    """Add captions to the video using the provided audio"""
    print("🎬 Adding captions to video...")
    
    # Load the video and audio
    video = VideoFileClip(video_path)
    audio = AudioFileClip(audio_path)
    video = video.set_audio(audio)
    video_duration = video.duration
    
    # Transcribe audio with word-level timestamps
    segments, _ = transcribe_audio(
        audio_path,
        model_size="base",
        language=None,
        device="cuda" if os.getenv('CUDA_VISIBLE_DEVICES') else "cpu",
        compute_type="float16" if os.getenv('CUDA_VISIBLE_DEVICES') else "int8"
    )
    
    # Create a list to hold all caption clips
    caption_clips = []
    
    # Group words into phrases based on timing
    current_phrase = []
    current_start = 0
    max_phrase_duration = 3.0  # Maximum duration for a single caption
    
    for segment in segments:
        if not current_phrase:
            current_phrase = [segment]
            current_start = segment['start']
        else:
            # If the gap is small and total duration is reasonable, add to current phrase
            if (segment['start'] - current_start < max_phrase_duration and 
                segment['end'] - current_start < 2 * max_phrase_duration):
                current_phrase.append(segment)
            else:
                # Create caption for current phrase
                text = ' '.join([s['text'] for s in current_phrase])
                start = current_phrase[0]['start']
                end = current_phrase[-1]['end']
                
                # Ensure caption duration is reasonable
                if end - start > max_phrase_duration * 1.5:
                    end = start + max_phrase_duration
                
                caption = create_caption_text_clip(
                    text, start, end, video.size,
                    fontsize=int(video.size[1] * 0.05),  # Slightly larger font
                    color='white',
                    stroke_color='black',
                    stroke_width=2
                )
                caption_clips.append(caption)
                
                # Start new phrase
                current_phrase = [segment]
                current_start = segment['start']
    
    # Add the last phrase
    if current_phrase:
        text = ' '.join([s['text'] for s in current_phrase])
        start = current_phrase[0]['start']
        end = min(current_phrase[-1]['end'], video_duration)
        
        caption = create_caption_text_clip(
            text, start, end, video.size,
            fontsize=int(video.size[1] * 0.04),
            color='white',
            stroke_color='black',
            stroke_width=2
        )
        caption_clips.append(caption)
    
    # Create final video with captions
    print("📼 Rendering final video with captions...")
    
    # Ensure all clips have the same size and format
    video = video.set_duration(video.duration)
    for i, clip in enumerate(caption_clips):
        caption_clips[i] = clip.set_duration(min(clip.duration, video.duration - clip.start))
    
    final = CompositeVideoClip([video] + caption_clips, use_bgclip=True)
    
    # Write the result to a file with optimized settings
    temp_dir = tempfile.mkdtemp()
    temp_audio = os.path.join(temp_dir, "temp_audio.aac")
    temp_video = os.path.join(temp_dir, "temp_video.mp4")
    
    try:
        # Extract audio
        audio = video.audio
        audio.write_audiofile(temp_audio, codec='aac', bitrate='192k', verbose=False, logger=None)
        
        # Set the final video's audio and write with audio
        if os.path.exists(audio_path):
            audio_clip = AudioFileClip(audio_path)
            # Ensure audio duration matches video duration
            if audio_clip.duration > final.duration:
                audio_clip = audio_clip.subclip(0, final.duration)
            final = final.set_audio(audio_clip)
        
        # Write final video with audio and captions
        final.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            fps=video.fps,
            preset='fast',
            ffmpeg_params=[
                '-crf', '23',
                '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart'
            ],
            audio_bitrate='192k',
            threads=4,
            logger=None
        )
        
    finally:
        # Clean up temporary files
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    video.close()
    final.close()
    
    print(f"✅ Captions added successfully: {output_path}")
    return output_path

def process_video_with_captions(video_path, audio_path, output_path="Final_With_Captions.mp4", model_size="base"):
    """Main function to process video and add captions with improved quality"""
    print("🚀 Starting video processing with improved captions...")
    
    # Transcribe audio with word-level timestamps
    segments, detected_language = transcribe_audio(
        audio_path,
        model_size=model_size,
        language=None,  # Auto-detect language
        device="cuda" if os.getenv('CUDA_VISIBLE_DEVICES') else "cpu",
        compute_type="float16" if os.getenv('CUDA_VISIBLE_DEVICES') else "int8"
    )
    
    print(f"✅ Transcription complete in {detected_language} (detected)")
    
    # Add captions to video
    output_path = add_captions_to_video(video_path, audio_path, output_path)
    
    # Save SRT file alongside the video
    srt_path = os.path.splitext(output_path)[0] + ".srt"
    create_srt(segments, srt_path)
    print(f"📝 SRT file saved: {srt_path}")
    
    return output_path
