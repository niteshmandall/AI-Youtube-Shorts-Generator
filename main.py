from tqdm import tqdm
import time
from Components.YoutubeDownloader import download_youtube_video
from Components.Edit import extractAudio, crop_video
from Components.Transcription import transcribeAudio
from Components.LanguageTasks import GetHighlight
from Components.FaceCrop import crop_to_vertical, combine_videos
from Components.improved_captioning import process_video_with_captions

import os
from datetime import datetime

def create_job_directory():
    """Create a timestamped directory for each job"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_dir = f"jobs/job_{timestamp}"
    os.makedirs(job_dir, exist_ok=True)
    os.makedirs(f"{job_dir}/assets", exist_ok=True)
    os.makedirs(f"{job_dir}/output", exist_ok=True)
    return job_dir

def main():
    # Create job directory
    job_dir = create_job_directory()
    print(f"📁 Job directory created: {job_dir}")
    
    url = input("Enter YouTube video URL: ")
    
    print("\n🚀 Starting video processing...")
    
    # Download video
    with tqdm(desc="📥 Downloading video", unit="MB") as pbar:
        def update_dl_progress(stream, chunk, bytes_remaining):
            total_size = stream.filesize
            bytes_downloaded = total_size - bytes_remaining
            pbar.total = total_size / (1024 * 1024)  # Convert to MB
            pbar.update((bytes_downloaded / (1024 * 1024)) - pbar.n)
            
        Vid = download_youtube_video(url, on_progress=update_dl_progress)
    
    if Vid:
        Vid = Vid.replace(".webm", ".mp4")
        print(f"\n✅ Downloaded video: {Vid}")
        
        # Extract audio
        with tqdm(desc="🔊 Extracting audio", leave=False) as pbar:
            Audio = extractAudio(Vid)
            pbar.update(1)
            
        if Audio:
            print(f"✅ Audio extracted: {Audio}")
            
            # Transcribe audio
            with tqdm(desc="🎤 Transcribing audio", leave=False) as pbar:
                transcriptions = transcribeAudio(Audio)
                pbar.update(1)
                
            if len(transcriptions) > 0:
                print(f"✅ Transcription complete ({len(transcriptions)} segments)")
                
                # Prepare transcription text
                TransText = ""
                for text, start, end in transcriptions:
                    TransText += (f"{start} - {end}: {text}")
                
                # Get highlight
                with tqdm(desc="🔍 Finding highlights", leave=False) as pbar:
                    start, stop = GetHighlight(TransText)
                    pbar.update(1)
                
                if start != 0 and stop != 0:
                    print(f"🎯 Highlight found: {start}s to {stop}s")
                    
                    # Process video with job-specific paths
                    output_base = os.path.basename(Vid).split('.')[0]
                    Output = os.path.join(job_dir, "assets", f"{output_base}_cropped.mp4")
                    croped = os.path.join(job_dir, "assets", f"{output_base}_vertical.mp4")
                    final_output = os.path.join(job_dir, "output", f"{output_base}_final_nocaptions.mp4")
                    captioned_output = os.path.join(job_dir, "output", f"{output_base}_final_with_captions.mp4")
                    
                    with tqdm(desc="✂️  Cropping video", leave=False) as pbar:
                        crop_video(Vid, Output, start, stop)
                        pbar.update(1)
                    
                    with tqdm(desc="🔄 Converting to vertical", leave=False) as pbar:
                        crop_to_vertical(Output, croped)
                        pbar.update(1)
                    
                    with tqdm(desc="🎬 Finalizing video", leave=False) as pbar:
                        combine_videos(Output, croped, final_output)
                        pbar.update(1)
                    
                    # Add captions to the final video
                    with tqdm(desc="📝 Adding captions", leave=False) as pbar:
                        process_video_with_captions(final_output, Audio, captioned_output)
                        pbar.update(1)
                    
                    print(f"\n✅ Processing complete!")
                    print(f"- Final video with captions: {captioned_output}")
                    print(f"- Job directory: {job_dir}")
                    
                    print("\n✨ Processing complete!")
                    print(f"- Video without captions: {final_output}")
                    print(f"- Video with captions: {captioned_output}")
                else:
                    print("❌ Error: Could not determine highlight segment")
            else:
                print("❌ No transcriptions found in the audio")
        else:
            print("❌ Failed to extract audio from video")
    else:
        print("❌ Failed to download the video")

if __name__ == "__main__":
    main()