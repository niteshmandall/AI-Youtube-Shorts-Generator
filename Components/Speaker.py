import cv2
import numpy as np
import webrtcvad
import wave
import contextlib
from pydub import AudioSegment
import os
os.environ["QT_QPA_PLATFORM"] = "xcb"



# Update paths to the model files
prototxt_path = "models/deploy.prototxt"
model_path = "models/res10_300x300_ssd_iter_140000_fp16.caffemodel"
temp_audio_path = "temp_audio.wav"

# Load DNN model
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)

# Initialize VAD
vad = webrtcvad.Vad(2)  # Aggressiveness mode from 0 to 3

def voice_activity_detection(audio_frame, sample_rate=16000):
    return vad.is_speech(audio_frame, sample_rate)

def extract_audio_from_video(video_path, audio_path):
    audio = AudioSegment.from_file(video_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
    audio.export(audio_path, format="wav")

def process_audio_frame(audio_data, sample_rate=16000, frame_duration_ms=30):
    n = int(sample_rate * frame_duration_ms / 1000) * 2  # 2 bytes per sample
    offset = 0
    while offset + n <= len(audio_data):
        frame = audio_data[offset:offset + n]
        offset += n
        yield frame

global Frames
Frames = [] # [x,y,w,h]

def detect_faces_and_speakers(input_video_path, output_video_path):
    global Frames
    Frames = []  # Reset frames for each video
    
    print("🔍 Detecting faces and speakers...")
    
    # Extract audio from the video
    extract_audio_from_video(input_video_path, temp_audio_path)

    # Read the extracted audio
    try:
        with contextlib.closing(wave.open(temp_audio_path, 'rb')) as wf:
            sample_rate = wf.getframerate()
            audio_data = wf.readframes(wf.getnframes())
    except Exception as e:
        print(f"❌ Error reading audio: {e}")
        return

    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print("❌ Error: Could not open video file")
        return
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    frame_duration_ms = 30  # 30ms frames
    audio_generator = process_audio_frame(audio_data, sample_rate, frame_duration_ms)
    
    # Initialize default face position (center of frame)
    last_face = [
        frame_width // 4,  # x
        frame_height // 4,  # y
        3 * frame_width // 4,  # x1
        3 * frame_height // 4  # y1
    ]
    face_found = False

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        if frame_count % 30 == 0:  # Log progress every 30 frames
            print(f"\r🔄 Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames)*100:.1f}%)", end="")

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        net.setInput(blob)
        detections = net.forward()

        # Get audio frame for VAD
        audio_frame = next(audio_generator, None)
        if audio_frame is None:
            break
            
        is_speaking_audio = voice_activity_detection(audio_frame, sample_rate)
        
        # Initialize variables for this frame
        current_faces = []
        max_lip_distance = 0
        active_speaker = None
        
        # First pass: detect all faces and calculate lip movements
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.3:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x, y, x1, y1) = box.astype("int")
                
                # Ensure coordinates are within frame bounds
                x, y = max(0, x), max(0, y)
                x1, y1 = min(w-1, x1), min(h-1, y1)
                
                face_width = x1 - x
                face_height = y1 - y
                
                if face_width < 20 or face_height < 20:  # Skip very small detections
                    continue
                    
                # Calculate lip movement (simplified)
                lip_distance = abs((y + 2 * face_height // 3) - y1)
                current_faces.append({
                    'box': (x, y, x1, y1),
                    'lip_distance': lip_distance,
                    'size': face_width * face_height
                })
                
                max_lip_distance = max(max_lip_distance, lip_distance)
        
        # Second pass: determine active speaker
        for face in current_faces:
            x, y, x1, y1 = face['box']
            lip_ratio = face['lip_distance'] / max_lip_distance if max_lip_distance > 0 else 0
            
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x1, y1), (0, 255, 0), 2)
            
            # Simple heuristic: if lips are moving and audio is detected, this is likely the speaker
            if lip_ratio > 0.7 and is_speaking_audio:
                cv2.putText(frame, "Active Speaker", (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                active_speaker = face['box']
                face_found = True
                break
        
        # If we found an active speaker, use that face. Otherwise, use the last known face or center
        if active_speaker is not None:
            last_face = list(active_speaker)
        elif not face_found and frame_count == 1:  # If first frame and no face found, use center
            last_face = [
                frame_width // 4,
                frame_height // 4,
                3 * frame_width // 4,
                3 * frame_height // 4
            ]
        
        # Store the face coordinates for this frame
        Frames.append(last_face.copy())
        
        # Write the frame to output
        out.write(frame)
        
        # Show preview (commented out for headless operation)
        # cv2.imshow('Frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

    print("\n✅ Face detection complete")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Clean up temporary audio file
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)



if __name__ == "__main__":
    detect_faces_and_speakers()
    print(Frames)
    print(len(Frames))
    print(Frames[1:5])
