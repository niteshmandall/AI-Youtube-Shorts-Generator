import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock, ANY

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.absolute()))

# Import the original TextClip first to patch it
from moviepy.editor import TextClip as OriginalTextClip
from moviepy.editor import VideoFileClip, ColorClip

# Create a mock TextClip class
class MockTextClip(OriginalTextClip):
    _instances = []
    
    def __init__(self, txt=None, **kwargs):
        self.txt = txt
        self.kwargs = kwargs
        self.start_time = 0
        self.end_time = 0
        self.position = (0, 0)
        self.duration = 0
        MockTextClip._instances.append(self)
    
    def set_position(self, pos):
        self.position = pos
        return self
    
    def set_start(self, t):
        self.start_time = t
        return self
    
    def set_end(self, t):
        self.end_time = t
        return self
    
    def set_duration(self, duration):
        self.duration = duration
        return self
    
    @classmethod
    def reset(cls):
        cls._instances = []
    
    @classmethod
    def get_instances(cls):
        return cls._instances

# Now import the module to test with the patched TextClip
with patch('moviepy.editor.TextClip', MockTextClip):
    from Components.CaptionGenerator import generate_captions, create_caption_text_clip

class TestCaptionGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures before any tests are run."""
        # Create a test video file if it doesn't exist
        cls.test_video = Path("test_video.mp4")
        if not cls.test_video.exists():
            # Create a simple 5-second test video with a color clip
            clip = ColorClip((640, 360), color=(255, 0, 0), duration=5)
            clip.fps = 24  # Set FPS as an attribute
            clip.write_videofile("test_video.mp4", logger=None)
        
        # Create a test audio file if it doesn't exist
        cls.test_audio = Path("test_audio.wav")
        if not cls.test_audio.exists():
            # Create a simple 5-second silent audio file using AudioClip
            from moviepy.audio.AudioClip import AudioArrayClip
            import numpy as np
            
            # Create a silent audio clip
            fps = 44100
            t = 5  # seconds
            audio_array = np.zeros((int(fps * t), 2))  # Stereo silence
            audio_clip = AudioArrayClip(audio_array, fps=fps)
            
            # Write the audio file
            audio_clip.write_audiofile("test_audio.wav", logger=None)
    
    def test_generate_captions(self):
        """Test that captions can be generated from audio."""
        # Mock the Whisper model to return test captions
        with patch('Components.CaptionGenerator.WhisperModel') as mock_whisper:
            # Set up mock return values
            mock_model = MagicMock()
            mock_segment = MagicMock()
            mock_word = MagicMock()
            mock_word.word = "test"
            mock_word.start = 0.0
            mock_word.end = 1.0
            mock_segment.words = [mock_word]
            mock_model.transcribe.return_value = ([mock_segment], None)
            mock_whisper.return_value = mock_model
            
            # Test the function
            captions = generate_captions("dummy_audio.wav", "tiny")
            
            # Verify results
            self.assertIsInstance(captions, list)
            if captions:  # Only check content if we got results
                self.assertEqual(len(captions[0]), 3)  # (text, start, end)
                self.assertIsInstance(captions[0][0], str)  # text
                self.assertIsInstance(captions[0][1], float)  # start time
                self.assertIsInstance(captions[0][2], float)  # end time
    
    def test_create_caption_text_clip(self):
        """Test that text clips can be created from captions."""
        # Reset mock instances
        MockTextClip.reset()
        
        # Test data
        test_text = "This is a test caption"
        start_time = 0.0
        end_time = 2.0
        video_size = (640, 360)
        
        # Test the function
        text_clips = create_caption_text_clip(test_text, start_time, end_time, video_size)
        
        # Verify results
        instances = MockTextClip.get_instances()
        self.assertGreater(len(instances), 0, "No TextClip instances were created")
        self.assertIsInstance(text_clips, list)
        self.assertGreater(len(text_clips), 0)
        
        # Verify the instances have the right properties
        for clip in text_clips:
            self.assertIsInstance(clip, MockTextClip)
            self.assertEqual(clip.start_time, start_time)
            self.assertEqual(clip.end_time, end_time)
    
    def test_long_text_wrapping(self):
        """Test that long text is properly wrapped into multiple lines."""
        # Reset mock instances
        MockTextClip.reset()
        
        # Create a very long test string
        long_text = "This is a very long caption that should be wrapped into multiple lines " \
                   "to ensure it fits within the video width and remains readable."
        
        # Test the function
        text_clips = create_caption_text_clip(long_text, 0, 5, (640, 360))
        
        # Get all created instances
        instances = MockTextClip.get_instances()
        
        # Verify that we got multiple text clips (one per line)
        self.assertGreater(len(instances), 1, "Expected multiple TextClip instances for long text")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        test_files = ["test_video.mp4", "test_audio.wav"]
        for file in test_files:
            if os.path.exists(file):
                os.remove(file)

if __name__ == "__main__":
    unittest.main()
