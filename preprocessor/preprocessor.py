import mimetypes
import magic  
import cv2
import requests
import tempfile
import librosa
from utils.content_type import ContentType

class Preprocessor:
    def process(self, content_path_or_url):
        if content_path_or_url.startswith('http'):
            content_data = self.process_url(content_path_or_url)
        else:
            content_data = self.process_file(content_path_or_url)
        return content_data

    def process_file(self, file_path):
        content_type = self.determine_content_type(file_path)
        return self.load_content(file_path, content_type)

    def process_url(self, url):
        response = requests.get(url, stream=True)
        # Save content to a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            tmp_file_path = tmp_file.name

        content_type = self.determine_content_type(tmp_file_path)
        content_data = self.load_content(tmp_file_path, content_type)

        return content_data
        
    def determine_content_type(self, content_path):
        # First, try using python-magic
        try:
            mime = magic.Magic(mime=True)
            mime_type = mime.from_file(content_path)
            if mime_type:
                if mime_type.startswith('image'):
                    return ContentType.IMAGE
                elif mime_type.startswith('video'):
                    return ContentType.VIDEO
                elif mime_type.startswith('audio'):
                    return ContentType.AUDIO
                elif mime_type.startswith('text'):
                    return ContentType.TEXT
        except Exception as e:
            print(f"Magic library error: {e}")

        # Fallback to mimetypes
        mime_type, _ = mimetypes.guess_type(content_path)
        if mime_type:
            if mime_type.startswith('image'):
                return ContentType.IMAGE
            elif mime_type.startswith('video'):
                return ContentType.VIDEO
            elif mime_type.startswith('audio'):
                return ContentType.AUDIO
            elif mime_type == 'text/plain':
                return ContentType.TEXT
        
    def load_content(self, path, content_type):
        if content_type == ContentType.IMAGE:
            return self.load_image(path)
        elif content_type == ContentType.VIDEO:
            return self.load_video(path)
        elif content_type == ContentType.AUDIO:
            return self.load_audio(path)
        elif content_type == ContentType.TEXT:
            return self.load_text(path)
        else:
            print(f"Unsupported content type: {content_type}")
            return None
    
    def load_image(self, path):
        # Read raw bytes
        with open(path, 'rb') as f:
            raw_bytes = f.read()
        # Load image using OpenCV
        image = cv2.imread(path)
        return {'type': 'image', 'data': image, 'raw_bytes': raw_bytes}

    def load_video(self, path):
        # Read raw bytes
        with open(path, 'rb') as f:
            raw_bytes = f.read()
        # Load video frames
        video = cv2.VideoCapture(path)
        frames = []
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frames.append(frame)
        video.release()
        return {'type': 'video', 'data': frames, 'raw_bytes': raw_bytes}

    def load_audio(self, path):
        # Read raw bytes
        with open(path, 'rb') as f:
            raw_bytes = f.read()
        # Load audio data using librosa or similar
        audio_data, sample_rate = librosa.load(path, sr=None)
        return {
            'type': 'audio',
            'data': audio_data,
            'sample_rate': sample_rate,
            'raw_bytes': raw_bytes
        }

    def load_text(self, path):
        with open(path, 'r', encoding='utf-8') as file:
            text_data = file.read()
        return {'type': 'text', 'data': text_data}