import os
import pypdf
import docx
import pytesseract
from PIL import Image
from faster_whisper import WhisperModel

# Load lightweight Whisper ASR model for local CPU audio processing
whisper_model = WhisperModel("tiny", device="cpu", compute_type="float32")

def process_file(file_path: str) -> str:
    """Detects file type and extracts all text so the LLM can analyze it."""
    if not os.path.exists(file_path):
        return f"Error: File not found at {file_path}"

    ext = file_path.split('.')[-1].lower()

    # 1. Text & Code Files
    if ext in ['txt', 'md', 'py', 'json', 'csv', 'html', 'js']:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

    # 2. PDF Documents
    elif ext == 'pdf':
        reader = pypdf.PdfReader(file_path)
        text = "\n".join([page.extract_text() or "" for page in reader.pages])
        return text if text.strip() else "PDF is empty or scanned (requires OCR)."

    # 3. Word Documents (.docx)
    elif ext == 'docx':
        doc = docx.Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])

    # 4. Images (OCR via Tesseract)
    elif ext in ['png', 'jpg', 'jpeg', 'webp', 'bmp']:
        try:
            image = Image.open(file_path)
            extracted_text = pytesseract.image_to_string(image)
            return extracted_text if extracted_text.strip() else "No text could be extracted from image."
        except Exception as e:
            return f"OCR processing failed: {e}"

    # 5. Audio Files (Transcribe via Faster-Whisper)
    elif ext in ['mp3', 'wav', 'm4a', 'ogg', 'flac']:
        segments, _ = whisper_model.transcribe(file_path, beam_size=2)
        transcription = " ".join([segment.text for segment in segments])
        return transcription

    else:
        return f"Unsupported file extension: .{ext}"
