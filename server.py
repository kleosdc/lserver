from fastapi import FastAPI, File, UploadFile
from faster_whisper import WhisperModel
from googletrans import Translator
import uvicorn
import tempfile

app = FastAPI()
translator = Translator()

# Load Whisper model on GPU
model = WhisperModel("large-v3", device="cuda", compute_type="float32")

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    segments, _ = model.transcribe(tmp_path, beam_size=5)

    full_text = ""
    for segment in segments:
        full_text += segment.text.strip() + " "

    try:
        translated = translator.translate(full_text, dest="en").text
    except:
        translated = "[Translation error]"

    return {
        "original": full_text.strip(),
        "translated": translated.strip()
    }

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
