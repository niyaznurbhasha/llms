import os
import json
import time
import torch
from transformers import pipeline
from typing import Dict, Any, Optional
from fpdf import FPDF
from pyannote.audio import Pipeline as DiarizationPipeline
from pydub import AudioSegment
import tempfile

# Default model constants
default_asr_model = "openai/whisper-large-v2"
default_summarization_model = "knkarthick/MEETING_SUMMARY"

available_models = {
    "meeting_summary": "knkarthick/MEETING_SUMMARY",
    "flan_t5_large": "google/flan-t5-large",
    "bart_large_cnn": "facebook/bart-large-cnn"
}

class LLMModelV6:
    def __init__(
        self,
        llm_model_key: str = "meeting_summary",
        device: Optional[str] = None,
        enable_speaker_separation: bool = False
    ):
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.enable_speaker_separation = enable_speaker_separation

        llm_model_path = available_models.get(llm_model_key, default_summarization_model)

        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=default_asr_model,
            device_map="auto",
            chunk_length_s=30
        )

        self.summarizer = pipeline(
            "summarization",
            model=llm_model_path,
            tokenizer=llm_model_path,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "mps" else torch.float32
        )

        if self.enable_speaker_separation:
            self.diarization_pipeline = DiarizationPipeline.from_pretrained(
                "pyannote/speaker-diarization",
                use_auth_token=True
            )

    def transcribe_audio(self, audio_file_path: str) -> str:
        start = time.time()
        audio = AudioSegment.from_file(audio_file_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
            audio.export(wav_path, format="wav")

        result = self.asr_pipeline(wav_path)
        os.remove(wav_path)

        text = "".join([r.get("text", "") for r in result]) if isinstance(result, list) else result.get("text", "")
        elapsed = time.time() - start
        print(f"[Timing] Transcription step took {elapsed:.2f} seconds.")
        return text

    def separate_speakers(self, audio_file_path: str) -> Dict[str, str]:
        start = time.time()
        audio = AudioSegment.from_file(audio_file_path)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            wav_path = tmp.name
            audio.export(wav_path, format="wav")

        diarization = self.diarization_pipeline(wav_path)
        speaker_texts: Dict[str, list] = {}
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            seg_start, seg_end = int(segment.start * 1000), int(segment.end * 1000)
            chunk = audio[seg_start:seg_end]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp2:
                chunk_path = tmp2.name
                chunk.export(chunk_path, format="wav")
            result = self.asr_pipeline(chunk_path)
            os.remove(chunk_path)
            text = "".join([r.get("text", "") for r in result]) if isinstance(result, list) else result.get("text", "")
            speaker_texts.setdefault(speaker, []).append(text)

        os.remove(wav_path)
        elapsed = time.time() - start
        print(f"[Timing] Speaker separation & transcription took {elapsed:.2f} seconds.")
        return {spk: " ".join(txts).strip() for spk, txts in speaker_texts.items()}

    def chunk_text(self, text: str, max_tokens: int = 800) -> list[str]:
        words = text.split()
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]

    def generate_summary(self, separated_transcript: Dict[str, str], detail_level: str = "standard") -> str:
        start = time.time()
        full_text = "\n".join(f"{spk}: {txt}" for spk, txt in separated_transcript.items())

        if detail_level == "detailed":
            max_length = 512
            min_length = 200
        else:
            max_length = 256
            min_length = 64

        summary_parts = []
        for chunk in self.chunk_text(full_text):
            out = self.summarizer(
                chunk,
                max_length=max_length,
                min_length=min_length,
                truncation=True
            )[0]
            summary_parts.append(out.get("summary_text", "").strip())

        summary = "\n\n".join(summary_parts)
        elapsed = time.time() - start
        print(f"[Timing] Summarization step took {elapsed:.2f} seconds.")
        return summary

    def save_to_pdf(self, content: Any, filename: str) -> None:
        start = time.time()
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)
        if isinstance(content, dict):
            for speaker, text in content.items():
                pdf.multi_cell(0, 10, f"{speaker}: {text}")
                pdf.ln(2)
        else:
            pdf.multi_cell(0, 10, content)
        pdf.output(filename)
        elapsed = time.time() - start
        print(f"[Timing] PDF save ({filename}) took {elapsed:.2f} seconds.")

    def transcribe_and_summarize(
        self,
        audio_file_path: str,
        transcript_pdf: str = "transcript.pdf",
        summary_pdf: str = "summary.pdf",
        detail_level: str = "standard"
    ) -> Dict[str, Any]:
        total_start = time.time()
        if self.enable_speaker_separation:
            separated = self.separate_speakers(audio_file_path)
        else:
            text = self.transcribe_audio(audio_file_path)
            separated = {"Speaker 1": text}

        summary = self.generate_summary(separated, detail_level)
        self.save_to_pdf(separated, transcript_pdf)
        self.save_to_pdf(summary, summary_pdf)
        total_elapsed = time.time() - total_start
        print(f"[Timing] Total pipeline took {total_elapsed:.2f} seconds.")
        return {"transcript": separated, "summary": summary}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="Transcribe & summarize audio with timing logs and model options"
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="meeting_summary",
        choices=available_models.keys(),
        help="Summarization model: meeting_summary, flan_t5_large, bart_large_cnn"
    )
    parser.add_argument("--audio", type=str, required=True, help="Path to input audio file.")
    parser.add_argument("--device", type=str, help="Compute device: 'mps' or 'cpu'.")
    parser.add_argument("--speaker_separation", action="store_true", help="Enable speaker separation.")
    parser.add_argument("--out_transcript", type=str, default="transcript.pdf", help="Output PDF for transcript.")
    parser.add_argument("--out_summary", type=str, default="summary.pdf", help="Output PDF for summary.")
    parser.add_argument("--detail_level", type=str, default="standard", choices=["standard", "detailed"], help="Summary detail level.")

    args = parser.parse_args()

    model = LLMModelV6(
        llm_model_key=args.llm_model,
        device=args.device,
        enable_speaker_separation=args.speaker_separation
    )

    results = model.transcribe_and_summarize(
        args.audio,
        args.out_transcript,
        args.out_summary,
        args.detail_level
    )

    print(json.dumps(results, indent=2))
