import os
import json
from pathlib import Path

from dotenv import load_dotenv
from sarvamai import SarvamAI

load_dotenv()

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")

client = SarvamAI(
    api_subscription_key=SARVAM_API_KEY
)


def transcribe_sarvam_batch(audio_path: str) -> str:

    print("Creating Sarvam Batch Job...")

    job = client.speech_to_text_job.create_job(
        model="saaras:v3",
        mode="transcribe",
        language_code="hi-IN",
        with_diarization=True
    )

    print(f"Job created: {job.job_id}")

    print("Uploading audio...")

    job.upload_files(
        file_paths=[audio_path]
    )

    print("Starting batch job...")

    job.start()

    print("Waiting for transcription...")

    job.wait_until_complete()

    file_results = job.get_file_results()

    if file_results["failed"]:
        raise RuntimeError(
            f"Sarvam transcription failed: "
            f"{file_results['failed']}"
        )

    print("Transcription completed.")

    output_dir = Path("./sarvam_output")
    output_dir.mkdir(exist_ok=True)

    job.download_outputs(
        output_dir=str(output_dir)
    )

    # Find downloaded JSON files
    json_files = list(output_dir.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            "Sarvam completed the job but no JSON output was found."
        )

    # Read first JSON file
    with open(json_files[0], "r", encoding="utf-8") as f:
        result = json.load(f)

    transcript = result.get("transcript", "")

    if not transcript:
        raise RuntimeError(
            "Transcript not found in Sarvam output."
        )

    return transcript