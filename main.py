"""PaperTrail entrypoint: `uv run python main.py` -> Gradio UI on :7860.

Set PAPERTRAIL_MOCK=true to run the UI with no GPU/model (mock inference).
"""
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent / "src"))
from papertrail.app.main import main as demo_main

if __name__ == "__main__":
    demo_main()
