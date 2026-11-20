# DSP starter — setup

Copy the `backend/`, `tests/`, `examples/` folders and `requirements.txt` into the cloned repository root. Run from that root:

```bash
python -m venv .venv
# macOS / Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest -q
python -m examples.run_audio_engine path/to/test.wav --gain -6
```

This is an offline audio engine, not a streaming or real-time audio engine. PCM export is guarded against sample peaks above 1.0; there is no automatic normalization.
