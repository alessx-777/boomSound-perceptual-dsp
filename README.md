
# SONICA Lab — Perceptual DSP

A psychoacoustic bass enhancement and digital signal
processing workbench for compact loudspeakers.

## Project Goal

Can digital signal processing overcome the perceptual
limitations imposed by a loudspeaker's physical design?

SONICA Lab explores how harmonic synthesis, equalization,
and listener-guided optimization can improve perceived
bass reproduction without physically extending the
loudspeaker's low-frequency capabilities.

## Main Components

### 1. Web Interface
- Audio file upload
- DSP controls and visualization
- A/B listening comparisons
- Results and preset management

### 2. DSP Engine
- Audio loading and processing
- Parametric EQ and digital filters
- Psychoacoustic bass enhancement
- Hybrid bass processing
- Audio analysis and export

### 3. Speaker Model & Perceptual Optimization
- Loudspeaker frequency-response import
- Linear speaker-response simulation
- DSP candidate generation
- Listener preference collection
- Configuration comparison and optimization

## Technology Stack

- Python
- NumPy / SciPy
- SoundFile
- FastAPI
- HTML / CSS / JavaScript

## Project Structure

- `backend/dsp/` — Digital signal processing
- `backend/speaker/` — Speaker modeling and optimization
- `backend/api/` — Backend API
- `frontend/` — Web application
- `tests/` — Automated tests
- `docs/` — Documentation
- `examples/` — Example configurations

## Development Workflow

The main branch contains integrated project code.

Each team member works on a dedicated feature branch:

- `feature/frontend`
- `feature/dsp-engine`
- `feature/speaker-model`

Changes are integrated through pull requests.

## Academic Context

Developed as a university project for ACTAM.

## Status

Initial development — Audio Engine v0.1.
