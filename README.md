# text2wav

A tiny Python CLI that hides a text message **visually** inside a `.wav` file as **spectrogram art** (viewable in **Audacity**).

It works by:
1) generating a temporary PNG with your text  
2) converting image brightness → audio tones across frequencies  
3) deleting the temporary PNG automatically

> ⚠️ Note: This is *spectrogram-visible* stego (CTF-style), not metadata embedding.

## Demo

Open the generated WAV in **Audacity**:
- Track dropdown → **Spectrogram**
- Recommended settings:
  - **Window size:** 2048 or 4096
  - **Max Frequency:** 8000–12000 Hz
  - **Range:** ~80 dB

## Install

```bash
pip install -r requirements.txt
```

## Usage

```bash
python3 hide_in_wav.py -o secret.wav
```

It will prompt:

```text
what do u wanna hide? >
```

### Options

```bash
python3 hide_in_wav.py --help
```

Common tweaks:

- Make it longer (more time to read in spectrogram):
```bash
python3 hide_in_wav.py -o secret.wav --pps 80
```

- Change frequency range:
```bash
python3 hide_in_wav.py -o secret.wav --fmin 300 --fmax 10000
```

- Use a custom font:
```bash
python3 hide_in_wav.py -o secret.wav --font "/path/to/font.ttf"
```

## Files

- `hide_in_wav.py` — the CLI app
- `requirements.txt` — dependencies

## License

MIT
