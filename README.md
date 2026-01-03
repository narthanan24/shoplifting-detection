# Shoplifting Detection System

A Python-based system for detecting suspicious shoplifting-like activity in CCTV-style videos using YOLOv8 object detection and person tracking.

## Features

- **Person & Object Detection**: Uses YOLOv8 pretrained model to detect persons, backpacks, handbags, bottles, and other items
- **Person Tracking**: Tracks persons across frames using ByteTrack algorithm
- **Suspicious Behavior Detection**: Flags suspicious activity based on heuristics:
  - Person stays near shelves longer than threshold (default: 15 seconds)
  - Person picks up an item and item disappears near bag/body
  - Person exits frame shortly after item disappears
- **Video Clip Extraction**: Automatically extracts suspicious segments using FFmpeg


## Requirements

- Python 3.8 or higher
- FFmpeg (for video clip extraction)

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Install FFmpeg:**
   
   **macOS:**
   ```bash
   brew install ffmpeg
   ```
   
   **Linux (Ubuntu/Debian):**
   ```bash
   sudo apt-get update
   sudo apt-get install ffmpeg
   ```
   
   **Windows:**
   Download from [FFmpeg website](https://ffmpeg.org/download.html) and add to PATH

## Usage

### Basic Usage

```bash
python main.py path/to/video.mp4
```

### Advanced Usage

```bash
python main.py path/to/video.mp4 \
    --output-dir output_clips \
    --shelf-region 100 200 800 600 \
    --time-threshold 15.0 \
    --item-buffer 3.0 \
    --exit-buffer 5.0
```

### Arguments

- `video_path` (required): Path to input MP4 video file
- `--output-dir`: Output directory for extracted clips (default: `output_clips`)
- `--shelf-region`: Shelf region coordinates as `X1 Y1 X2 Y2` (optional, entire frame used if not specified)
- `--time-threshold`: Time in seconds person must stay near shelves to be flagged (default: 15.0)
- `--item-buffer`: Buffer time in seconds for item disappearance check (default: 3.0)
- `--exit-buffer`: Buffer time in seconds for exit check after item disappears (default: 5.0)

### Example

```bash
# Process video with default settings
python main.py store_video.mp4

# Process video with custom shelf region
python main.py store_video.mp4 --shelf-region 50 100 700 500 --time-threshold 20.0
```

## Output

The system generates:

1. **Console Output**: Summary of detected suspicious events with:
   - Person ID
   - Start and end timestamps
   - Reason for flagging
   - Clip filename

2. **Video Clips**: Extracted MP4 files saved in the output directory (default: `output_clips/`)

### Example Output

```
============================================================
SUSPICIOUS EVENTS SUMMARY
============================================================

Event 1:
  Person ID: 1
  Start Time: 00:00:15.234
  End Time: 00:00:32.567
  Duration: 17.33 seconds
  Reason: Stayed near shelves for 17.3s
  Clip File: output_clips/suspicious_event_1_1.mp4

Event 2:
  Person ID: 2
  Start Time: 00:01:45.123
  End Time: 00:01:53.456
  Duration: 8.33 seconds
  Reason: Item disappeared near person, then person exited
  Clip File: output_clips/suspicious_event_2_2.mp4
```

## Project Structure

```
shoplifting-detector/
├── main.py              # Main orchestration script
├── detector.py          # YOLOv8 object detection
├── tracker.py           # ByteTrack person tracking
├── clipper.py           # FFmpeg video clip extraction
├── utils.py             # Utility functions
├── requirements.txt     # Python dependencies
└── README.md           # This file
```

## How It Works

1. **Video Input**: Loads MP4 video and extracts FPS and frame information
2. **Detection**: Each frame is processed with YOLOv8 to detect persons and items
3. **Tracking**: Persons are tracked across frames using ByteTrack algorithm
4. **Behavior Analysis**: Heuristics check for:
   - Extended time near shelves
   - Item disappearance near person
   - Quick exit after item interaction
5. **Timestamp Extraction**: Start/end times are extracted and overlapping events are merged
6. **Clip Extraction**: FFmpeg extracts suspicious segments as separate MP4 files

## Limitations

- Uses pretrained models (no custom training)
- Heuristics-based detection (may have false positives/negatives)
- Requires FFmpeg for clip extraction
- Performance depends on video resolution and length

## Troubleshooting

### FFmpeg not found
If you see "FFmpeg is not installed", install FFmpeg using the instructions above. The system will still detect suspicious events but won't extract clips.

### Low detection accuracy
- Ensure video quality is good (clear lighting, stable camera)
- Adjust `--time-threshold` based on your use case
- Specify `--shelf-region` to focus on specific areas

### Slow processing
- YOLOv8n (nano) is used by default for speed. For better accuracy, you can modify `detector.py` to use `yolov8s.pt` or `yolov8m.pt`
- Processing time depends on video length and resolution

## License

This project uses open-source libraries:
- Ultralytics YOLOv8 (AGPL-3.0)
- OpenCV (Apache 2.0)
- NumPy, SciPy, FilterPy (various open-source licenses)

## Notes

- The system is designed for demonstration purposes
- Real-world deployment may require additional tuning and validation
- Always review detected events manually before taking action

