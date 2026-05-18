import os
import subprocess
from pathlib import Path

def main():
    shoplifting_dir = "/Users/narthanan24/Desktop/shoplifting detection/archive/shoplifting"
    normal_dir = "/Users/narthanan24/Desktop/shoplifting detection/archive/normal"
    
    videos = []
    # 5 shoplifting and 5 normal videos
    for i in range(1, 11):
        videos.append(f"{shoplifting_dir}/shoplifting-{i}.mp4")
        videos.append(f"{normal_dir}/normal-{i}.mp4")
    
    output_base = "evaluation_results_10_improved"
    os.makedirs(output_base, exist_ok=True)
    
    print("Starting batch evaluation for 10 videos...")
    for video in videos:
        print(f"Processing {video}...")
        video_name = Path(video).stem
        output_dir = os.path.join(output_base, video_name)
        os.makedirs(output_dir, exist_ok=True)
        
        # Run main.py using the virtual environment python
        cmd = [
            "./venv/bin/python", "main.py",
            video,
            "--output-dir", output_dir
        ]
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Finished {video_name} successfully.")
            with open(os.path.join(output_dir, "log.txt"), "w") as f:
                f.write(result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video_name}. See error.txt for details.")
            with open(os.path.join(output_dir, "error.txt"), "w") as f:
                f.write(e.stderr)
                if e.stdout:
                    f.write("\n\nSTDOUT:\n" + e.stdout)

if __name__ == "__main__":
    main()
