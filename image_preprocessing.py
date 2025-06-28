import os
import cv2
from pathlib import Path

def process_images():
    input_dir = "data/raw_data_cropped"
    output_dir = "data/preprocessed_data"
    
    os.makedirs(output_dir, exist_ok=True)

    count = 0
    
    print(f"Looking for images in: {os.path.abspath(input_dir)}")
    
    if not os.path.exists(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist!")
        return
    
    for root, dirs, files in os.walk(input_dir):
        print(f"Checking directory: {root}")
        print(f"Found files: {files}")
        
        rel_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, rel_path)
        os.makedirs(current_output_dir, exist_ok=True)

        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                input_path = os.path.join(root, filename)
                print(f"Processing: {input_path}")
                
                img = cv2.imread(input_path)
                if img is None:
                    print(f"Error: Could not load image {input_path}")
                    continue
                    
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (1024, 1024))
                output_path = os.path.join(current_output_dir, filename)
                cv2.imwrite(output_path, resized)
                print(f"Saved: {output_path}")
                count += 1
    
    print(f"{count} images processed")

if __name__ == "__main__":
    process_images()