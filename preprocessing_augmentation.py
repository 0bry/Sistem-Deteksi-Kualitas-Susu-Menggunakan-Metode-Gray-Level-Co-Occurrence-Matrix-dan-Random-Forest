import os
import cv2
import Augmentor

def preprocess_images():
    os.makedirs("datas/preprocessed_data", exist_ok=True)
    
    count = 0
    for root, dirs, files in os.walk("datas/raw_data_cropped"):
        rel_path = os.path.relpath(root, "datas/raw_data_cropped")
        output_dir = os.path.join("datas/preprocessed_data", rel_path)
        os.makedirs(output_dir, exist_ok=True)
        
        for filename in files:
            if filename.lower().endswith(('.jpg', '.jpeg')):
                img = cv2.imread(os.path.join(root, filename))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (1024, 1024))
                cv2.imwrite(os.path.join(output_dir, filename), resized)
                count += 1
    
    return count

def augment_data():
    count = 0
    
    for root, dirs, files in os.walk("datas/preprocessed_data"):
        if not files:
            continue

        p = Augmentor.Pipeline(root)
        p.flip_left_right(probability=0.5)
        p.rotate90(probability=0.5)
        p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15)
        p.shear(probability=0.5, max_shear_left=25, max_shear_right=25)
        p.skew_tilt(probability=0.5, magnitude=0.7)
        
        original_count = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg'))])
        p.sample(original_count * 2)
        count += original_count * 2
    
    return count

def main():
    print("Starting preprocessing...")
    preprocessed_count = preprocess_images()
    print(f"{preprocessed_count} data preprocessed")
    
    print("Starting augmentation...")
    augmented_count = augment_data()
    print(f"{augmented_count} data augmented")

if __name__ == "__main__":
    main()
