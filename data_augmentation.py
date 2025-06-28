import Augmentor #augmentasi gambar menggunakan Augmentor
import os #untuk menangani direktori dan file

def augment_data():
    count = 0
    
    for root, dirs, files in os.walk("data/preprocessed_data"): #meastikan direktori yang berisi data
        if not files:
            continue

        p = Augmentor.Pipeline(root) #inisialisasi pipeline augmentasi
        p.flip_left_right(probability=0.5) #flip horizontal
        p.rotate90(probability=0.5) # rotasi 90 derajat
        p.rotate(probability=0.7, max_left_rotation=15, max_right_rotation=15) # rotasi acak
        p.shear(probability=0.5, max_shear_left=25, max_shear_right=25) # shear
        p.skew_tilt(probability=0.5, magnitude=0.7)# skew tilt
        
        original_count = len([f for f in files if f.lower().endswith(('.jpg', '.jpeg'))]) # menghitung jumlah gambar asli
        p.sample(original_count * 2) # menghasilkan augmentasi dua kali lipat dari jumlah gambar asli
        count += original_count * 2 # total gambar yang dihasilkan
    
    # augmentasi(504 gambar) + gambar asli(252 gambar) = 756 gambar
    print(f"{count} images augmented") 

if __name__ == "__main__":
    augment_data()