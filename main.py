import cv2
import numpy as np
import pandas as pd
import joblib
import time
import os
from datetime import datetime
from skimage.feature import graycomatrix, graycoprops
from skimage import img_as_ubyte
import RPi.GPIO as GPIO
from RPLCD.i2c import CharLCD

DISTANCE, ANGLE = 4, 135
BUTTON_PIN = 18
I2C_ADDR = 0x27
MODEL_PATH, CSV_LOG = "model.pkl", "milk_quality_log.csv"
QUALITY_MAP = {'1': 'Baik', '2': 'Rusak', '3': 'R. Berat'}

QUALITY_PARAMS = {
    '1': {'bacteria': '150K-1.5M CFU', 'acid': '< 0.01', 'ph': '6.15 - 6.70'},
    '2': {'bacteria': '2.5M-40M CFU', 'acid': '0.01 - 0.10', 'ph': '5.30 - 6.05'},
    '3': {'bacteria': '90M-1.4B CFU', 'acid': '0.12 - 0.23', 'ph': '4.40 - 5.15'}
}

class MilkQualityDetector:
    def __init__(self):
        os.makedirs("ori", exist_ok=True)
        os.makedirs("pro", exist_ok=True)
        
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(BUTTON_PIN, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        self.lcd = CharLCD('PCF8574', I2C_ADDR, port=1, cols=16, rows=2)
        self.lcd.clear()
        self.lcd.write_string("Milk Quality")
        self.lcd.cursor_pos = (1, 0)
        self.lcd.write_string("Detector Ready")
        
        self.camera = cv2.VideoCapture(0, cv2.CAP_V4L2)
        
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
        self.camera.set(cv2.CAP_PROP_FPS, 10)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        time.sleep(1)
        self.camera.read()
        
        self.model = joblib.load(MODEL_PATH)
        self.image_counter = 1

    def display_message(self, line1, line2=""):
        self.lcd.clear()
        self.lcd.write_string(line1)
        if line2:
            self.lcd.cursor_pos = (1, 0)
            self.lcd.write_string(line2)

    def process_detection(self):
        self.display_message("Capturing...", "Please wait") 
        start_time = time.time()
        
        self.camera.read()
        self.camera.read()
        time.sleep(0.3)
        ret, frame = self.camera.read()
        
        img_height, img_width = frame.shape[:2]
        
        if img_width != 2560 or img_height != 1440:
            x1 = int(846 * img_width / 2560)
            y1 = int(387 * img_height / 1440)
            x2 = int(1617 * img_width / 2560)
            y2 = int(1158 * img_height / 1440)
        else:
            x1, y1 = 846, 387
            x2, y2 = 1617, 1158
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_path = f"ori/original_{self.image_counter}_{timestamp}.jpg"
        cv2.imwrite(original_path, frame)
        
        cropped = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (1024, 1024))
        
        processed_path = f"pro/processed_{self.image_counter}_{timestamp}.jpg"
        cv2.imwrite(processed_path, resized)
        
        gray_ubyte = img_as_ubyte(resized)
        gray_ubyte = (gray_ubyte // 4) * 4
        glcm = graycomatrix(gray_ubyte, [DISTANCE], [np.radians(ANGLE)],
                           levels=256, symmetric=True, normed=True)
        
        features = {prop: graycoprops(glcm, prop)[0, 0]
                   for prop in ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity']} 
        
        feature_names = ['correlation', 'contrast', 'homogeneity', 'energy', 'dissimilarity']
        feature_df = pd.DataFrame([[features[f] for f in feature_names]], columns=feature_names)
        quality = str(self.model.predict(feature_df)[0])
        quality_text = QUALITY_MAP[quality]
        
        comp_time = round(time.time() - start_time, 2)
        self.display_message(f"Kual: {quality_text}", f"Time: {comp_time}s")
        time.sleep(3)
        
        params = QUALITY_PARAMS[quality]
        self.display_message("Bakteri:", params['bacteria'])
        time.sleep(2)
        self.display_message("Total Asam:", params['acid'])
        time.sleep(2)
        self.display_message("pH:", params['ph'])
        time.sleep(2)
        
        log_data = {
            'number': self.image_counter,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            **features,
            'quality': quality_text,
            'computational_time': comp_time
        }

        df = pd.DataFrame([log_data])
        df.to_csv(CSV_LOG, mode='a', header=not os.path.exists(CSV_LOG), index=False)
        self.image_counter += 1
        
    def run(self):
        self.display_message("Press button", "to detect milk")
        
        while True:
            if GPIO.input(BUTTON_PIN) == GPIO.LOW:
                self.process_detection()
                while GPIO.input(BUTTON_PIN) == GPIO.LOW:
                    time.sleep(0.1)
                self.display_message("Press button", "to detect milk")
            time.sleep(0.1)
            
    def cleanup(self):
        self.camera.release()
        self.lcd.clear()
        GPIO.cleanup()

if __name__ == "__main__":
    detector = MilkQualityDetector()
    try:
        detector.run()
    except KeyboardInterrupt:
        pass
    finally:
        detector.cleanup()
