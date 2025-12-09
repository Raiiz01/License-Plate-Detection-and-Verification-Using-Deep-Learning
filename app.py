import os
import re
import mysql.connector
from flask import Flask, render_template, request, jsonify, url_for
import cv2
import easyocr
from ultralytics import YOLO
from PIL import Image
import base64
from io import BytesIO
from flask_cors import CORS  

app = Flask(__name__)
CORS(app)  
model = YOLO('C:\\Users\\hp\\runs\\detect\\train5\\weights/best.pt')  
reader = easyocr.Reader(['en'])  


UPLOAD_FOLDER = 'C:/Users/hp/license_plate_detection/static/uploaded_images/'
RESULT_FOLDER = 'C:/Users/hp/license_plate_detection/static/results/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

def check_license_plate(plate_number):
    plate_number_cleaned = plate_number.replace(" ", "").upper()
    try:
        
        conn = mysql.connector.connect(
            host="127.0.0.1",  
            user="root",  
            password="", 
            database="plates" 
        )
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM vehicles WHERE license_plate = %s", (plate_number_cleaned,))
        result = cursor.fetchone()
        conn.close()  
        return result
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan_image():
    
    image_data = request.json.get('image')
    image_data = image_data.split(',')[1] 
    image_bytes = base64.b64decode(image_data)
    img = Image.open(BytesIO(image_bytes))

    img.save("scanned_image.jpg") 

    result_image, detected_text, plate_number = process_image("scanned_image.jpg")

    result_image_path = os.path.join(RESULT_FOLDER, 'result_scanned_image.jpg')
    result_image.save(result_image_path)

    plate_info = check_license_plate(plate_number) if plate_number else None

    return jsonify({
        'result_image': url_for('static', filename='results/' + 'result_scanned_image.jpg'),
        'detected_text': plate_number or detected_text,
        'plate_info': plate_info
    })

def process_image(image_path):
    img = cv2.imread(image_path)
    results = model.predict(image_path)
    detected_text = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box) 
            cropped_plate = img[y1:y2, x1:x2]

            ocr_results = reader.readtext(cropped_plate)
            for _, text, _ in ocr_results:
                detected_text.append(text)

    result_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(result_img)

    raw_text = ' '.join(detected_text).upper()
    matches = re.findall(r'\b[A-Z]{1,3}[\s\-]?[0-9]{3,5}\b', raw_text)
    plate_number = matches[0].replace(" ", "").replace("-", "") if matches else None

    return pil_image, raw_text, plate_number

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)  
