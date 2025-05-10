from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 确保上传目录存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(image_path, operation):
    img = cv2.imread(image_path)
    if img is None:
        return None
    
    if operation == 'resize':
        result = cv2.resize(img, (300, 300))
    elif operation == 'rotate':
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols/2, rows/2), 90, 1)
        result = cv2.warpAffine(img, M, (cols, rows))
    elif operation == 'grayscale':
        result = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif operation == 'blur':
        result = cv2.GaussianBlur(img, (5, 5), 0)
    elif operation == 'edge':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray, 100, 200)
    elif operation == 'brightness':
        # 增加亮度
        result = cv2.convertScaleAbs(img, alpha=1.2, beta=30)
    elif operation == 'contrast':
        # 增加对比度
        result = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
    elif operation == 'sharpen':
        # 锐化
        kernel = np.array([[-1,-1,-1],
                          [-1, 9,-1],
                          [-1,-1,-1]])
        result = cv2.filter2D(img, -1, kernel)
    elif operation == 'emboss':
        # 浮雕效果
        kernel = np.array([[-2,-1,0],
                          [-1,1,1],
                          [0,1,2]])
        result = cv2.filter2D(img, -1, kernel)
    elif operation == 'sketch':
        # 素描效果
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        blur = cv2.GaussianBlur(inverted, (21, 21), 0)
        inverted_blur = 255 - blur
        result = cv2.divide(gray, inverted_blur, scale=256.0)
    elif operation == 'sepia':
        # 复古效果
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        result = cv2.transform(img, kernel)
    elif operation == 'negative':
        # 负片效果
        result = 255 - img
    else:
        return None
    
    # 保存处理后的图片
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'processed_{operation}.jpg')
    cv2.imwrite(output_path, result)
    return output_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': '没有文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # 获取要执行的操作
        operation = request.form.get('operation', 'resize')
        
        # 处理图片
        processed_path = process_image(filepath, operation)
        if processed_path:
            return jsonify({
                'success': True,
                'original': f'/static/uploads/{filename}',
                'processed': f'/static/uploads/processed_{operation}.jpg'
            })
    
    return jsonify({'error': '文件处理失败'}), 400

if __name__ == '__main__':
    app.run(host='192.168.43.79', port=5000)  # 关键参数！