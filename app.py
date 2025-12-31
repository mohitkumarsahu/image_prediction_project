from flask import Flask, render_template_string, request, jsonify
from keras.applications import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

# Create uploads folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the VGG16 pre-trained model
print("Loading VGG16 model...")
model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
print("Model loaded successfully!")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Image Classifier</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #f0f4f8 0%, #d9e2ec 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 900px;
            margin: 0 auto;
            background: white;
            border-radius: 24px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 40px;
            text-align: center;
            color: white;
        }
        
        .ai-icon {
            width: 80px;
            height: 80px;
            margin: 0 auto 20px;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 45px;
            backdrop-filter: blur(10px);
        }
        
        h1 {
            font-size: 36px;
            margin-bottom: 10px;
            font-weight: 700;
        }
        
        .subtitle {
            font-size: 16px;
            opacity: 0.95;
            font-weight: 300;
        }
        
        .content {
            padding: 40px;
        }
        
        .upload-section {
            border: 3px dashed #cbd5e0;
            border-radius: 16px;
            padding: 40px;
            text-align: center;
            background: #f7fafc;
            transition: all 0.3s ease;
            cursor: pointer;
            position: relative;
        }
        
        .upload-section:hover {
            border-color: #667eea;
            background: #edf2f7;
        }
        
        .upload-section.dragover {
            border-color: #667eea;
            background: #e6f0ff;
            transform: scale(1.02);
        }
        
        .upload-icon {
            font-size: 60px;
            margin-bottom: 20px;
            opacity: 0.6;
        }
        
        .upload-text {
            color: #4a5568;
            font-size: 18px;
            margin-bottom: 10px;
        }
        
        .upload-subtext {
            color: #718096;
            font-size: 14px;
        }
        
        #fileInput {
            display: none;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 16px 40px;
            border-radius: 12px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
            display: none;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(102, 126, 234, 0.4);
        }
        
        .btn:active {
            transform: translateY(0);
        }
        
        .btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .preview-section {
            margin-top: 30px;
            display: none;
        }
        
        .preview-section.show {
            display: block;
            animation: fadeIn 0.5s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .image-preview {
            text-align: center;
            margin-bottom: 30px;
        }
        
        .image-preview img {
            max-width: 100%;
            max-height: 400px;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
        }
        
        .result-section {
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
            border-radius: 16px;
            padding: 30px;
            border-left: 5px solid #667eea;
        }
        
        .result-title {
            color: #2d3748;
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .result-box {
            background: white;
            border: 2px solid #e2e8f0;
            border-radius: 12px;
            padding: 20px;
            margin-top: 15px;
        }
        
        .prediction-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 15px;
            background: #f7fafc;
            border-radius: 8px;
            margin-bottom: 10px;
        }
        
        .prediction-item:last-child {
            margin-bottom: 0;
        }
        
        .prediction-label {
            color: #2d3748;
            font-size: 18px;
            font-weight: 600;
            text-transform: capitalize;
        }
        
        .prediction-confidence {
            color: #667eea;
            font-size: 16px;
            font-weight: 700;
        }
        
        .confidence-bar {
            width: 100%;
            height: 8px;
            background: #e2e8f0;
            border-radius: 4px;
            margin-top: 8px;
            overflow: hidden;
        }
        
        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            border-radius: 4px;
            transition: width 0.8s ease;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 30px;
        }
        
        .loading.show {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .loading-text {
            color: #4a5568;
            font-size: 16px;
        }
        
        .reset-btn {
            background: #e2e8f0;
            color: #4a5568;
            margin-left: 10px;
        }
        
        .reset-btn:hover {
            background: #cbd5e0;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.1);
        }
        
        .error-message {
            background: #fed7d7;
            color: #c53030;
            padding: 15px;
            border-radius: 8px;
            margin-top: 15px;
            display: none;
        }
        
        .error-message.show {
            display: block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="ai-icon">🧠</div>
            <h1>AI Image Classifier</h1>
            <p class="subtitle">Powered by VGG16 Deep Learning Model</p>
        </div>
        
        <div class="content">
            <div class="upload-section" id="uploadSection" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📸</div>
                <div class="upload-text">Click to upload or drag and drop</div>
                <div class="upload-subtext">Supports: JPG, JPEG, PNG, GIF, BMP</div>
                <input type="file" id="fileInput" accept="image/*">
            </div>
            
            <button class="btn" id="classifyBtn" onclick="classifyImage()">
                🔍 Classify Image
            </button>
            
            <button class="btn reset-btn" id="resetBtn" onclick="resetApp()" style="display: none;">
                🔄 Upload Another Image
            </button>
            
            <div class="error-message" id="errorMessage"></div>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div class="loading-text">Analyzing image with AI...</div>
            </div>
            
            <div class="preview-section" id="previewSection">
                <div class="image-preview" id="imagePreview"></div>
                
                <div class="result-section" id="resultSection" style="display: none;">
                    <div class="result-title">
                        <span>✨</span>
                        <span>Prediction Results</span>
                    </div>
                    <div class="result-box" id="resultBox"></div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        let selectedFile = null;
        
        const fileInput = document.getElementById('fileInput');
        const uploadSection = document.getElementById('uploadSection');
        const classifyBtn = document.getElementById('classifyBtn');
        const resetBtn = document.getElementById('resetBtn');
        const previewSection = document.getElementById('previewSection');
        const imagePreview = document.getElementById('imagePreview');
        const resultSection = document.getElementById('resultSection');
        const resultBox = document.getElementById('resultBox');
        const loading = document.getElementById('loading');
        const errorMessage = document.getElementById('errorMessage');
        
        // File input change handler
        fileInput.addEventListener('change', function(e) {
            handleFile(e.target.files[0]);
        });
        
        // Drag and drop handlers
        uploadSection.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadSection.classList.add('dragover');
        });
        
        uploadSection.addEventListener('dragleave', function(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
        });
        
        uploadSection.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadSection.classList.remove('dragover');
            handleFile(e.dataTransfer.files[0]);
        });
        
        function handleFile(file) {
            if (!file) return;
            
            if (!file.type.startsWith('image/')) {
                showError('Please select a valid image file');
                return;
            }
            
            selectedFile = file;
            
            const reader = new FileReader();
            reader.onload = function(e) {
                imagePreview.innerHTML = `<img src="${e.target.result}" alt="Preview">`;
                previewSection.classList.add('show');
                classifyBtn.style.display = 'inline-block';
                resetBtn.style.display = 'inline-block';
                resultSection.style.display = 'none';
                errorMessage.classList.remove('show');
            };
            reader.readAsDataURL(file);
        }
        
        async function classifyImage() {
            if (!selectedFile) {
                showError('Please select an image first');
                return;
            }
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            
            loading.classList.add('show');
            classifyBtn.disabled = true;
            resultSection.style.display = 'none';
            errorMessage.classList.remove('show');
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (data.error) {
                    showError(data.error);
                } else {
                    displayResults(data.predictions);
                }
            } catch (error) {
                showError('Error: ' + error.message);
            } finally {
                loading.classList.remove('show');
                classifyBtn.disabled = false;
            }
        }
        
        function displayResults(predictions) {
            resultBox.innerHTML = '';
            
            predictions.forEach(pred => {
                const confidence = (pred.confidence * 100).toFixed(2);
                
                const predItem = document.createElement('div');
                predItem.className = 'prediction-item';
                predItem.innerHTML = `
                    <div>
                        <div class="prediction-label">${pred.label.replace(/_/g, ' ')}</div>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%"></div>
                        </div>
                    </div>
                    <div class="prediction-confidence">${confidence}%</div>
                `;
                
                resultBox.appendChild(predItem);
            });
            
            resultSection.style.display = 'block';
        }
        
        function showError(message) {
            errorMessage.textContent = message;
            errorMessage.classList.add('show');
        }
        
        function resetApp() {
            selectedFile = null;
            fileInput.value = '';
            imagePreview.innerHTML = '';
            previewSection.classList.remove('show');
            classifyBtn.style.display = 'none';
            resetBtn.style.display = 'none';
            resultSection.style.display = 'none';
            errorMessage.classList.remove('show');
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Image prediction endpoint using VGG16 model
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload an image.'}), 400
        
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load and preprocess the image
        img = image.load_img(filepath, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)  # Add batch dimension
        x = preprocess_input(x)  # Preprocess for VGG16
        
        # Make prediction
        pred = model.predict(x)
        
        # Decode predictions (top 3 predictions)
        decode_prediction = decode_predictions(pred, top=1)[0]
        
        # Format results
        predictions = []
        for p in decode_prediction:
            predictions.append({
                'label': p[1],
                'confidence': float(p[2])
            })
        
        # Clean up temporary file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({'predictions': predictions})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 AI Image Classifier is starting...")
    print("="*50)
    
    # Get port from environment variable (Render provides this)
    port = int(os.environ.get('PORT', 10000))
    
    print(f"🔡 Server running on port: {port}")
    print("="*50 + "\n")
    
    # Run with debug=False for production
    app.run(debug=False, host='0.0.0.0', port=port)