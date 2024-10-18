from flask import Flask, request, jsonify
import os

app = Flask(__name__)  # Use double underscores

# Create an uploads directory if it doesn't exist
UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save the file to the uploads directory
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)  
    
    # Here you can add your prediction logic and generate a result
    # For now, we'll return a dummy prediction
    prediction = "Market will go up"  # Replace this with your prediction logic
    
    return jsonify({'prediction': prediction}), 200

if __name__ == '__main__':  # Use double underscores
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
