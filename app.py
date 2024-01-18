from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import cv2
import numpy as np
from image_processing_script  import AR

app = Flask(__name__, template_folder='templates')
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
AR_instance = AR()

@app.route('/')
def index():
    return render_template('index.html')  

def base64_to_image(base64_str):
    # Decode base64 string to OpenCV image
    img_data = base64.b64decode(base64_str)
    np_arr = np.fromstring(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

def image_to_base64(image):
    # Convert OpenCV image to base64 string
    _, buffer = cv2.imencode('.png', image)
    base64_str = base64.b64encode(buffer).decode('utf-8')
    return base64_str

@socketio.on('image')
def handle_image(json):
    image_data = json['data'].split(",")[1]  # Remove base64 prefix
    image = base64_to_image(image_data)  # Convert to image
    processed_image = AR_instance.process_frame(image)  # Process image
    response_data = image_to_base64(processed_image)  # Convert back to base64
    emit('response_back', {'data': response_data})

if __name__ == '__main__':
    # SSL context with self-signed certificate and private key
    ssl_context = ('./server.crt', './server.key')

    # Running the Socket.IO server with SSL context
    socketio.run(app, host='0.0.0.0', port=443, debug=True, ssl_context=ssl_context)
