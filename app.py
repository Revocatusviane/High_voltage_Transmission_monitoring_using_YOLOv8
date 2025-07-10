from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, Response
import numpy as np
import joblib
from ultralytics import YOLO
from PIL import Image
import os
import requests
import cv2
from datetime import datetime
import uuid
import time
import logging
import platform

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Use the exact absolute path for uploads
app.config['UPLOAD_FOLDER'] = '/Users/machd/Desktop/Python/Transmission/static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Model paths using exact absolute paths
vc_model_path = '/Users/machd/Desktop/Python/Transmission/models/hv_transmission_model.joblib'
yolo_model_path = '/Users/machd/Desktop/Python/Transmission/models/best.pt'

# Load models with error handling
try:
    vc_clf = joblib.load(vc_model_path)
    logger.debug(f"Loaded voltage/current model: {vc_model_path}")
except Exception as e:
    logger.error(f"Failed to load voltage/current model: {e}")
    raise

try:
    yolo_clf = YOLO(yolo_model_path)
    logger.debug(f"Loaded YOLO model: {yolo_model_path}")
except Exception as e:
    logger.error(f"Failed to load YOLO model: {e}")
    raise

# Simulated database for measurements and ESP32 readings
measurements = []
latest_esp32_readings = {'voltage': None, 'current': None}
selected_image_path = None
camera = None
webcam_active = False

def get_location_from_ip():
    """Fetch approximate location using IP address for Fault location."""
    try:
        response = requests.get("https://ipinfo.io/json", timeout=5)
        if response.status_code == 200:
            data = response.json()
            lat, lon = map(float, data.get("loc", "0,0").split(","))
            response = requests.get(
                f"https://nominatim.openstreetmap.org/reverse?lat={lat}&lon={lon}&format=json",
                headers={'User-Agent': 'Voltguard/1.0'}, timeout=5)
            if response.status_code == 200:
                address = response.json().get('address', {})
                return {
                    'country': address.get('country', 'Unknown'),
                    'region': address.get('state', 'Unknown'),
                    'district': address.get('county', 'Unknown'),
                    'ward': address.get('suburb', 'Unknown'),
                    'street': address.get('road', 'Unknown'),
                    'latitude': str(lat),
                    'longitude': str(lon)
                }
    except Exception as e:
        logger.error(f"IP Location Fetch Error: {e}")
    return {'error': 'Unable to fetch machine location'}

def format_location(location_data):
    """Format location dictionary into a concise string, omitting Unknown fields."""
    if not location_data or 'error' in location_data:
        return ""
    parts = []
    for key in ['country', 'ward', 'street']:
        value = location_data.get(key, 'Unknown')
        if value != 'Unknown':
            parts.append(value)
    if location_data.get('latitude') and location_data.get('longitude'):
        parts.append(f"({location_data['latitude']}, {location_data['longitude']})")
    return ", ".join(parts) or "N/A"

@app.route('/esp32_readings', methods=['POST'])
def esp32_readings():
    """Receive voltage and current readings from ESP32."""
    global latest_esp32_readings
    try:
        data = request.get_json()
        voltage = float(data.get('voltage', 0))
        current = float(data.get('current', 0))
        latest_esp32_readings = {'voltage': voltage, 'current': current}
        logger.debug(f"ESP32 Readings received: voltage={voltage}, current={current}")
        return jsonify({'status': 'success', 'message': 'Readings received'})
    except Exception as e:
        logger.error(f"ESP32 Readings Error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/get_esp32_readings', methods=['GET'])
def get_esp32_readings():
    """Return the latest ESP32 readings as JSON."""
    logger.debug("Fetched ESP32 readings")
    return jsonify(latest_esp32_readings)

@app.route('/get_history', methods=['GET'])
def get_history():
    """Return the current measurement history for debugging."""
    formatted_measurements = [
        {**m, 'location': format_location(m.get('location_data', {}))}
        for m in measurements
    ]
    logger.debug(f"Returning measurement history: {len(formatted_measurements)} entries")
    return jsonify({
        'status': 'success',
        'history': formatted_measurements[-5:][::-1],
        'total_measurements': len(formatted_measurements)
    })

@app.route('/capture_webcam', methods=['POST'])
def capture_webcam():
    """Capture an image from the machine's webcam."""
    global selected_image_path
    logger.debug("Starting webcam capture")
    try:
        if platform.system() == "Darwin":
            logger.debug("Running on macOS, checking camera permissions")

        for attempt in range(2):
            for index in [0, 1, 2, 3]:
                logger.debug(f"Attempt {attempt + 1}, trying camera index: {index}")
                cap = cv2.VideoCapture(index)
                if cap.isOpened():
                    time.sleep(0.5)
                    ret, frame = cap.read()
                    if ret:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        temp_filename = f"webcam_{timestamp}_{uuid.uuid4().hex}.png"
                        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        img_pil = Image.fromarray(frame_rgb)
                        img_pil.save(temp_path, 'PNG')
                        cap.release()
                        
                        if os.path.exists(temp_path) and os.access(temp_path, os.R_OK) and os.path.getsize(temp_path) > 0:
                            selected_image_path = temp_path
                            img_size = img_pil.size
                            file_size = os.path.getsize(temp_path)
                            logger.debug(f"Webcam image saved: {temp_path}, size={img_size}, file_size={file_size} bytes")
                            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                                return jsonify({
                                    'status': 'success',
                                    'image_path': temp_path,
                                    'image_url': url_for('static', filename=f'uploads/{temp_filename}')
                                })
                            flash("Webcam image captured successfully!", "success")
                            return redirect(url_for('home'))
                        else:
                            logger.error(f"Failed to save webcam image: {temp_path}")
                            cap.release()
                            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                                return jsonify({'status': 'error', 'message': 'Failed to save webcam image'}), 500
                            flash("Error: Failed to save webcam image!", "error")
                            return redirect(url_for('home'))
                    cap.release()
                    logger.debug(f"Camera index {index} failed to capture frame")
                else:
                    logger.debug(f"Camera index {index} not opened")
            time.sleep(1)
        error_msg = "Could not access webcam! Ensure camera is connected, not in use, and permissions are granted."
        logger.error("Webcam capture failed: No accessible camera found after retries")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'error', 'message': error_msg}), 500
        flash(error_msg, "error")
        return redirect(url_for('home'))
    except Exception as e:
        error_msg = f"Error capturing webcam image: {e}. On macOS, ensure camera permissions are granted."
        logger.error(f"Webcam capture error: {e}")
        if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
            return jsonify({'status': 'error', 'message': error_msg}), 500
        flash(error_msg, "error")
        return redirect(url_for('home'))

@app.route('/start_webcam', methods=['POST'])
def start_webcam():
    """Start the webcam for live streaming."""
    global camera, webcam_active
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.error("Failed to access webcam")
            return jsonify({'status': 'error', 'message': 'Failed to access webcam'}), 500
        webcam_active = True
        logger.debug("Webcam started")
    return jsonify({'status': 'success', 'message': 'Webcam started'})

@app.route('/stop_webcam', methods=['POST'])
def stop_webcam():
    """Stop the webcam and release resources."""
    global camera, webcam_active, selected_image_path
    if camera is not None:
        camera.release()
        camera = None
        webcam_active = False
        selected_image_path = None
        logger.debug("Webcam stopped")
    return jsonify({'status': 'success', 'message': 'Webcam stopped'})

def generate_frames():
    """Generate frames for live webcam feed with YOLO analysis."""
    global camera, webcam_active, selected_image_path, measurements
    if camera is None or not camera.isOpened():
        logger.error("Camera not initialized")
        return

    while webcam_active:
        success, frame = camera.read()
        if not success:
            logger.error("Failed to capture frame")
            break

        # Save frame temporarily for analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_filename = f"webcam_{timestamp}_{uuid.uuid4().hex}.png"
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        img_pil.save(temp_path, 'PNG')
        selected_image_path = temp_path

        # Run analysis
        try:
            if latest_esp32_readings['voltage'] is None or latest_esp32_readings['current'] is None:
                logger.warning("No valid ESP32 readings")
                continue

            voltage = latest_esp32_readings['voltage']
            current = latest_esp32_readings['current']
            prediction = vc_clf.predict(np.array([[voltage, current]]))[0]

            img = Image.open(temp_path)
            results = yolo_clf(img)
            labels = results[0].names
            detected = results[0].boxes.cls.tolist()
            insulator_status = labels[int(detected[0])] if detected else "No Weakness Detected"

            location_data = {}
            if insulator_status == "WEAKENED INSULATOR":
                location_data = get_location_from_ip()

            original_filename = f"original_{timestamp}.png"
            predicted_filename = f"predicted_{timestamp}.png"
            original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            predicted_path = os.path.join(app.config['UPLOAD_FOLDER'], predicted_filename)

            img_pil.save(original_path, 'PNG')
            img_with_boxes = results[0].plot(font_size=10, labels=True, boxes=True, conf=False)
            img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_with_boxes_rgb)
            img_pil.save(predicted_path, 'PNG')

            formatted_location = format_location(location_data)
            measurement = {
                'id': len(measurements) + 1,
                'voltage': voltage,
                'current': current,
                'prediction': prediction,
                'insulator_status': insulator_status,
                'location': formatted_location,
                'location_data': location_data,
                'original_image_url': url_for('static', filename=f'uploads/{original_filename}'),
                'predicted_image_url': url_for('static', filename=f'uploads/{predicted_filename}'),
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            measurements.append(measurement)

            # Use annotated frame for streaming
            frame = img_with_boxes
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            logger.error("Failed to encode frame")
            continue
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Clean up
        if os.path.exists(temp_path):
            os.remove(temp_path)
        time.sleep(1.5)  # Adjust for processing speed

@app.route('/video_feed')
def video_feed():
    """Stream webcam feed with YOLO analysis."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/auto_analyze', methods=['POST'])
def auto_analyze():
    """Perform automatic analysis using the latest webcam image and ESP32 readings."""
    global selected_image_path, measurements
    logger.debug("Starting auto analysis")
    try:
        if latest_esp32_readings['voltage'] is None or latest_esp32_readings['current'] is None:
            error_msg = "No valid ESP32 readings available for analysis"
            logger.warning(error_msg)
            return jsonify({'status': 'error', 'message': error_msg}), 400

        voltage = latest_esp32_readings['voltage']
        current = latest_esp32_readings['current']
        logger.debug(f"Using ESP32 readings: voltage={voltage}, current={current}")

        logger.debug("Running voltage/current prediction")
        prediction = vc_clf.predict(np.array([[voltage, current]]))[0]
        logger.debug(f"Voltage/Current Prediction: {prediction}")

        if not selected_image_path or not os.path.exists(selected_image_path):
            error_msg = f"No valid image available at {selected_image_path}"
            logger.error(error_msg)
            return jsonify({'status': 'error', 'message': error_msg}), 400

        logger.debug(f"Loading image for YOLO analysis: {selected_image_path}")
        img = Image.open(selected_image_path)

        logger.debug("Running YOLO inference")
        results = yolo_clf(img)
        labels = results[0].names
        detected = results[0].boxes.cls.tolist()
        insulator_status = labels[int(detected[0])] if detected else "No Weakness Detected"
        logger.debug(f"YOLO Prediction: {insulator_status}")

        location_data = {}
        if insulator_status == "WEAKENED INSULATOR":
            logger.debug("Fault detected, fetching location")
            location_data = get_location_from_ip()
            logger.debug(f"Location data: {location_data}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"original_{timestamp}.png"
        predicted_filename = f"predicted_{timestamp}.png"

        original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
        predicted_path = os.path.join(app.config['UPLOAD_FOLDER'], predicted_filename)

        logger.debug(f"Saving original image: {original_path}")
        img_pil = img.convert('RGB')
        img_pil.save(original_path, 'PNG')
        if not (os.path.exists(original_path) and os.access(original_path, os.R_OK) and os.path.getsize(original_path) > 0):
            error_msg = f"Failed to save original image: {original_path}"
            logger.error(error_msg)
            return jsonify({'status': 'error', 'message': error_msg}), 500
        original_image_url = url_for('static', filename=f'uploads/{original_filename}')

        logger.debug(f"Saving predicted image: {predicted_path}")
        img_with_boxes = results[0].plot(font_size=10, labels=True, boxes=True, conf=False)
        img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_with_boxes_rgb)
        img_pil.save(predicted_path, 'PNG')
        if not (os.path.exists(predicted_path) and os.access(predicted_path, os.R_OK) and os.path.getsize(predicted_path) > 0):
            error_msg = f"Failed to save predicted image: {predicted_path}"
            logger.error(error_msg)
            return jsonify({'status': 'error', 'message': error_msg}), 500
        predicted_image_url = url_for('static', filename=f'uploads/{predicted_filename}')

        formatted_location = format_location(location_data)
        logger.debug(f"Formatted location: {formatted_location}")

        measurement = {
            'id': len(measurements) + 1,
            'voltage': voltage,
            'current': current,
            'prediction': prediction,
            'insulator_status': insulator_status,
            'location': formatted_location,
            'location_data': location_data,
            'original_image_url': original_image_url,
            'predicted_image_url': predicted_image_url,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        measurements.append(measurement)
        logger.debug(f"Measurement appended: {measurement}, total measurements: {len(measurements)}")

        history = [
            {**m, 'location': m['location']}
            for m in measurements[-5:][::-1]
        ]
        logger.debug(f"Returning history: {len(history)} entries")
        selected_image_path = None
        logger.debug("Auto analysis completed successfully")

        return jsonify({
            'status': 'success',
            'prediction': prediction,
            'insulator_status': insulator_status,
            'original_image_url': original_image_url,
            'predicted_image_url': predicted_image_url,
            'location_data': location_data,
            'history': history,
            'esp32_readings': latest_esp32_readings
        })

    except Exception as e:
        logger.error(f"Auto analysis error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500

@app.route('/', methods=['GET', 'POST'])
def home():
    global selected_image_path
    prediction, insulator_status, original_image_url, predicted_image_url = None, None, None, None
    location_data = {}
    image_preview_url = url_for('static', filename=f'uploads/{os.path.basename(selected_image_path)}') if selected_image_path else None

    if request.method == 'POST':
        logger.debug(f"POST request to /: {request.form}")
        if 'clear_table' in request.form:
            global measurements
            measurements = []
            selected_image_path = None
            flash("Measurement history cleared!", "success")
            logger.debug("Measurement history cleared")
            return redirect(url_for('home'))

        if 'analyze' in request.form:
            try:
                if latest_esp32_readings['voltage'] is None or latest_esp32_readings['current'] is None:
                    flash("Waiting for ESP32 voltage and current readings!", "warning")
                    logger.warning("No ESP32 readings available")
                    return redirect(url_for('home'))

                voltage = latest_esp32_readings['voltage']
                current = latest_esp32_readings['current']
                prediction = vc_clf.predict(np.array([[voltage, current]]))[0]
                logger.debug(f"Voltage/Current Prediction: {prediction}")

                if selected_image_path:
                    img = Image.open(selected_image_path)
                    logger.debug(f"Using selected image: {selected_image_path}")
                elif 'image' in request.files and request.files['image'].filename != '':
                    image_file = request.files['image']
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    temp_filename = f"upload_{timestamp}_{uuid.uuid4().hex}.png"
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
                    img_pil = Image.open(image_file).convert('RGB')
                    img_pil.save(temp_path, 'PNG')
                    if os.path.exists(temp_path) and os.access(temp_path, os.R_OK) and os.path.getsize(temp_path) > 0:
                        selected_image_path = temp_path
                        img = Image.open(selected_image_path)
                        image_preview_url = url_for('static', filename=f'uploads/{temp_filename}')
                        logger.debug(f"Uploaded image saved: {temp_path}")
                    else:
                        flash("Error: Failed to save uploaded image!", "error")
                        logger.error(f"Failed to save uploaded image: {temp_path}")
                        return redirect(url_for('home'))
                else:
                    flash("No image selected! Please capture a webcam image or upload a file.", "error")
                    logger.error("No image provided for analysis")
                    return redirect(url_for('home'))

                time.sleep(2)

                results = yolo_clf(img)
                labels = results[0].names
                detected = results[0].boxes.cls.tolist()
                insulator_status = labels[int(detected[0])] if detected else "No Weakness Detected"
                logger.debug(f"YOLO Prediction: {insulator_status}")

                if insulator_status == "WEAKENED INSULATOR":
                    location_data = get_location_from_ip()
                    logger.debug(f"Location data: {location_data}")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                original_filename = f"original_{timestamp}.png"
                predicted_filename = f"predicted_{timestamp}.png"

                original_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
                predicted_path = os.path.join(app.config['UPLOAD_FOLDER'], predicted_filename)

                img_pil = img.convert('RGB')
                img_pil.save(original_path, 'PNG')
                if os.path.exists(original_path) and os.access(original_path, os.R_OK) and os.path.getsize(original_path) > 0:
                    original_image_url = url_for('static', filename=f'uploads/{original_filename}')
                    logger.debug(f"Original image saved: {original_path}")
                else:
                    flash("Error: Failed to save original image!", "error")
                    logger.error(f"Failed to save original image: {original_path}")
                    return redirect(url_for('home'))

                img_with_boxes = results[0].plot(font_size=10, labels=True, boxes=True, conf=False)
                img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_with_boxes_rgb)
                img_pil.save(predicted_path, 'PNG')
                if os.path.exists(predicted_path) and os.access(predicted_path, os.R_OK) and os.path.getsize(predicted_path) > 0:
                    predicted_image_url = url_for('static', filename=f'uploads/{predicted_filename}')
                    logger.debug(f"Predicted image saved: {predicted_path}")
                else:
                    flash("Error: Failed to save predicted image!", "error")
                    logger.error(f"Failed to save predicted image: {predicted_path}")
                    return redirect(url_for('home'))

                formatted_location = format_location(location_data)
                logger.debug(f"Formatted location: {formatted_location}")

                measurement = {
                    'id': len(measurements) + 1,
                    'voltage': voltage,
                    'current': current,
                    'prediction': prediction,
                    'insulator_status': insulator_status,
                    'location': formatted_location,
                    'location_data': location_data,
                    'original_image_url': original_image_url,
                    'predicted_image_url': predicted_image_url,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                measurements.append(measurement)
                logger.debug(f"Measurement appended: {measurement}, total measurements: {len(measurements)}")
                selected_image_path = None
                flash("Measurement saved successfully!", "success")
                logger.debug("Measurement saved successfully")

            except ValueError as e:
                flash(f"Error processing data: {e}", "error")
                logger.error(f"Analysis error: {e}")

    history = [
        {**m, 'location': m['location']}
        for m in measurements[-5:][::-1]
    ]
    logger.debug(f"Rendering home with history: {len(history)} entries")
    return render_template('home.html', 
        prediction=prediction,
        insulator_status=insulator_status,
        original_image_url=original_image_url,
        predicted_image_url=predicted_image_url,
        location_data=location_data,
        history=history,
        esp32_readings=latest_esp32_readings,
        image_preview_url=image_preview_url,
        webcam_active=webcam_active)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/project')
def project():
    return render_template('project.html')

if __name__ == '__main__':
    try:
        app.run(host='0.0.0.0', port=5000, debug=True)
    finally:
        if camera is not None:
            camera.release()
            logger.debug("Camera released on shutdown")