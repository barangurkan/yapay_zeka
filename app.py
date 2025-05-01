from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, Response
from ultralytics import YOLO
import cv2
import numpy as np
import os
from datetime import datetime
from models import Session, OccupancyRecord, User, create_admin
from functools import wraps
import json
from sqlalchemy.orm import joinedload
import threading
import time
from werkzeug.utils import secure_filename
import csv
from io import StringIO

camera = None
last_frame = None
last_analysis = None
analysis_lock = threading.Lock()

app = Flask(__name__)
app.secret_key = 'your-secret-key-here' 

# YOLO modeli yükle
model = YOLO("yolo12m.pt")


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Bu sayfayı görüntülemek için giriş yapmalısınız.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)

    return decorated_function


def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Bu sayfayı görüntülemek için giriş yapmalısınız.', 'warning')
            return redirect(url_for('login'))

        db_session = Session()
        user = db_session.query(User).filter_by(id=session['user_id']).first()
        db_session.close()

        if not user or not user.is_admin:
            flash('Bu sayfaya erişim yetkiniz yok.', 'danger')
            return redirect(url_for('index'))
        return f(*args, **kwargs)

    return decorated_function


class Camera:
    def __init__(self):
        try:
            self.cap = cv2.VideoCapture(0)

            if not self.cap.isOpened():
                raise RuntimeError('Kameraya bağlanılamadı.')

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        except Exception as e:
            print(f"Kamera başlatma hatası: {str(e)}")
            raise

        self.last_analysis = {
            'sitting_count': 0,
            'total_people': 0,
            'total_chairs': 0,
            'empty_chairs': 0,
            'timestamp': datetime.now()
        }

    def analyze_frame(self, frame):
        try:
            results = model(frame)[0]

            person_boxes = []
            chair_boxes = []

            for result in results.boxes:
                cls = int(result.cls[0])
                label = model.names[cls]
                conf = float(result.conf[0])

                if conf < 0.3:
                    continue

                x1, y1, x2, y2 = map(int, result.xyxy[0])

                if label == "person":
                    person_boxes.append([x1, y1, x2, y2])
                elif label in ["chair", "bench", "couch"]:
                    chair_boxes.append([x1, y1, x2, y2])

            sitting_people = 0
            for person_box in person_boxes:
                for chair_box in chair_boxes:
                    if is_sitting_on_chair(person_box, chair_box):
                        sitting_people += 1
                        break

            cv2.rectangle(frame, (person_box[0], person_box[1]),
                          (person_box[2], person_box[3]), (0, 0, 255), 2)
            for box in chair_boxes:
                cv2.rectangle(frame, (box[0], box[1]),
                              (box[2], box[3]), (255, 165, 0), 2)

            cv2.putText(frame, f"Oturanlar: {sitting_people}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Toplam Kisi: {len(person_boxes)}", (30, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(frame, f"Sandalyeler: {len(chair_boxes)}", (30, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
            self.last_analysis = {
                'sitting_count': sitting_people,
                'total_people': len(person_boxes),
                'total_chairs': len(chair_boxes),
                'empty_chairs': len(chair_boxes) - sitting_people,
                'timestamp': datetime.now()
            }

            return frame

        except Exception as e:
            print(f"Frame analiz hatası: {str(e)}")
            return frame

    def get_frame(self):
        if not self.cap.isOpened():
            return None

        ret, frame = self.cap.read()
        if not ret:
            return None

        analyzed_frame = self.analyze_frame(frame.copy())
        return analyzed_frame

    def __del__(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


def generate_frames():
    global camera
    if camera is None:
        camera = Camera()
    while True:
        frame = camera.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.033)  # ~30 FPS


@app.route('/')
def index():
    if 'user_id' not in session:
        return redirect(url_for('login'))

    global camera
    if camera is None:
        camera = Camera()

    if camera and hasattr(camera, 'last_analysis'):
        with analysis_lock:
            analysis = camera.last_analysis.copy()
            if hasattr(analysis['timestamp'], 'isoformat'):
                analysis['timestamp'] = analysis['timestamp'].isoformat()
    else:
        analysis = {
            'sitting_count': 0,
            'total_people': 0,
            'total_chairs': 0,
            'empty_chairs': 0,
            'timestamp': datetime.now().isoformat()
        }

    db_session = Session()
    user_records = db_session.query(OccupancyRecord).filter_by(user_id=session['user_id']).order_by(
        OccupancyRecord.timestamp.desc()).limit(5).all()
    db_session.close()

    try:
        if camera is None or not camera.cap or not camera.cap.isOpened():
            if camera is not None:
                camera.__del__()
            camera = Camera()
            time.sleep(2) 
    except Exception as e:
        print(f"Kamera başlatma hatası: {str(e)}")

    return render_template('index.html', current_status=analysis, user_records=user_records)


@app.route('/video_feed')
def video_feed():
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/check_camera')
@admin_required
def check_camera():
    global camera
    try:
        if camera is None or not camera.cap or not camera.cap.isOpened():
            if camera is not None:
                camera.__del__()
            camera = Camera()
            time.sleep(2) 

            if camera.cap is not None and camera.cap.isOpened():
                return jsonify({"status": "success", "message": "Kamera başarıyla başlatıldı"})
            else:
                return jsonify({"status": "error", "message": "Kamera başlatılamadı"})
        return jsonify({"status": "success", "message": "Kamera zaten çalışıyor"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/get_current_status')
@login_required
def get_current_status():
    global camera
    try:
        if camera is None:
            camera = Camera()

        if camera and hasattr(camera, 'last_analysis'):
            with analysis_lock:
                analysis = camera.last_analysis.copy()
                if hasattr(analysis['timestamp'], 'isoformat'):
                    analysis['timestamp'] = analysis['timestamp'].isoformat()
                return jsonify(analysis)
    except Exception as e:
        print(f"İstatistik güncelleme hatası: {str(e)}")

    return jsonify({
        'sitting_count': 0,
        'total_people': 0,
        'total_chairs': 0,
        'empty_chairs': 0,
        'timestamp': datetime.now().isoformat()
    })


def is_sitting_on_chair(person_box, chair_box):
    person_center_x = (person_box[0] + person_box[2]) / 2
    person_center_y = (person_box[1] + person_box[3]) / 2

    chair_center_x = (chair_box[0] + chair_box[2]) / 2
    chair_center_y = (chair_box[1] + chair_box[3]) / 2

    chair_width = chair_box[2] - chair_box[0]
    chair_height = chair_box[3] - chair_box[1]

    x_margin = chair_width * 1.5  
    y_margin = chair_height * 1.0  

    is_within_x = (chair_box[0] - x_margin) <= person_center_x <= (chair_box[2] + x_margin)
    is_within_y = (chair_box[1] - y_margin) <= person_center_y <= (chair_box[3] + y_margin)

    return is_within_x and is_within_y


def process_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, None, None, None

    results = model(img)[0]

    person_boxes = []
    chair_boxes = []

    for result in results.boxes:
        cls = int(result.cls[0])
        label = model.names[cls]
        conf = float(result.conf[0])
        x1, y1, x2, y2 = map(int, result.xyxy[0])

        if conf < 0.3:  
            continue

        if label == "person":
            person_boxes.append([x1, y1, x2, y2])
        elif label in ["chair", "bench", "couch"]:
            chair_boxes.append([x1, y1, x2, y2])

    sitting_people = 0
    for person_box in person_boxes:
        is_sitting = False
        for chair_box in chair_boxes:
            if is_sitting_on_chair(person_box, chair_box):
                sitting_people += 1
                is_sitting = True
                break

        color = (0, 255, 0) if is_sitting else (0, 0, 255)  
        cv2.rectangle(img, (person_box[0], person_box[1]), (person_box[2], person_box[3]), color, 2)

        center_x = int((person_box[0] + person_box[2]) / 2)
        center_y = int((person_box[1] + person_box[3]) / 2)
        cv2.circle(img, (center_x, center_y), 5, (255, 0, 255), -1)

    for box in chair_boxes:
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 165, 0), 2)  

    cv2.putText(img, f"Oturanlar: {sitting_people}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(img, f"Toplam Kisi: {len(person_boxes)}", (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.putText(img, f"Sandalyeler: {len(chair_boxes)}", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

    output_path = f"static/results/{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    os.makedirs("static/results", exist_ok=True)
    cv2.imwrite(output_path, img)

    return sitting_people, len(person_boxes), len(chair_boxes), output_path


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        db_session = Session()
        user = db_session.query(User).filter_by(username=username).first()

        if user and user.check_password(password):
            session['user_id'] = user.id
            session['is_admin'] = user.is_admin
            flash('Başarıyla giriş yaptınız!', 'success')
            db_session.close()
            return redirect(url_for('index'))

        db_session.close()
        flash('Geçersiz kullanıcı adı veya şifre!', 'danger')
    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')

        db_session = Session()
        if db_session.query(User).filter_by(username=username).first():
            flash('Bu kullanıcı adı zaten kullanılıyor!', 'danger')
            db_session.close()
            return redirect(url_for('register'))

        user = User(username=username, email=email)
        user.set_password(password)
        db_session.add(user)
        db_session.commit()
        db_session.close()

        flash('Başarıyla kayıt oldunuz! Şimdi giriş yapabilirsiniz.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    flash('Başarıyla çıkış yaptınız!', 'success')
    return redirect(url_for('login'))


@app.route('/upload', methods=['POST'])
@login_required
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    temp_path = "temp_image.jpg"
    file.save(temp_path)

    sitting_count, total_people, total_chairs, result_path = process_image(temp_path)

    if sitting_count is None:
        return jsonify({'error': 'Failed to process image'}), 500

    db_session = Session()
    record = OccupancyRecord(
        sitting_count=sitting_count,
        image_path=result_path,
        total_chairs=total_chairs,
        total_people=total_people,
        user_id=session['user_id']
    )
    db_session.add(record)
    db_session.commit()
    db_session.close()

    os.remove(temp_path)

    return jsonify({
        'sitting_count': sitting_count,
        'total_people': total_people,
        'total_chairs': total_chairs,
        'empty_chairs': total_chairs - sitting_count,
        'result_image': result_path
    })


@app.route('/history')
@login_required
def history():
    db_session = Session()
    if session.get('is_admin'):
        records = db_session.query(OccupancyRecord).order_by(OccupancyRecord.timestamp.desc()).all()
    else:
        records = db_session.query(OccupancyRecord).filter_by(user_id=session['user_id']).order_by(
            OccupancyRecord.timestamp.desc()).all()
    db_session.close()
    return render_template('history.html', records=records)


@app.route('/admin')
@admin_required
def admin_panel():
    db_session = Session()
    users = db_session.query(User).all()
    records = db_session.query(OccupancyRecord).options(joinedload(OccupancyRecord.user)).order_by(
        OccupancyRecord.timestamp.desc()).all()

    records_data = []
    for record in records:
        records_data.append({
            'id': record.id,
            'image_path': record.image_path,
            'timestamp': record.timestamp,
            'sitting_count': record.sitting_count,
            'total_people': record.total_people,
            'total_chairs': record.total_chairs,
            'username': record.user.username
        })

    db_session.close()
    return render_template('admin.html', users=users, records=records_data)


@app.route('/admin/delete_record/<int:record_id>')
@admin_required
def delete_record(record_id):
    db_session = Session()
    record = db_session.query(OccupancyRecord).get(record_id)
    if record:
        if os.path.exists(record.image_path):
            os.remove(record.image_path)
        db_session.delete(record)
        db_session.commit()
        flash('Kayıt başarıyla silindi!', 'success')
    db_session.close()
    return redirect(url_for('admin_panel'))


@app.route('/admin/toggle_admin/<int:user_id>')
@admin_required
def toggle_admin(user_id):
    db_session = Session()
    user = db_session.query(User).get(user_id)
    if user and user.id != session['user_id']:  
        user.is_admin = not user.is_admin
        db_session.commit()
        flash(f"{user.username} için admin yetkisi {'verildi' if user.is_admin else 'alındı'}!", 'success')
    db_session.close()
    return redirect(url_for('admin_panel'))


@app.route('/analyze_image', methods=['POST'])
@login_required
def analyze_image():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'Görüntü dosyası bulunamadı'}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Dosya seçilmedi'}), 400

        temp_path = os.path.join('static', 'temp', secure_filename(file.filename))
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        file.save(temp_path)

        frame = cv2.imread(temp_path)
        if frame is None:
            return jsonify({'error': 'Görüntü okunamadı'}), 400


        results = model(frame)[0]

        person_boxes = []
        chair_boxes = []

        for result in results.boxes:
            cls = int(result.cls[0])
            label = model.names[cls]
            conf = float(result.conf[0])

            if conf < 0.3:
                continue

            x1, y1, x2, y2 = map(int, result.xyxy[0])

            if label == "person":
                person_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            elif label in ["chair", "bench", "couch"]:
                chair_boxes.append([x1, y1, x2, y2])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)

        sitting_people = 0
        for person_box in person_boxes:
            for chair_box in chair_boxes:
                if is_sitting_on_chair(person_box, chair_box):
                    sitting_people += 1
                    break

        cv2.putText(frame, f"Oturanlar: {sitting_people}", (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Toplam Kisi: {len(person_boxes)}", (30, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Sandalyeler: {len(chair_boxes)}", (30, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

        output_path = os.path.join('static', 'results',
                                   datetime.now().strftime('%Y%m%d_%H%M%S_') + secure_filename(file.filename))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, frame)

        os.remove(temp_path)

        db_session = Session()
        record = OccupancyRecord(
            sitting_count=sitting_people,
            image_path=output_path,
            total_chairs=len(chair_boxes),
            total_people=len(person_boxes),
            user_id=session.get('user_id')
        )
        db_session.add(record)
        db_session.commit()
        db_session.close()

        return jsonify({
            'status': 'success',
            'image_url': '/' + output_path.replace('\\', '/'),
            'sitting_count': sitting_people,
            'total_people': len(person_boxes),
            'total_chairs': len(chair_boxes),
            'empty_chairs': len(chair_boxes) - sitting_people,
            'timestamp': datetime.now().isoformat()
        })

    except Exception as e:
        print(f"Görüntü analiz hatası: {str(e)}")
        return jsonify({'error': 'Görüntü analizi sırasında bir hata oluştu'}), 500


@app.route('/save_settings', methods=['POST'])
@admin_required
def save_settings():
    try:
        settings = request.get_json()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/export_data')
@admin_required
def export_data():
    try:
        output = StringIO()
        writer = csv.writer(output)
        writer.writerow(['Tarih', 'Oturan Kişi', 'Toplam Kişi', 'Toplam Sandalye', 'Boş Sandalye'])

        db_session = Session()
        records = db_session.query(OccupancyRecord).order_by(OccupancyRecord.timestamp.desc()).all()

        for record in records:
            writer.writerow([
                record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                record.sitting_count,
                record.total_people,
                record.total_chairs,
                record.total_chairs - record.sitting_count
            ])

        db_session.close()

        output.seek(0)
        return Response(
            output,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment;filename=doluluk_raporu_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'}
        )

    except Exception as e:
        print(f"Veri dışa aktarma hatası: {str(e)}")
        return "Veriler dışa aktarılırken bir hata oluştu", 500


if __name__ == '__main__':
    create_admin()  
    camera = Camera() 
    app.run(debug=True, threaded=True) 
