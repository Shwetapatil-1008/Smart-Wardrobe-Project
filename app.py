
from flask import *
import mysql.connector
import re
import joblib
import hashlib
import smtplib
import requests
import os
from datetime import datetime
# ______for login_____________
from werkzeug.security import generate_password_hash, check_password_hash
# ________for files uploading___________
from werkzeug.utils import secure_filename
import os
import shutil

 # ____for contact page mails_____
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
# ____for continew with google____________
# from google.oauth2 import id_token
# from google.auth.transport import requests
import logging
logging.basicConfig(level=logging.INFO)

# ________For Model Development_________
import cv2
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import webcolors

color_dict = webcolors.CSS3_NAMES_TO_HEX  # Updated way to access it

import tensorflow as tf
from sklearn.cluster import KMeans

# ______Skin Tone Recognition________
from skimage import io


import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ GPU is enabled")
    except RuntimeError as e:
        print(e)



# Load the model

model = tf.keras.models.load_model("FinalFab_fix_classifier.h5", compile=False)
print(" Model loaded successfully!")


app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/clothes/'
app.config['UPLOAD_FOLDER'] = 'static/uploads/clothes'  

app.secret_key = 'xyz'

def create_connection():
    return mysql.connector.connect(
        host='localhost',
        user='root',
        password='mysql',
        database='SW'
    )

# _________For OpenWeather API_____________

api_key = os.getenv('OPENWEATHER_API_KEY')
print(f"API Key: {api_key}")  # This will print to the console

api_key = "XYZ"

# lat = 16.7
# lon = 74.2167
# url = f"http://api.openweathermap.org/data/2.5/onecall?lat={lat}&lon={lon}&appid={api_key}&units=metric"

# response = requests.get(url)
# print(f"Status Code: {response.status_code}")
# print(f"Response Body: {response.text}")

# ______function for Handle Method Override:_______________

@app.before_request
def method_override():
    if request.method == 'POST' and '_method' in request.form:
        request.method = request.form['_method']

# ___________IndexPage______________

@app.route('/')
def index():
    return render_template('index.html')


# __________Login,Logout & Register______________


@app.route('/login/', methods=['GET', 'POST'])
def login():
    msg = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']

        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        account = cursor.fetchone()

        if account:
            if check_password_hash(account['password'], password):
                session['loggedin'] = True
                session['email'] = account['email']
                session['id'] = account['id']  # Make sure to set the user ID in session
                return redirect(url_for("closet"))
            else:
                msg = 'Incorrect email/password!'
        else:
            msg = 'Account not found!'

        cursor.close()
        connection.close()
    elif request.method == 'POST':
        msg = 'Please fill out the form!'

    return render_template('login.html', msg=msg)



@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))



@app.route('/login/register', methods=['GET', 'POST'])
def register():
    msg = ''
    if request.method == 'POST' and 'email' in request.form and 'password' in request.form:
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password)

        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
        account = cursor.fetchone()

        if account:
            msg = 'Account already exists!'
        elif not re.match(r'[^@]+@[^@]+\.[^@]+', email):
            msg = 'Invalid email address!'
        elif not email or not password:
            msg = 'Please fill out the form!'
        else:
            cursor.execute('INSERT INTO users (email, password) VALUES (%s, %s)', (email, hashed_password))
            connection.commit()
            msg = "You have successfully registered! Login now."

        cursor.close()
        connection.close()
    elif request.method == 'POST':
        msg = 'Please fill out the form!'

    return render_template('register.html', msg=msg)



# _______________Main PAge #Home Page___________

@app.route('/login/closet', methods=['GET'])
def closet():
    if 'loggedin' in session:
        user_id = session['id']
        connection = create_connection()

        # Use dictionary cursor to get data as a dictionary
        cursor = connection.cursor(dictionary=True)
        cursor.execute('SELECT id, name FROM sw.closet_new WHERE user_id = %s', (user_id,))
        closets = cursor.fetchall()

        cursor.execute("SELECT * FROM sw.closet_new;")
        data = cursor.fetchall()

        cursor.execute('SELECT * FROM sw.users WHERE id = %s', (user_id,))
        hello = cursor.fetchall()

        cursor.close()
        connection.close()

        return render_template('closet.html', closets=closets, data=data, hello=hello)  # Return closets as JSON
    return redirect(url_for('login'))  # Redirect to login if not logged in



# ____________user dashboard(work remaining)__________

# @app.route('/login/profile')
# def profile():
#     if 'loggedin' in session:
#         connection = create_connection()
#         cursor = connection.cursor(dictionary=True)
#         cursor.execute('SELECT * FROM users WHERE id = %s', (session['id'],))
#         account = cursor.fetchone()
#         cursor.close()
#         connection.close()
#         return render_template('profile.html', account=account)
#     return redirect(url_for('login'))




# ____________________View Closet_________

@app.route('/login/see_closet', strict_slashes=False)
def see_closet():
    closet_id = request.args.get('closet_id')

    if not closet_id or not closet_id.isdigit():
        app.logger.error('Closet ID is missing or invalid.')
        return redirect(url_for('closet'))

    connection = create_connection()
    cursor = connection.cursor(dictionary=True)

    try:
        # Fetch clothes linked to the given closet_id
        cursor.execute('''
            SELECT c.* FROM sw.clothes_new c
            JOIN sw.clothes_closet cc ON c.id = cc.clothes_id
            WHERE cc.closet_id = %s
        ''', (closet_id,))
        clothes = cursor.fetchall()

        # Fetch all outfits
        cursor.execute('SELECT * FROM sw.outfits')
        outfits = cursor.fetchall()

        # Combine clothes with outfits
        for cloth in clothes:
            cloth['outfit_dates'] = []  # Initialize the list for (date, outfit_name) pairs

            for outfit in outfits:
                if cloth['image_url'] == outfit['outfit_image']:  # Adjust this logic as needed
                    cloth['outfit_dates'].append((outfit['date'], outfit['outfit_name']))

            if cloth['outfit_dates']:
                cloth['outfit_dates'] = [
                    (datetime.strptime(date, '%Y-%m-%d') if isinstance(date, str) else date, outfit_name)
                    for date, outfit_name in cloth['outfit_dates']
                ]
                cloth['recent_date'] = max(date for date, _ in cloth['outfit_dates']).strftime('%Y-%m-%d')
            else:
                cloth['recent_date'] = None

        return render_template('get_clothes.html', clothes=clothes, closet_id=closet_id)

    except Exception as e:
        app.logger.error(f'Error retrieving clothes for closet_id {closet_id}: {e}')
        return redirect(url_for('closet'))

    finally:
        cursor.close()
        connection.close()



# _________Clothes Management____________


@app.route("/login/delete_cloth/<int:clothes_id>", methods=["POST"])
def delete_cloth(clothes_id):
    conn = create_connection()
    cursor = conn.cursor()

    # Fetch the closet_id first
    cursor.execute("SELECT closet_id FROM clothes_closet WHERE clothes_id = %s", (clothes_id,))
    closet_info = cursor.fetchone()

    if not closet_info:
        cursor.close()
        conn.close()
        return jsonify({"error": "Clothing item not found"}), 404

    closet_id = closet_info[0]  # Extract closet_id

    # Close previous query results before executing new ones
    cursor.fetchall()  # This discards any unread results (important fix)

    # Delete from clothes_closet first
    cursor.execute("DELETE FROM clothes_closet WHERE clothes_id = %s", (clothes_id,))
    conn.commit()  # Commit after deleting from clothes_closet

    # Delete from clothes_new
    cursor.execute("DELETE FROM clothes_new WHERE id = %s", (clothes_id,))
    conn.commit()  # Commit the final deletion

    cursor.close()
    conn.close()

    app.logger.info(f'Successfully deleted cloth with id {clothes_id} from closet {closet_id}.')
    
    return redirect(url_for('see_closet', closet_id=closet_id))





@app.route("/login/edit_cloth/<int:clothes_id>", methods=["POST", "PUT"])
def edit_cloth(clothes_id):
    data = request.form  # Use form data instead of JSON

    conn = create_connection()
    cursor = conn.cursor(dictionary=True)

    # Check if the clothes exist
    cursor.execute("SELECT * FROM clothes_new WHERE id = %s", (clothes_id,))
    clothes = cursor.fetchone()

    if not clothes:
        cursor.close()
        conn.close()
        return jsonify({"error": "Clothing item not found"}), 404

    # Update the clothes_new table
    update_query = """
        UPDATE clothes_new 
        SET season = %s, occasions = %s, color = %s, brand = %s, 
            material = %s, category = %s, purchase_info = %s
        WHERE id = %s
    """
    cursor.execute(update_query, (
        data.get("season", clothes["season"]),
        data.get("occasions", clothes["occasions"]),
        data.get("color", clothes["color"]),
        data.get("brand", clothes["brand"]),
        data.get("material", clothes["material"]),
        data.get("category", clothes["category"]),
        data.get("purchase_info", clothes["purchase_info"]),
        clothes_id
    ))

    conn.commit()
    cursor.close()
    conn.close()

    flash('Cloth details updated successfully!', 'success')
    return redirect(url_for('closet', closet_id=clothes.get("closet_id", None)))




# __________________Closet Management____________


@app.route('/login/add_closet/', methods=['GET', 'POST'])
def add_closet():
    if request.method == 'POST':
        if 'loggedin' in session:
            data = request.get_json()
            closet_name = data.get('name')

            connection = create_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT COUNT(*) FROM sw.closet_new WHERE user_id = %s AND name = %s', (session['id'], closet_name))
            count = cursor.fetchone()[0]

            if count > 0:
                cursor.close()
                connection.close()
                return jsonify(success=False, message="Please enter a unique closet name.")
            
            cursor.execute('INSERT INTO sw.closet_new (user_id, name) VALUES (%s, %s)', (session['id'], closet_name))
            connection.commit()
            cursor.close()
            connection.close()
            return jsonify(success=True, redirect=url_for('closet'))

    return render_template("add_closet.html")


@app.route('/login/manage_closets', methods=['GET'])
def manage_closets():
    if 'loggedin' in session:
        user_id = session['id']
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute('SELECT id, name FROM sw.closet_new WHERE user_id = %s', (user_id,))
        closets = cursor.fetchall()
        cursor.close()
        connection.close()
        return render_template('manage_closets.html', closets=closets)
    return redirect(url_for('login'))



@app.route('/login/edit_closet/<int:closet_id>', methods=['GET', 'POST'])
def edit_closet(closet_id):
    if 'loggedin' in session:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        
        if request.method == 'POST':
            new_name = request.form.get('name')
            cursor.execute('UPDATE sw.closet_new SET name = %s WHERE id = %s', (new_name, closet_id))
            connection.commit()
            cursor.close()
            connection.close()
            return redirect(url_for('closet'))

        cursor.execute('SELECT * FROM sw.closet_new WHERE id = %s', (closet_id,))
        closet = cursor.fetchone()
        cursor.close()
        connection.close()
        return render_template('edit_closet.html', closet=closet)



@app.route('/login/delete_closet/<int:closet_id>', methods=['POST'])
def delete_closet(closet_id):
    if 'loggedin' in session:
        flash("Closet Has Been Deleted Successfully")
        connection = create_connection()
        cursor = connection.cursor()
        cursor.execute('DELETE FROM sw.closet_new WHERE id = %s', (closet_id,))
        connection.commit()
        cursor.close()
        connection.close()
        return redirect(url_for('closet'))
    return redirect(url_for('login'))



# ____________Clothes Management____________


fabric_season_mapping = {
    "Cotton": ["Summer", "Spring", 'Early Fall'], # Cotton is breathable and works well in mild winters too
    "Denim": ["Summer", "Fall", "Late Fall" "Winter", "Spring", "Early Fall", "Mild Winter"],  # Denim is heavier and better suited for cooler weather
    "Silk": ["Spring", "Summer", "Early Fall"],  # Silk is lightweight and comfortable in warm weather
    "Wool": ["Winter", "Fall", "Late Fall", "Early Spring"]  # Wool is warm and best for cold weather
}

# Class labels (Ensure these match your training labels)
class_labels = ['Cotton', 'Denim', 'Silk', 'Wool']



def load_segmentation_model():
    """Load the DeepLabV3 model for background removal."""
    model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.eval()
    return model

segmentation_model = load_segmentation_model()


def load_model():
    """Load the DeepLabV3 model for background removal."""
    model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.eval()
    return model

def refine_mask(mask):
    """Apply morphological operations to refine the segmentation mask."""
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask

def remove_background(img_path, model):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)['out'][0]

    mask = output.argmax(0).byte().numpy()
    clothing_mask = (mask == 0).astype(np.uint8)  # Assume clothing pixels are labeled as '0'
    clothing_mask = cv2.resize(clothing_mask, (img.shape[1], img.shape[0]))
    clothing_mask = refine_mask(clothing_mask)  # Apply mask refinement
    result = img * clothing_mask[:, :, np.newaxis]

    # Show refined mask
    return result, clothing_mask


def filter_colors(pixels):
    """Remove near-white and gray pixels from color detection."""
    return [p for p in pixels if np.std(p) > 20 and np.mean(p) < 230]  # Avoid gray/white tones



def get_dominant_color(image, mask):
    """Finds the dominant color from an image using KMeans clustering."""
    pixels = image.reshape(-1, 3)
    mask_flat = mask.flatten()

    valid_pixels = pixels[mask_flat > 0]
    valid_pixels = filter_colors(valid_pixels)  # Exclude background-like colors

    if len(valid_pixels) == 0:
        return (0, 0, 0)  # Fallback to black if no valid pixels detected

    # Set the number of clusters to the minimum of 3 or the number of valid pixels
    n_clusters = min(3, len(valid_pixels))

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    kmeans.fit(valid_pixels)

    # Get the dominant color, prioritizing vibrant colors
    dominant_color = sorted(kmeans.cluster_centers_, key=lambda x: np.linalg.norm(x - [128, 128, 128]), reverse=True)[0]

    return tuple(map(int, dominant_color))



def closest_color(requested_color):
    """Find the closest named CSS3 color."""
    def euclidean_distance(rgb1, rgb2):
        return sum((a - b) ** 2 for a, b in zip(rgb1, rgb2))

    css3_colors = {name: webcolors.hex_to_rgb(hex_code) for name, hex_code in webcolors.CSS3_NAMES_TO_HEX.items()}
    closest_name = min(css3_colors, key=lambda name: euclidean_distance(css3_colors[name], requested_color))

    return closest_name


@app.route('/predict_fabric_color', methods=['POST'])
def predict_fabric_color():
    if 'image' not in request.files:
        return jsonify(success=False, message="No image uploaded")

    image_file = request.files['image']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(image_file.filename))
    image_file.save(file_path)

    try:
        # Load and preprocess the image
        img = load_img(file_path, target_size=(224, 224))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict Fabric
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions)
        detected_fabric = class_labels[predicted_class]

        # Detect Color
        processed_image, mask = remove_background(file_path, segmentation_model)
        dominant_color = get_dominant_color(processed_image, mask)
        detected_color = closest_color(dominant_color)
        # Get the suitable seasons for the predicted fabric
        suitable_seasons = fabric_season_mapping.get(class_labels[predicted_class], ["Unknown"])
        print(f"Suitable Seasons for {class_labels[predicted_class]}: {', '.join(suitable_seasons)}")

        return jsonify(success=True, fabric=detected_fabric, color=detected_color,seasons=suitable_seasons)

    except Exception as e:
        app.logger.error(f"Prediction Error: {e}")
        return jsonify(success=False, message="Error predicting fabric and color")





@app.route('/login/add_clothes', methods=['GET', 'POST'])
def add_clothes():
    if 'loggedin' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        user_id = session['id']
        occasions = request.form.get('occasions')
        brand = request.form.get('brand')
        category = request.form.get('category')
        purchase_info = request.form.get('purchase_info')

        image_url, detected_color, detected_fabric, suitable_seasons = None, None, None, []

        if 'image_url' in request.files and request.files['image_url'].filename:
            image_file = request.files['image_url']
            image_url = secure_filename(image_file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], image_url)
            image_file.save(file_path)

            img = load_img(file_path, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            predictions = model.predict(img_array)
            detected_fabric = class_labels[np.argmax(predictions)]

            processed_image, mask = remove_background(file_path, segmentation_model)
            dominant_color = get_dominant_color(processed_image, mask)
            detected_color = closest_color(dominant_color)

            suitable_seasons = fabric_season_mapping.get(detected_fabric, ["Unknown"])

        try:
            connection = create_connection()
            cursor = connection.cursor()

            # Insert clothing item ONCE in clothes_new (no closet_id here)
            cursor.execute(
                'INSERT INTO clothes_new (user_id, season, occasions, color, brand, material, category, purchase_info, image_url) '
                'VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)',
                (user_id, ', '.join(suitable_seasons), occasions, detected_color, brand, detected_fabric, category, purchase_info, image_url)
            )
            connection.commit()
            clothes_id = cursor.lastrowid  # Get the newly inserted clothing item ID

            # Link the clothing item to multiple closets
            for season_name in suitable_seasons:
                cursor.execute('SELECT id FROM sw.closet_new WHERE user_id = %s AND name = %s', (user_id, season_name))
                closet = cursor.fetchone()

                if not closet:
                    cursor.execute('INSERT INTO sw.closet_new (user_id, name) VALUES (%s, %s)', (user_id, season_name))
                    connection.commit()
                    closet_id = cursor.lastrowid
                else:
                    closet_id = closet[0]

                cursor.execute(
                    'INSERT INTO clothes_closet (clothes_id, closet_id) VALUES (%s, %s)',
                    (clothes_id, closet_id)
                )

            connection.commit()
            return jsonify(success=True, redirect=url_for('closet'))

        except Exception as e:
            app.logger.error(f"Error inserting data: {e}")
            return jsonify(success=False, message="Error inserting data.")
        finally:
            if cursor:
                cursor.close()
            if connection:
                connection.close()


    # GET Request: Display Form
    connection = create_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT id, name FROM sw.closet_new WHERE user_id = %s', (session['id'],))
    closets = cursor.fetchall()
    cursor.close()
    connection.close()

    return render_template("add_clothes_byModel.html", closets=closets)

    return redirect(url_for('login'))



@app.route('/login/get_clothes', methods=['GET'])
def get_clothes():
    if 'loggedin' in session:
        closet_id = request.args.get('closet_id')
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)
        cursor.execute('SELECT * FROM sw.clothes_closet WHERE closet_id = %s', (closet_id,))
        clothes = cursor.fetchall()
        cursor.close()
        connection.close()
        return render_template('get_clothes.html', clothes=clothes)
    return redirect(url_for('login'))



# ________Weather based Recommendation________


@app.route('/get_weather_new', methods=['GET'])
def get_weather_new():
    city = request.args.get('city')
    api_key = os.getenv('OPENWEATHER_API_KEY')
    
    if not api_key:
        return jsonify({'error': 'API key for OpenWeather is not configured'}), 500

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        temperature = data['main']['temp']
        weather_description = data['weather'][0]['description']
        return jsonify({'temperature': temperature, 'description': weather_description})
    else:
        error_message = response.json().get('message', 'Unknown error')
        return jsonify({'error': f'Failed to fetch weather data: {error_message}'}), 400


@app.route('/recommend_outfits_new', methods=['GET', 'POST'])
def recommend_outfits_new():
    if 'loggedin' not in session:
        return redirect('/login_new')

    outfits = []
    current_weather = None
    user_id = session['id']
    city = None

    if request.method == 'POST':
        data = request.form
        city = data.get('city')
        if not city:
            return render_template('recommend_outfits_new.html', error='City not provided')

        api_key = os.getenv('OPENWEATHER_API_KEY')
        if not api_key:
            return render_template('recommend_outfits_new.html', error='API key is not configured')

        current_url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        current_response = requests.get(current_url)
        
        if current_response.status_code == 200:
            current_weather_data = current_response.json()
            lat = current_weather_data['coord']['lat']
            lon = current_weather_data['coord']['lon']
            temperature = current_weather_data['main']['temp']
            weather_description = current_weather_data['weather'][0]['description']
            current_weather = f"{temperature}°C, {weather_description}"
            # print(f"Weather Description: {weather_description}")  # Log weather description
        else:
            return render_template('recommend_outfits_new.html', error='Failed to fetch current weather data')

        conn = create_connection()
        cursor = conn.cursor(dictionary=True)
        try:
            cursor.execute("""
                SELECT c.occasions, c.category, c.image_url, c.season, c.material
                FROM sw.clothes_new c
                JOIN sw.clothes_closet cc ON c.id = cc.clothes_id
                JOIN sw.closet_new cl ON cc.closet_id = cl.id
                WHERE cl.user_id = %s

            """, (user_id,))
            clothes = cursor.fetchall()
            # print(f"Clothes Retrieved: {clothes}")  # Log retrieved clothes

            temperature = current_weather_data['main']['temp']
            humidity = current_weather_data['main']['humidity']
            wind_speed = current_weather_data['wind']['speed']

            # Initialize the outfits list and a set to track unique outfits
            outfits = []
            recommended_outfit_ids = set()  # Use a set to track unique outfits
            for item in clothes:
                item_season = item.get('season', '').lower()  # Default to empty string if None
                item_material = item.get('material', '').lower()  

                outfit_id = item['image_url']  # Unique identifier for outfits

                # Ensure we only add unique outfits
                if outfit_id in recommended_outfit_ids:
                    continue  # Skip if already added

                # Temperature-based recommendations
                if temperature < 10:
                    if any(season in item_season for season in ["winter", "cool", "early spring", "mild winter", "late fall"]):
                        outfits.append(item)
                        recommended_outfit_ids.add(outfit_id)
                        continue  

                elif 10 <= temperature < 25:
                    if any(season in item_season for season in ["spring", "cool", "fall", "early fall", "mild winter"]):
                        outfits.append(item)
                        recommended_outfit_ids.add(outfit_id)
                        continue  

                elif 25 <= temperature < 30:
                    if any(season in item_season for season in ["summer"]):
                        outfits.append(item)
                        recommended_outfit_ids.add(outfit_id)
                        continue  

                elif temperature >= 30:
                    if any(season in item_season for season in ["summer"]):
                        outfits.append(item)
                        recommended_outfit_ids.add(outfit_id)
                        continue  

                # Material-based recommendations (optional)
                if temperature < 0:  # For very cold temperatures
                    if item_material in ["wool", "denim"]: 
                        outfits.append(item)
                        recommended_outfit_ids.add(outfit_id)
                        continue  

                elif temperature >= 25:  # For warm temperatures
                    if item_material in ["cotton", "linen", "silk"]:  # Breathable fabrics for hot weather
                        outfits.append(item)
                        recommended_outfit_ids.add(outfit_id)
                        continue  

                # Humidity-based recommendations (for high humidity, prefer breathable fabrics)
                # if humidity > 70 and item_material in ["cotton", "linen"]:
                #     if outfit_id not in recommended_outfit_ids:
                #         outfits.append(item)  
                #         recommended_outfit_ids.add(outfit_id)

                # # Wind-based recommendations (for high wind speeds, recommend protective clothing)
                # if wind_speed > 25 and any(season in item_season for season in ["windy", "cool"]):
                #     if outfit_id not in recommended_outfit_ids:
                #         outfits.append(item)  
                #         recommended_outfit_ids.add(outfit_id)

                # # Precipitation-based recommendations (recommend rainy outfits if rain is detected)
                # if 'rain' in weather_description.lower() and "rainy" in item_season:
                #     if outfit_id not in recommended_outfit_ids:
                #         outfits.append(item)  
                #         recommended_outfit_ids.add(outfit_id)



        finally:
            cursor.close()
            conn.close()

    # print(f"Final Outfits List: {outfits}")  # Log final outfits list
    return render_template('recommend_outfits_new.html', outfits=outfits, current_weather=current_weather, city=city)



 # ___________Steps__________

# @app.route('/step1')
# def step1():
#     return render_template('step1.html')



# ____________OOTD ________________




@app.route('/login/cal')
def cal():
    return render_template('event_calender.html')



@app.route('/login/ootd', methods=['GET', 'POST'])
def ootd():

    import shutil
    if 'loggedin' not in session:
        return redirect(url_for('login'))  # Redirect to login if not logged in

    user_id = session['id']
    outfits = []  # Initialize outfits to an empty list
    clothes_data = []  # Initialize clothes_data as a list
    outfit_counts = {}  # To keep track of outfit counts
    most_worn_outfit = {'name': None, 'count': 0, 'image_url': None}  # To track the most worn outfit

    try:
        connection = create_connection()
        cursor = connection.cursor(dictionary=True)

        # Fetch closets for the logged-in user
        cursor.execute('SELECT id, name FROM sw.closet_new WHERE user_id = %s', (user_id,))
        closets = cursor.fetchall()

      

        # Fetch clothes from each closet
        clothes_by_closet = {}
        for closet in closets:
            cursor.execute('''
                SELECT c.* FROM sw.clothes_new c
                JOIN sw.clothes_closet cc ON c.id = cc.clothes_id
                WHERE cc.closet_id = %s
            ''', (closet['id'],))
            clothes_by_closet[closet['id']] = cursor.fetchall()

        # Handle form submission for adding a new outfit
        if request.method == 'POST':
            outfit_name = request.form['outfitName']
            selected_clothes = request.form.getlist('selected_clothes[]')  # Get selected clothes from the form
            date = request.form.get('date') or datetime.now().date()  # Use current date if not provided

            # Validate input
            if selected_clothes:
                print("Outfit Name:", outfit_name)
                print("Selected Clothes:", selected_clothes)
                print("Date:", date)

                try:
                    image_paths = []
                    for cloth_id in selected_clothes:
                        # Fetch the image path for each selected cloth
                        cursor.execute('SELECT image_url FROM sw.clothes_new WHERE id = %s', (cloth_id,))
                        cloth = cursor.fetchone()
                        if cloth:
                            image_url = cloth['image_url']
                            image_paths.append(image_url)

                            # Copy the image to the wornClothes folder
                            source_path = os.path.join('static/uploads/clothes', image_url)
                            destination_path = os.path.join('static/uploads/wornClothes', image_url)
                            shutil.copy(source_path, destination_path)

                    # Save the outfit to the database
                    cursor.execute('INSERT INTO sw.outfits (user_id, outfit_name, outfit_image, date) VALUES (%s, %s, %s, %s)',
                                   (user_id, outfit_name, ','.join(image_paths), date))
                    connection.commit()  # Commit the transaction

                    # Update outfit counts
                    outfit_counts[outfit_name] = outfit_counts.get(outfit_name, 0) + 1

                    # Update most worn outfit if necessary
                    if outfit_counts[outfit_name] > most_worn_outfit['count']:
                        most_worn_outfit = {
                            'name': outfit_name,
                            'count': outfit_counts[outfit_name],
                            'image_url': ','.join(image_paths)  # Use the joined image paths
                        }
                    flash('Outfit added successfully!', 'success')

                except Exception as e:
                    flash(f'An error occurred while processing your request: {str(e)}', 'error')
                    app.logger.error(f'Error during outfit insertion: {str(e)}')  # Log the error
            else:
                flash('Please provide both outfit name and select clothes.', 'error')

            return redirect(url_for('ootd'))

        # Fetch outfits for the logged-in user
        cursor.execute('SELECT * FROM sw.outfits WHERE user_id = %s', (user_id,))
        outfits = cursor.fetchall()

        # Update most worn outfit based on fetched outfits
        for outfit in outfits:
            outfit_image = outfit['outfit_image']
            outfit_counts[outfit_image] = outfit_counts.get(outfit_image, 0) + 1
            if outfit_counts[outfit_image] > most_worn_outfit['count']:
                most_worn_outfit = {
                    'name': outfit_image,
                    'count': outfit_counts[outfit_image],
                    'image_url': outfit['outfit_image'].split(',')[0]  # Use the first image for display
                }

    except Exception as e:
        flash(f'An error occurred while processing your request: {str(e)}', 'error')
        app.logger.error(f'General error: {str(e)}')  # Log the general error

    app.logger.info(f'Most Worn Outfit: {most_worn_outfit}')
    app.logger.info(f'Outfit Counts: {outfit_counts}')
    return render_template('ootd.html', closets=closets, clothes_by_closet=clothes_by_closet, outfits=outfits, most_worn_outfit=most_worn_outfit, clothes=clothes_data )



# @app.route('/delete_ootd', methods=['POST'])
# def delete_ootd():
#     outfit_id = request.form['outfit_id']
#     try:
#         with create_connection() as connection:
#             with connection.cursor() as cursor:
#                 cursor.execute('DELETE FROM sw.outfits WHERE id = %s', (outfit_id,))
#                 connection.commit()
#         return {'success': True}
#     except Exception as e:
#         return {'success': False, 'error': str(e)}



# @app.route('/edit_ootd', methods=['POST'])
# def edit_ootd():
#     outfit_id = request.form['outfit_id']
#     outfit_name = request.form['outfitName']
#     date = request.form['date']
#     # Handle image update if necessary
#     try:
#         with create_connection() as connection:
#             with connection.cursor() as cursor:
#                 cursor.execute('UPDATE sw.outfits SET outfit_name = %s, date = %s WHERE id = %s',
#                                (outfit_name, date, outfit_id))
#                 connection.commit()
#         return {'success': True}
#     except Exception as e:
#         return {'success': False, 'error': str(e)}



# ____________color_palette____________


@app.route('/color_palette')
def color_palette():
    return render_template('color_palette.html')  # Make sure to save the HTML file as color_palette.html


# _____________Design your Own Outfits_______



# Set the correct upload folder
UPLOAD_FOLDER2 = 'static/uploads/design_ot'
os.makedirs(UPLOAD_FOLDER2, exist_ok=True)  # Ensure folder exists
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2



@app.route('/design_outfits')
def design_outfits():
    return render_template('design_outfits.html')



@app.route('/save_outfits', methods=['POST'])
def save_outfits():
    if 'outfit_image' not in request.files:
        return jsonify({'error': 'No image file'}), 400
    
    file = request.files['outfit_image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Ensure unique filename
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER2'], filename)

    # Rename file if it already exists
    counter = 1
    while os.path.exists(filepath):
        filename = f"{os.path.splitext(file.filename)[0]}_{counter}{os.path.splitext(file.filename)[1]}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER2'], filename)
        counter += 1

    file.save(filepath)  # Save to the correct folder
    
    # Save file path and timestamp to database
    conn = create_connection()
    cursor = conn.cursor(dictionary=True)
    created_at = datetime.now()
    cursor.execute("INSERT INTO design_outfits (image_path, created_at) VALUES (%s, %s)", (filepath, created_at))
    conn.commit()
    cursor.close()
    conn.close()
    
    return jsonify({'message': 'Outfit saved successfully!', 'image_url': filepath ,'redirect_url': '/design_outfits'  })



@app.route("/see_design_outfits")
def see_design_outfits():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, image_path, created_at FROM design_outfits")
    raw_outfits = cursor.fetchall()
    cursor.close()
    
    outfits = [
        {'id': row[0], 'image_path': row[1], 'created_at': row[2]}
        for row in raw_outfits
    ]

    return render_template("see_design_outfits.html", outfits=outfits)



# @app.route('/edit_design_outfits/<int:outfit_id>', methods=['POST','GET'])
# def edit_design_outfits(outfit_id):
#     new_image = request.files.get('outfit_image')

#     # Ensure the database connection is established
#     conn = create_connection()
#     if conn is None:
#         return "Database connection error", 500

#     cursor = conn.cursor()
    
#     try:
#         if new_image:
#             filename = secure_filename(new_image.filename)
#             filepath = os.path.join(app.config['UPLOAD_FOLDER2'], filename)
#             new_image.save(filepath)

#             cursor.execute("UPDATE design_outfits SET image_path=%s WHERE id=%s", (filepath, outfit_id))
        
#         conn.commit()
#     except Exception as e:
#         conn.rollback()
#         return f"Error updating outfit: {str(e)}", 500
#     finally:
#         cursor.close()
#         conn.close()
    
#     return redirect(url_for('see_design_outfits'))



@app.route('/delete_design_outfits/<int:outfit_id>', methods=['POST'])
def delete_design_outfits(outfit_id):
    conn = create_connection()
    if conn is None:
        return "Database connection error", 500

    cursor = conn.cursor()
    
    try:
        # Get file path to delete the image
        cursor.execute("SELECT image_path FROM design_outfits WHERE id = %s", (outfit_id,))
        result = cursor.fetchone()

        if result:  # Ensure there is a result
            image_path = result[0]  # Access tuple by index
            if image_path and os.path.exists(image_path):
                os.remove(image_path)  # Delete image from the server

        # Delete the outfit from the database
        cursor.execute("DELETE FROM design_outfits WHERE id=%s", (outfit_id,))
        conn.commit()
        
    except Exception as e:
        conn.rollback()
        return f"Error deleting outfit: {str(e)}", 500
    finally:
        cursor.close()
        conn.close()
    
    return redirect(url_for('see_design_outfits'))



# __________________Skin Tone Recognition________


UPLOAD_FOLDER3 = 'static/uploads/skin_tone'
app.config['UPLOAD_FOLDER3'] = UPLOAD_FOLDER3
os.makedirs(UPLOAD_FOLDER3, exist_ok=True)


# Recommended colors based on skin tone
skin_tone_colors = {
    "Type I (Very Fair)": ["Best Colors: " ,"   ","Soft Pastels – Baby blue, blush pink, lavender, mint green","   ", "Cool Tones – Sky blue, icy gray, sapphire, emerald green","   ", "Jewel Tones – Ruby red, amethyst purple, royal blue","   ", "Neutral Tones – Cool beige, soft gray, taupe","   ","   ", " Colors to Avoid : ","   ", "Neon Shades – Bright orange, neon green, electric yellow","   ","Overly Pale Colors – White, beige, light yellow (may blend too much with skin)","   ", "Warm Browns & Yellows – These can create a dull look"],

    "Type II (Fair)": [" Best Colors : ","   ", "Cool Pastels – Light pink, baby blue, lavender, soft peach","   ", "Earthy Neutrals – Soft gray, taupe, cool beige, light olive","   ", "Rich & Deep Tones – Burgundy, emerald green, royal blue, deep teal","   ", "Soft Metallics – Silver, cool gold, champagne","   ","   "," Colors to Avoid : ","   ","Overly Pale Colors – Pale yellow, off-white, beige (can make skin look washed out)","   ", "Too Warm & Earthy Tones – Mustard yellow, dark brown, burnt orange (may clash)","   ", "Neon Shades – Bright orange, lime green, fluorescent pink (too overpowering)"],

    "Type III (Medium)": [" Best Colors : ","   ", "Warm & Rich Tones – Mustard yellow, burnt orange, terracotta, deep coral","   ", "Earthy & Natural Shades – Olive green, camel, warm brown, taupe","   ", "Jewel Tones – Emerald green, royal blue, deep purple, ruby red", "Classic Neutrals – Charcoal gray, navy blue, warm beige","   ","   ", " Colors to Avoid : ","   ", "Overly Pale Colors – Light beige, pale gray, pastel yellow (can wash out your skin)","   ", "Neon & Super Bright Shades – Fluorescent pink, electric blue, lime green (can be overpowering)"],

    "Type IV (Olive)": [" Best Colors : ","Earthy & Rich Tones – Terracotta, rust, caramel, deep olive, burnt orange", "Warm Neutrals – Warm beige, camel, mocha, tan, chestnut brown", "Jewel Tones – Emerald green, deep sapphire blue, rich burgundy, teal", "Soft Pastels (with warm undertones) – Peach, warm blush pink, golden yellow", " Colors to Avoid : ", "Cool-Toned Pastels – Icy blue, baby pink, pale lavender (may wash out your warmth)", "Overly Bright or Neon Colors – Neon yellow, hot pink, lime green (too harsh)"],

    "Type V (Brown)": [" Best Colors : ","   " ,"Jewel Tones – Royal blue, deep purple, emerald green, ruby red","   " , "Warm Earthy Shades – Mustard yellow, burnt orange, terracotta, warm browns","   " , "Deep & Bold Colors – Burgundy, maroon, chocolate brown, deep teal","   " , "Bright & Bold Accents – Cobalt blue, fuchsia, rich gold, warm pink","   " ,"   " , " Colors to Avoid : ","   " , "Pale Pastels – Light pink, icy blue, pale lavender (may not contrast well)","   " , "Muted or Ashy Colors – Cool gray, washed-out beige, dull olive (can make skin look dull)"],

    "Type VI (Dark Brown/Black)": [" Best Colors : " ,"   " ,"Bright & Bold Colors – Vivid orange, hot pink, bright yellow, electric blue" ," " , "Rich Earthy Tones – Deep terracotta, burnt orange, rich browns, warm reds"," " , "Jewel Tones – Emerald green, royal blue, deep purple, ruby red"," " , "Metallic Shades – Rich gold, bronze, champagne, copper","   " ,"   " , " Colors to Avoid : ","   " , "Muted Pastels – Pale pink, icy lavender, washed-out shades (can lack contrast)"," " , "Ashy & Cool Grays – Light gray, dusty mauve, muted cool tones (can appear dull against rich skin tone)"]
}


def preprocess_image(image):
    image = cv2.bilateralFilter(image, 9, 75, 75)
    image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
    return image


def detect_skin(image):
    """Detects skin using improved face-based region extraction."""
    # Load OpenCV pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))

    if len(faces) == 0:
        return None  # No face detected, avoid misclassifying clothing

    # Crop the first detected face
    x, y, w, h = faces[0]
    face_region = image[y:y + h, x:x + w]

    # Convert to HSV & YCrCb for better skin segmentation
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(face_region, cv2.COLOR_BGR2YCrCb)

    lower_hsv = np.array([0, 40, 50], dtype="uint8")
    upper_hsv = np.array([25, 255, 255], dtype="uint8")

    lower_ycrcb = np.array([0, 133, 77], dtype="uint8")
    upper_ycrcb = np.array([255, 173, 127], dtype="uint8")

    mask_hsv = cv2.inRange(hsv, lower_hsv, upper_hsv)
    mask_ycrcb = cv2.inRange(ycrcb, lower_ycrcb, upper_ycrcb)

    # Combine masks
    mask = cv2.bitwise_and(mask_hsv, mask_ycrcb)
    skin = cv2.bitwise_and(face_region, face_region, mask=mask)

    return skin


def extract_dominant_skin_color(image, k=3):
    """Extracts the dominant skin tone from the detected skin area using K-Means clustering."""
    skin = detect_skin(image)
    if skin is None:
        return None  # No skin detected

    lab = cv2.cvtColor(skin, cv2.COLOR_BGR2LAB)
    lab = lab.reshape((-1, 3))

    # Remove black pixels (non-skin areas)
    lab = lab[np.all(lab != [0, 0, 0], axis=1)]
    if len(lab) < 10:  # If too few skin pixels remain, return None
        return None

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(lab)

    dominant_color = kmeans.cluster_centers_[np.argmax(np.bincount(kmeans.labels_))]
    return tuple(map(int, dominant_color))




def classify_skin_tone(rgb):
    """Classifies skin tone based on Fitzpatrick scale."""
    if rgb is None:
        return "No Skin Detected"

    lab_color = np.uint8([[rgb]])
    rgb_color = cv2.cvtColor(lab_color, cv2.COLOR_LAB2RGB)[0][0]

    r, g, b = rgb_color
    if r > 200 and g > 180 and b > 160:
        return "Type I (Very Fair)"
    elif r > 180 and g > 160 and b > 140:
        return "Type II (Fair)"
    elif r > 160 and g > 140 and b > 120:
        return "Type III (Medium)"
    elif r > 140 and g > 120 and b > 100:
        return "Type IV (Olive)"
    elif r > 110 and g > 90 and b > 70:
        return "Type V (Brown)"
    else:
        return "Type VI (Dark Brown/Black)"



@app.route("/skin_tone")
def skin_tone():
    return render_template("skin_tone.html")



@app.route("/analyze_skin_tone", methods=["POST"])
def analyze_skin_tone():
    if "image" not in request.files:
        return jsonify({"error": "No image file"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config["UPLOAD_FOLDER3"], filename)
    file.save(filepath)

    image = io.imread(filepath)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = preprocess_image(image)

    dominant_color = extract_dominant_skin_color(image)
    
    if dominant_color is None:
        return jsonify({"error": "No skin detected. Try a clearer image with less background or clothing."}), 400

    skin_tone = classify_skin_tone(dominant_color)
    suggested_colors = skin_tone_colors.get(skin_tone, [])

    # Save results in MySQL
    conn = create_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("INSERT INTO skin_analysis (image_path, skin_tone) VALUES (%s, %s)", (filepath, skin_tone))
    conn.commit()
    cursor.close()
    conn.close()

    return jsonify({
        "skin_tone": skin_tone,
        "suggested_colors": suggested_colors,
        "image_url": filepath
    })




# ______________Detect Body Shape___________


# import mediapipe as mp

# mp_pose = mp.solutions.pose
# pose = mp_pose.Pose()
# mp_drawing = mp.solutions.drawing_utils



# def detect_body_shape(image):
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image_rgb)

#     if not results.pose_landmarks:
#         return "No body detected"

#     landmarks = results.pose_landmarks.landmark

#     # Extract key body points
#     shoulder_width = abs(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x - landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x)
#     waist_width = abs(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x - landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x)
#     hip_width = abs(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x - landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x)

#     # Determine body shape
#     if waist_width / shoulder_width < 0.75 and hip_width / shoulder_width < 0.75:
#         return "Rectangle Shape"
#     elif waist_width < shoulder_width and waist_width < hip_width:
#         return "Hourglass Shape"
#     elif shoulder_width > hip_width:
#         return "Inverted Triangle Shape"
#     elif hip_width > shoulder_width:
#         return "Pear Shape"
#     else:
#         return "Apple Shape"

# # Test with an image
# image = cv2.imread("body.jpg")
# body_shape = detect_body_shape(image)
# print("Detected Body Shape:", body_shape)



# @app.route("/analyze_body_shape", methods=["POST"])
# def analyze_body_shape():
#     if "image" not in request.files:
#         return jsonify({"error": "No image file"}), 400

#     file = request.files["image"]
#     if file.filename == "":
#         return jsonify({"error": "No selected file"}), 400

#     filename = secure_filename(file.filename)
#     filepath = os.path.join(app.config["UPLOAD_FOLDER3"], filename)
#     file.save(filepath)

#     # Load and process the image
#     image = io.imread(filepath)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     body_shape = detect_body_shape(image)  # Function to detect body shape
#     recommendations = get_recommendations(body_shape)

#     # Save to MySQL
#     conn = create_connection()
#     cursor = conn.cursor(dictionary=True)
#     cursor.execute("INSERT INTO user_body_analysis (image_path, body_shape) VALUES (%s, %s)", (filepath, body_shape))
#     conn.commit()
#     cursor.close()
#     conn.close()

#     return jsonify({
#         "body_shape": body_shape,
#         "recommended_outfits": recommendations,
#         "image_url": filepath
#     })




# @app.route('/about')
# def about():
#     return render_template('about.html')



# _______Contact Page____________

@app.route('/contact')
def contact():
    return render_template('contact.html')



# @app.route('/login/contact_php', methods=['POST'])
# def contact_php():
#     try:
#         data = request.get_json()
#         name = data['name']
#         sender_email = data['email']
#         subject = data['subject']
#         message = data['message']
        
#         receiver_email = 'spatilpatil108@gmail.com'

#         msg = MIMEMultipart()
#         msg['From'] = sender_email
#         msg['To'] = receiver_email
#         msg['Subject'] = subject
        
#         body = f"From: {name}\nEmail: {sender_email}\n\n{message}"
#         msg.attach(MIMEText(body, 'plain'))

#         server = smtplib.SMTP('smtp.example.com', 587)  # Update with your SMTP server details
#         server.starttls()
#         server.login('spatilpatil108@gmail.com', 'my_password')  # Use your SMTP credentials
#         server.sendmail(sender_email, receiver_email, msg.as_string())
#         server.quit()

#         return jsonify(success=True)
#     except Exception as e:
#         return jsonify(success=False, error=str(e))


# @app.route('/recommend_outfits_by_occasion', methods=['GET', 'POST'])
# def recommend_outfits_by_occasion():
#     if request.method == 'POST':
#         data = request.get_json()  # Get JSON data from the request body
#         occasion = data.get('occasion')  # Extract the occasion from the JSON data
#     else:
#         occasion = request.args.get('occasion')  # For GET requests

#     if not occasion:
#         return jsonify({'status': 'error', 'message': 'No occasion provided.'}), 400

#     # Normalize the occasion input
#     occasion = occasion.lower()

#     connection = create_connection()
#     cursor = connection.cursor(dictionary=True)

#     try:
#         # Fetch clothes that match the specified occasion
#         cursor.execute('SELECT * FROM clothes WHERE occasions LIKE %s', (f'%{occasion}%',))
#         clothes = cursor.fetchall()

#         # Prepare the list of recommended outfits
#         recommended_outfits = []
#         for item in clothes:
#             recommended_outfits.append({
#                 'id': item['id'],
#                 'season': item['season'],
#                 'occasions': item['occasions'],
#                 'color': item['color'],
#                 'brand': item['brand'],
#                 'material': item['material'],
#                 'category': item['category'],
#                 'purchase_info': item['purchase_info'],
#                 'image_url': item['image_url']
#             })

#         return jsonify({'status': 'success', 'outfits': recommended_outfits})

#     except Exception as e:
#         return jsonify({'status': 'error', 'message': 'An error occurred while fetching outfits.', 'details': str(e)}), 500

#     finally:
#         cursor.close()
#         connection.close()

#     # If you want to render a template for GET requests, handle it here
#     return render_template('recommend_outfits_occasion.html')




if __name__ == '__main__':
    app.run(debug=True)


