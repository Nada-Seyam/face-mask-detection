import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import tensorflow_hub as hub
import joblib
from skimage.feature import graycomatrix, graycoprops

# Path to the downloaded model files - update these to your actual paths
# Path to the downloaded model files - update these to your actual paths
MODELS_DIR = r"C:\Users\NADA\Downloads"  # Use r prefix for raw string to handle backslashes
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "best_xgb_model.pkl")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_face_mask_model.h5")
BBOX_MODEL_PATH = os.path.join(MODELS_DIR, "bbox.h5")
class MaskDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Mask Detection")
        self.root.geometry("1000x800")
        
        # Variables
        self.current_image = None
        self.models_loaded = False
        
        # Top frame for buttons
        self.top_frame = Frame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Load Image Button
        self.load_btn = Button(self.top_frame, text="Load Image", command=self.load_image, 
                               bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # Detect Button
        self.detect_btn = Button(self.top_frame, text="Detect Masks", command=self.detect_masks,
                                bg="#2196F3", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        self.detect_btn.config(state=tk.DISABLED)
        
        # Radio buttons for model selection
        self.model_var = tk.StringVar(value="cnn")
        self.model_frame = Frame(self.top_frame)
        self.model_frame.pack(side=tk.LEFT, padx=20)
        
        self.cnn_radio = tk.Radiobutton(self.model_frame, text="CNN Model", 
                                       variable=self.model_var, value="cnn")
        self.cnn_radio.pack(anchor=tk.W)
        
        self.xgb_radio = tk.Radiobutton(self.model_frame, text="XGBoost Model", 
                                       variable=self.model_var, value="xgb")
        self.xgb_radio.pack(anchor=tk.W)
        
        # Main display area
        self.display_frame = Frame(root)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure for displaying images
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Load an image to start")
        self.status_bar = Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load the trained models from disk"""
        try:
            self.status_var.set("Loading models, please wait...")
            self.root.update()
            
            # Load XGBoost model
            self.xgb_model = joblib.load(XGB_MODEL_PATH)
            
            # Load CNN model
            self.cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
            
            # Load Bounding Box model
            self.bbox_model = tf.keras.models.load_model(BBOX_MODEL_PATH)
            
            self.models_loaded = True
            self.status_var.set("Models loaded successfully!")
        except Exception as e:
            self.status_var.set(f"Error loading models: {str(e)}")
            print(f"Error loading models: {str(e)}")
    
    def load_image(self):
        """Open file dialog to select and load an image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    self.status_var.set(f"Error: Could not read image {file_path}")
                    return
                
                # Display the image
                self.show_image(self.current_image)
                self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
                self.detect_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.status_var.set(f"Error loading image: {str(e)}")
    
    def show_image(self, image):
        """Show an image in the matplotlib figure"""
        self.ax.clear()
        # Convert from BGR to RGB for display
        self.ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.ax.axis('off')
        self.canvas.draw()
    
    def detect_masks(self):
        """Detect faces and mask status in the current image"""
        if self.current_image is None:
            self.status_var.set("No image loaded!")
            return
            
        if not self.models_loaded:
            self.status_var.set("Models not loaded correctly!")
            return
        
        model_type = self.model_var.get()
        self.status_var.set(f"Detecting with {model_type.upper()} model...")
        self.root.update()
        
        try:
            # Process the image
            result_img = self.process_image(self.current_image, model_type)
            
            # Show the result
            self.show_image(result_img)
            self.status_var.set("Detection completed!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error during detection: {str(e)}")
    
    def process_image(self, image, model_type):
        """Process the image to detect faces and predict mask status"""
        result_img = image.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        # If no faces detected with Haar cascade, try the bounding box model
        if len(faces) == 0:
            h, w = image.shape[:2]
            image_resized = cv2.resize(image, (224, 224)) / 255.0
            input_image = np.expand_dims(image_resized, axis=0)
            
            # Predict bounding box
            bbox_pred = self.bbox_model.predict(input_image)[0]
            
            # Scale back to original image size
            xmin = max(0, int(bbox_pred[0] * w))
            ymin = max(0, int(bbox_pred[1] * h))
            xmax = min(w, int(bbox_pred[2] * w))
            ymax = min(h, int(bbox_pred[3] * h))
            
            # Add to faces list if the box is valid
            if xmax > xmin and ymax > ymin:
                faces = np.array([[xmin, ymin, xmax - xmin, ymax - ymin]])
        
        # Label names and colors
        label_names = ['With Mask', 'No Mask']
        color_map = [(0, 255, 0), (0, 0, 255)]  # Green for mask, Red for no mask
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            
            if face_img.size == 0:
                continue
            
            if model_type == "cnn":
                # Process for CNN
                face_resized = cv2.resize(face_img, (224, 224))
                processed_face = face_resized.astype("float32") / 255.0
                processed_face = np.expand_dims(processed_face, axis=0)
                prediction = self.cnn_model.predict(processed_face, verbose=0)[0]
                
            elif model_type == "xgb":
                # Process for XGBoost
                features = self.extract_features_from_image(face_img)
                prediction = self.xgb_model.predict_proba([features])[0]
            
            class_id = np.argmax(prediction)
            confidence = prediction[class_id]
            
            #label = f"{label_names[class_id]}: {confidence:.2f}"
            if 0 <= class_id < len(label_names):
               label = f"{label_names[class_id]}: {confidence:.2f}"
            else:
               label = f"Unknown ({class_id}): {confidence:.2f}"

            color = (0, 255, 0)
            
            # Draw rectangle and label
            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
            
            # Create a filled rectangle for text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_img, (x, y - 25), (x + text_size[0], y), color, -1)
            
            # Add text
            cv2.putText(result_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 2)
        
        return result_img
    
    def extract_features_from_image(self, image):
        """Extract features from image for XGBoost model"""
        features = []
        
        # Convert to color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 1. SIFT Features
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is not None and len(keypoints) > 0:
            n_keypoints = len(keypoints)
            avg_response = np.mean([kp.response for kp in keypoints])
            max_response = np.max([kp.response for kp in keypoints])
            avg_size = np.mean([kp.size for kp in keypoints])
            avg_descriptors = np.mean(descriptors, axis=0) if descriptors.size > 0 else np.zeros(128)
            features.extend([n_keypoints, avg_response, max_response, avg_size])
            features.extend(avg_descriptors[::4])  # Take every 4th value 32 features
        else:
            features.extend([0, 0, 0, 0])
            features.extend(np.zeros(32))

        # 2. Gaussian Features
        #for sigma in [1, 3, 5]:
          #  gaussian = cv2.GaussianBlur(gray, (0, 0), sigma)
            #features.append(np.mean(gaussian))
            #features.append(np.std(gaussian))

        for sigma in [1, 3, 5]:
          ksize = int(6 * sigma + 1)
          if ksize % 2 == 0:
                ksize += 1
          gaussian = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)
          mean = np.mean(gaussian)
          std = np.std(gaussian)
          features.extend([mean, std])
            
         

        # 3. Harris Corner
        harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        features.append(np.mean(harris))
        features.append(np.std(harris))
        features.append(np.max(harris))
        features.append(np.sum(harris > 0.01 * harris.max()))

        # 4. Color Features BGR and HSV
        for i in range(3):
            channel = image[:, :, i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(np.median(channel))
            features.append(np.max(channel) - np.min(channel))

        for i in range(3):
            channel = hsv[:, :, i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(np.median(channel))
            features.append(np.max(channel) - np.min(channel))

        # 5. Texture Features GLCM
        glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        glcm = graycomatrix(gray, distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

        for prop in glcm_props:
            prop_values = graycoprops(glcm, prop)
            features.extend(prop_values.flatten())

        # 6. Shape Features Hu Moments
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                hu_moments = cv2.HuMoments(moments)
                hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
                features.extend(hu_moments.flatten())
            else:
                features.extend(np.zeros(7))
        else:
            features.extend(np.zeros(7))

        return np.array(features)

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = MaskDetectionGUI(root)

import os
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow as tf
import tensorflow_hub as hub
import joblib
from skimage.feature import graycomatrix, graycoprops

# Path to the downloaded model files - update these to your actual paths
# Path to the downloaded model files - update these to your actual paths
MODELS_DIR = r"C:\Users\NADA\Downloads"  # Use r prefix for raw string to handle backslashes
XGB_MODEL_PATH = os.path.join(MODELS_DIR, "best_xgb_model.pkl")
CNN_MODEL_PATH = os.path.join(MODELS_DIR, "cnn_face_mask_model.h5")
BBOX_MODEL_PATH = os.path.join(MODELS_DIR, "bbox.h5")
class MaskDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Mask Detection")
        self.root.geometry("1000x800")
        
        # Variables
        self.current_image = None
        self.models_loaded = False
        
        # Top frame for buttons
        self.top_frame = Frame(root)
        self.top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # Load Image Button
        self.load_btn = Button(self.top_frame, text="Load Image", command=self.load_image, 
                               bg="#4CAF50", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.load_btn.pack(side=tk.LEFT, padx=5)
        
        # Detect Button
        self.detect_btn = Button(self.top_frame, text="Detect Masks", command=self.detect_masks,
                                bg="#2196F3", fg="white", font=("Arial", 12), padx=10, pady=5)
        self.detect_btn.pack(side=tk.LEFT, padx=5)
        self.detect_btn.config(state=tk.DISABLED)
        
        # Radio buttons for model selection
        self.model_var = tk.StringVar(value="cnn")
        self.model_frame = Frame(self.top_frame)
        self.model_frame.pack(side=tk.LEFT, padx=20)
        
        self.cnn_radio = tk.Radiobutton(self.model_frame, text="CNN Model", 
                                       variable=self.model_var, value="cnn")
        self.cnn_radio.pack(anchor=tk.W)
        
        self.xgb_radio = tk.Radiobutton(self.model_frame, text="XGBoost Model", 
                                       variable=self.model_var, value="xgb")
        self.xgb_radio.pack(anchor=tk.W)
        
        # Main display area
        self.display_frame = Frame(root)
        self.display_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create matplotlib figure for displaying images
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.display_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Load an image to start")
        self.status_bar = Label(root, textvariable=self.status_var, bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Load models
        self.load_models()
    
    def load_models(self):
        """Load the trained models from disk"""
        try:
            self.status_var.set("Loading models, please wait...")
            self.root.update()
            
            # Load XGBoost model
            self.xgb_model = joblib.load(XGB_MODEL_PATH)
            
            # Load CNN model
            self.cnn_model = tf.keras.models.load_model(CNN_MODEL_PATH)
            
            # Load Bounding Box model
            self.bbox_model = tf.keras.models.load_model(BBOX_MODEL_PATH)
            
            self.models_loaded = True
            self.status_var.set("Models loaded successfully!")
        except Exception as e:
            self.status_var.set(f"Error loading models: {str(e)}")
            print(f"Error loading models: {str(e)}")
    
    def load_image(self):
        """Open file dialog to select and load an image"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        
        if file_path:
            try:
                self.current_image = cv2.imread(file_path)
                if self.current_image is None:
                    self.status_var.set(f"Error: Could not read image {file_path}")
                    return
                
                # Display the image
                self.show_image(self.current_image)
                self.status_var.set(f"Loaded image: {os.path.basename(file_path)}")
                self.detect_btn.config(state=tk.NORMAL)
            except Exception as e:
                self.status_var.set(f"Error loading image: {str(e)}")
    
    def show_image(self, image):
        """Show an image in the matplotlib figure"""
        self.ax.clear()
        # Convert from BGR to RGB for display
        self.ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.ax.axis('off')
        self.canvas.draw()
    
    def detect_masks(self):
        """Detect faces and mask status in the current image"""
        if self.current_image is None:
            self.status_var.set("No image loaded!")
            return
            
        if not self.models_loaded:
            self.status_var.set("Models not loaded correctly!")
            return
        
        model_type = self.model_var.get()
        self.status_var.set(f"Detecting with {model_type.upper()} model...")
        self.root.update()
        
        try:
            # Process the image
            result_img = self.process_image(self.current_image, model_type)
            
            # Show the result
            self.show_image(result_img)
            self.status_var.set("Detection completed!")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.status_var.set(f"Error during detection: {str(e)}")
    
    def process_image(self, image, model_type):
        """Process the image to detect faces and predict mask status"""
        result_img = image.copy()
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        # If no faces detected with Haar cascade, try the bounding box model
        if len(faces) == 0:
            h, w = image.shape[:2]
            image_resized = cv2.resize(image, (224, 224)) / 255.0
            input_image = np.expand_dims(image_resized, axis=0)
            
            # Predict bounding box
            bbox_pred = self.bbox_model.predict(input_image)[0]
            
            # Scale back to original image size
            xmin = max(0, int(bbox_pred[0] * w))
            ymin = max(0, int(bbox_pred[1] * h))
            xmax = min(w, int(bbox_pred[2] * w))
            ymax = min(h, int(bbox_pred[3] * h))
            
            # Add to faces list if the box is valid
            if xmax > xmin and ymax > ymin:
                faces = np.array([[xmin, ymin, xmax - xmin, ymax - ymin]])
        
        # Label names and colors
        label_names = ['With Mask', 'No Mask']
        color_map = [(0, 255, 0), (0, 0, 255)]  # Green for mask, Red for no mask
        
        # Process each detected face
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            
            if face_img.size == 0:
                continue
            
            if model_type == "cnn":
                # Process for CNN
                face_resized = cv2.resize(face_img, (224, 224))
                processed_face = face_resized.astype("float32") / 255.0
                processed_face = np.expand_dims(processed_face, axis=0)
                prediction = self.cnn_model.predict(processed_face, verbose=0)[0]
                
            elif model_type == "xgb":
                # Process for XGBoost
                features = self.extract_features_from_image(face_img)
                prediction = self.xgb_model.predict_proba([features])[0]
            
            class_id = np.argmax(prediction)
            confidence = prediction[class_id]
            
            #label = f"{label_names[class_id]}: {confidence:.2f}"
            if 0 <= class_id < len(label_names):
               label = f"{label_names[class_id]}: {confidence:.2f}"
            else:
               label = f"Unknown ({class_id}): {confidence:.2f}"

            color = (0, 255, 0)
            
            # Draw rectangle and label
            cv2.rectangle(result_img, (x, y), (x+w, y+h), color, 2)
            
            # Create a filled rectangle for text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(result_img, (x, y - 25), (x + text_size[0], y), color, -1)
            
            # Add text
            cv2.putText(result_img, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, (255, 255, 255), 2)
        
        return result_img
    
    def extract_features_from_image(self, image):
        """Extract features from image for XGBoost model"""
        features = []
        
        # Convert to color spaces
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # 1. SIFT Features
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)

        if descriptors is not None and len(keypoints) > 0:
            n_keypoints = len(keypoints)
            avg_response = np.mean([kp.response for kp in keypoints])
            max_response = np.max([kp.response for kp in keypoints])
            avg_size = np.mean([kp.size for kp in keypoints])
            avg_descriptors = np.mean(descriptors, axis=0) if descriptors.size > 0 else np.zeros(128)
            features.extend([n_keypoints, avg_response, max_response, avg_size])
            features.extend(avg_descriptors[::4])  # Take every 4th value 32 features
        else:
            features.extend([0, 0, 0, 0])
            features.extend(np.zeros(32))

        # 2. Gaussian Features
        #for sigma in [1, 3, 5]:
          #  gaussian = cv2.GaussianBlur(gray, (0, 0), sigma)
            #features.append(np.mean(gaussian))
            #features.append(np.std(gaussian))

        for sigma in [1, 3, 5]:
          ksize = int(6 * sigma + 1)
          if ksize % 2 == 0:
                ksize += 1
          gaussian = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)
          mean = np.mean(gaussian)
          std = np.std(gaussian)
          features.extend([mean, std])
            
         

        # 3. Harris Corner
        harris = cv2.cornerHarris(np.float32(gray), 2, 3, 0.04)
        features.append(np.mean(harris))
        features.append(np.std(harris))
        features.append(np.max(harris))
        features.append(np.sum(harris > 0.01 * harris.max()))

        # 4. Color Features BGR and HSV
        for i in range(3):
            channel = image[:, :, i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(np.median(channel))
            features.append(np.max(channel) - np.min(channel))

        for i in range(3):
            channel = hsv[:, :, i]
            features.append(np.mean(channel))
            features.append(np.std(channel))
            features.append(np.median(channel))
            features.append(np.max(channel) - np.min(channel))

        # 5. Texture Features GLCM
        glcm_props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        
        glcm = graycomatrix(gray, distances=[1, 3],
                        angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                        levels=256, symmetric=True, normed=True)

        for prop in glcm_props:
            prop_values = graycoprops(glcm, prop)
            features.extend(prop_values.flatten())

        # 6. Shape Features Hu Moments
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                hu_moments = cv2.HuMoments(moments)
                hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments))
                features.extend(hu_moments.flatten())
            else:
                features.extend(np.zeros(7))
        else:
            features.extend(np.zeros(7))

        return np.array(features)

# Main application entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = MaskDetectionGUI(root)
    root.mainloop()