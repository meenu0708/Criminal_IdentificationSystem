import os
import numpy as np
from mtcnn import MTCNN
import cv2
from django.conf import settings
from django.shortcuts import render, redirect
from .models import UploadedImage, CriminalImage
from .forms import ImageForm, CanvasImageForm
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
from .models import Feature, FeatureOption
import base64
from django.http import HttpResponse
from .forms import LoginForm
from .forms import UserForm
from django.contrib import messages, auth
from django.contrib.auth.models import User
# Create your views here

media_root = settings.MEDIA_ROOT
def home(request):
    return render(request, 'home.html')

def login(request):
    if request.method == 'POST':
        form = LoginForm(request.POST)
        username = request.POST['username']
        password = request.POST['password']
        if form.is_valid():
            user = auth.authenticate(username=username, password=password)

            if user is not None:
                auth.login(request, user)
                return redirect('/')
            else:
                messages.info(request, "Invalid Credentials")
                return redirect('login')

    else:
        form = LoginForm()
    return render(request, "login.html",{'form': form})

def logout(request):
    auth.logout(request)
    return redirect('/')

def register(request):
    if request.method == 'POST':
        form = UserForm(request.POST)
        username = request.POST['username']
        password = request.POST['password']
        cpass = request.POST['cpass']

        if form.is_valid():
            if password == cpass:
                if User.objects.filter(username=username).exists():
                    messages.info(request, "Username Taken")
                    return redirect('register')
                else:
                    user = User.objects.create_user(username=username, password=password)
                    user.save();
                    print("User created")
            else:
                messages.info(request, "Password not matching")
                return redirect('register')
            return redirect('login')
    else:
        form = UserForm()

    return render(request, 'register.html', {'form': form})

# -------------------------------------------------------------------------------------------------------------
# Creating a sketching interface
def sketch_interface(request):

    return render(request, 'sketch_interface.html')


def feature_detail(request, feature_slug):
    feature = Feature.objects.get(slug=feature_slug)
    feature_options = FeatureOption.objects.filter(feature=feature)
    return render(request, 'sketch_interface.html', {'feature_options': feature_options,'feature':feature})

def save_image(request):
    if request.method == 'POST':
        # Get the canvas image data from the POST request
        canvas_image_data = request.POST.get('canvas_image_data')
        image_name = request.POST.get('image_name')
        try:
            # Decode the base64-encoded canvas image data
            decoded_image_data = base64.b64decode(canvas_image_data.split(',')[1])
            # Save the canvas image data to a file
            save_path = os.path.join('E:/Projects2024/Criminal_Identification_project/created_sketches/',f'{image_name}.jpg')
            with open(save_path, 'wb') as f:
                f.write(decoded_image_data)
            return render(request,"sketch_success.html",{"Success":"Sketch saved Successfully"})
        except Exception as e:
            # Handle any errors
            return HttpResponse(f'Error saving image: {e}', status=500)
    else:
        # Handle invalid request method
        return HttpResponse('Invalid request method', status=400)


# ------------------------------------------------------------------------------------------------------------
# Identify the Sketch
def upload_sketch(request):
    if request.method == 'POST':
        form = ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            form.save()
            obj = form.instance
            return render(request, "sketch_upload.html", {"obj": obj})
    else:
        form = ImageForm()
    return render(request, 'sketch_upload.html', {"form": form})



#original_images_dir = os.path.join(media_root, "dataset/original_images")
runtime_images_dir = os.path.join(media_root, "runtime")

# Face Recognition
def sketch_match(request):
    # Extracting the features of input sketch
    input_image = UploadedImage.objects.last()
    image_path = input_image.image.path
    image_array = cv2.imread(image_path)
    cropped_image = face_detect(image_array)
    file_path = os.path.join(runtime_images_dir, "input_cropped_face.jpg")
    cv2.imwrite(file_path, cropped_image)
    image_features = extract_features(file_path)

    # Extracting the features of dataset sketches
    criminal_images = CriminalImage.objects.all()
    features_file = os.path.join(media_root, 'features.npy')
    dataset_features = extract_features_dataset(criminal_images, features_file)

    # Finding the best match for the input sketch using KNN Algorithm
    k = 5
    knn_model = NearestNeighbors(n_neighbors=k, metric='euclidean')
    dataset_features_flattened = dataset_features.reshape(dataset_features.shape[0], -1)
    knn_model.fit(dataset_features_flattened)
    distances, indices = knn_model.kneighbors(image_features)
    best_match_index = indices[0, 0]
    best_match_index=int(best_match_index)
    match_image = CriminalImage.objects.values_list('original_image', flat=True)[best_match_index]

    # Calculating the similarity score
    image_features_2d = image_features.reshape(1, -1)
    matched_image_features_2d = dataset_features[best_match_index].reshape(1, -1)
    similarity_score = cosine_similarity(image_features_2d, matched_image_features_2d)
    similarity_percent = similarity_score * 100

    return render(request, 'sketch_match.html', {"obj": input_image, "match_image": match_image,"score":round(similarity_percent[0][0],1)})


# -------------------------------------*-------------------------------------------------------------------


# Face Detection
detector = MTCNN()


def face_detect(sketch_image):
    image_rgb = cv2.cvtColor(sketch_image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    cropped_face = []
    for i, face in enumerate(faces):
        x, y, width, height = face['box']
        x = max(0, x)
        y = max(0, y)
        width = min(sketch_image.shape[1] - x, width)
        height = min(sketch_image.shape[0] - y, height)
        cropped_face = sketch_image[y:y + height, x:x + width]
    return cropped_face


# Feature Extraction
weights_path = os.path.join(media_root, 'vgg16_weights_tf_dim_ordering_tf_kernels.h5')
base_model = VGG16(weights=weights_path, include_top=True)
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)


# Feature extraction of input image
def extract_features(image_path):
    cropped_img = image.load_img(image_path, target_size=(224, 224))
    cropped_img = image.img_to_array(cropped_img)
    cropped_img = np.expand_dims(cropped_img, axis=0)
    cropped_img = preprocess_input(cropped_img)
    features = model.predict(cropped_img)
    return features


# Feature extraction of dataset images
def extract_features_dataset(criminal_images, features_file):
    dataset_cropped_dir = os.path.join(media_root, "runtime/dataset_cropped")
    if os.path.exists(features_file):
        features_array = np.load(features_file)
    else:
        features_array = []

    features_timestamp = os.path.getmtime(features_file) if os.path.exists(features_file) else 0
    dataset_timestamp = max(image_instance.created_at for image_instance in criminal_images)
    if features_timestamp < dataset_timestamp.timestamp():
        features_array = []
        for image_instance in criminal_images:
            img_path = image_instance.sketched_image.path
            sketch_image = cv2.imread(img_path)
            cropped_image = face_detect(sketch_image)
            file_path = os.path.join(dataset_cropped_dir, 'crop_dataset.jpg')
            cv2.imwrite(file_path, cropped_image)
            image_features = extract_features(file_path)
            features_array.append(image_features)
        features_array = np.array(features_array)
        np.save(features_file, features_array)
    return features_array
