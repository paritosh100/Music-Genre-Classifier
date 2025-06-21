import os
import numpy as np
import tensorflow as tf
import librosa
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt # For simplicity in demo, disable CSRF for POST
from django.conf import settings
import json # For handling JSON response

# --- Model Loading (Optimized for Django) ---
# It's best to load the model once when the server starts, not on every request.
# We'll use a global variable and a flag to ensure it's loaded only once.
# In a larger production app, consider using Django AppConfig.ready() for this.
GLOBAL_MODEL = None
GLOBAL_GENRES = []
MODEL_LOAD_ERROR = None

def load_ml_model():
    global GLOBAL_MODEL, GLOBAL_GENRES, MODEL_LOAD_ERROR

    if GLOBAL_MODEL is not None:
        print("Model already loaded.")
        return

    print("Loading ML model...")
    try:
        # Define model path relative to Django app directory
        # The model is expected to be directly inside the 'classifier_app' folder
        model_path = os.path.join('classifier_app', 'Genre-Classifier.h5')
        print(f"Attempting to load model from: {model_path}")

        # Use tf.keras.models.load_model to load the .h5 file
        GLOBAL_MODEL = tf.keras.models.load_model(model_path)
        GLOBAL_MODEL.summary() # Print model summary to console for verification
        print("ML model loaded successfully!")

        # Define genres - MUST match the order used during training
        # If your SPEC_DIR was in the main project folder previously, adjust path accordingly.
        # Otherwise, hardcoding these is fine if they are consistent.
        # This assumes the order: ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']
        GLOBAL_GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
        print(f"Loaded genres: {GLOBAL_GENRES}")

    except Exception as e:
        MODEL_LOAD_ERROR = f"Failed to load ML model: {e}"
        GLOBAL_MODEL = None # Ensure model is None if loading fails
        print(MODEL_LOAD_ERROR)

# Load the model when Django server starts for the first time
# This is a simple way for development. For production, consider AppConfig.
load_ml_model()


# --- Audio Processing and Spectrogram Generation (Adapted from your Classifier_code) ---
def process_audio_to_spectrogram(audio_file_path):
    """
    Processes an audio file to generate a Mel spectrogram, matching the dimensions
    and normalization of the trained Keras model.
    """
    SAMPLE_RATE = 22050
    DURATION = 30
    N_MELS = 128
    HOP_LENGTH = 512
    TARGET_T = 128
    N_FFT = 2048 # Standard FFT window size

    try:
        # Load audio file: y is the audio time series, sr is the sampling rate
        y, sr = librosa.load(audio_file_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)

        # Trim or pad audio to match DURATION (30 seconds)
        target_samples = SAMPLE_RATE * DURATION
        if y.shape[0] > target_samples:
            y = y[:target_samples]
        elif y.shape[0] < target_samples:
            padding = np.zeros(target_samples - y.shape[0], dtype=y.dtype)
            y = np.concatenate((y, padding))

        # Compute Mel spectrogram
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
        # Convert power spectrogram to decibel (dB) scale
        S_db = librosa.power_to_db(S, ref=np.max)

        # Pad/Trim spectrogram to TARGET_T frames (time axis)
        S_db_fixed = librosa.util.fix_length(S_db, size=TARGET_T, axis=1)

        # Add channel dimension: (N_MELS, TARGET_T, 1)
        spec = S_db_fixed[..., np.newaxis]

        # Normalize to [0,1]
        spec_min = spec.min()
        spec_max = spec.max()
        spec_normalized = (spec - spec_min) / (spec_max - spec_min + 1e-6) # Add epsilon for stability

        # Reshape for model input: [1, height, width, channels]
        model_input = np.expand_dims(spec_normalized, axis=0) # Add batch dimension

        return model_input.astype(np.float32)

    except Exception as e:
        raise Exception(f"Audio processing failed: {e}")

# --- Django Views ---

# Use csrf_exempt for simplicity in this demo.
# In a production environment, you should properly handle CSRF tokens in your forms.
@csrf_exempt
def upload_file_and_predict(request):
    if request.method == 'POST':
        if GLOBAL_MODEL is None:
            # Attempt to reload model if it failed previously (e.g., first request after server start)
            load_ml_model()
            if GLOBAL_MODEL is None:
                return JsonResponse({'error': MODEL_LOAD_ERROR}, status=500)

        if 'audio_file' not in request.FILES:
            return JsonResponse({'error': 'No audio file provided.'}, status=400)

        audio_file = request.FILES['audio_file']
        # Ensure file type is supported (e.g., .wav, .mp3)
        if not audio_file.name.endswith(('.wav', '.mp3', '.flac', '.ogg')):
            return JsonResponse({'error': 'Unsupported file format. Please upload .wav, .mp3, .flac, or .ogg.'}, status=400)

        # Save the uploaded file temporarily
        fs_path = os.path.join(settings.MEDIA_ROOT, audio_file.name)
        os.makedirs(settings.MEDIA_ROOT, exist_ok=True) # Ensure media directory exists
        try:
            with open(fs_path, 'wb+') as destination:
                for chunk in audio_file.chunks():
                    destination.write(chunk)

            # Process audio and make prediction
            processed_input = process_audio_to_spectrogram(fs_path)
            predictions = GLOBAL_MODEL.predict(processed_input)
            scores = predictions[0] # Get probabilities for the single input
            predicted_class_id = np.argmax(scores)
            predicted_genre = GLOBAL_GENRES[predicted_class_id]
            confidence = float(scores[predicted_class_id] * 100) # Convert to float for JSON serialization

            response_data = {
                'success': True,
                'predicted_genre': predicted_genre,
                'confidence': confidence,
            }
            return JsonResponse(response_data)

        except Exception as e:
            return JsonResponse({'error': f'Prediction failed: {e}'}, status=500)
        finally:
            # Clean up the temporarily saved file
            if os.path.exists(fs_path):
                os.remove(fs_path)
    else:
        # Render the upload form on GET request
        return render(request, 'classifier_app/upload.html')

