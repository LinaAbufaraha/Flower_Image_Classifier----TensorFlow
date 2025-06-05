import argparse
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import tensorflow_hub as hub
from tensorflow.keras.preprocessing import image


def load_model(model_path):
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})
    return model    

def process_image(image):
    img = tf.image.resize(image, (224, 224))
    img = img / 255.0 
    return img.numpy()

def predict(image_path, model, top_k=5):
 
    img = Image.open(image_path)
    processed_img = process_image(np.asarray(img))

    img_expanded = np.expand_dims(processed_img, axis=0)
    predictions = model.predict(img_expanded)

    top_k_probs, top_k_classes = tf.nn.top_k(tf.nn.softmax(predictions[0]), k=top_k)
    
    return top_k_probs.numpy(), top_k_classes.numpy()


def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image.")

    parser.add_argument('image_path', type=str, help="Path to image file.")
    parser.add_argument('model', type=str, help="Path to trained Keras model.")
    parser.add_argument('--top_k', type=int, default=5, help="Return top K most likely classes.")
    parser.add_argument('--label_map', type=str, help="Path to JSON file mapping labels to names.")
    
    args = parser.parse_args()

    model = load_model(args.model)
    
  
    with open(args.label_map, 'r') as f:
        class_names = json.load(f)
    
  
    probs, classes = predict(args.image_path, model, args.top_k)
    
  
    class_names = [class_names[str(int(cls))] for cls in classes]
    
    
    print("Predicted Classes and Probabilities:")
    for prob, cls in zip(probs, class_names):
        print(f"{cls}: {prob:.4f}")


if __name__ == "__main__":
    main()