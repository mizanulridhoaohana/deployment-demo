import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the saved model
model = tf.keras.models.load_model('./model/model.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))  # Load and resize the image
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image
    return img_array

img_path = './dataset/validation/paper1.png'

# Print prediction
img_array = preprocess_image(img_path)
predictions = model.predict(img_array)

labels = ['Paper', 'Rock', 'Scissors']
predicted_index = np.argmax(predictions, axis=1)[0]
print("Predictions:", labels[predicted_index])

print("\nConfidence result: ")
for i,j in enumerate(predictions[0]):
    print(labels[i], "\t:  ", j)

plt.imshow(image.load_img(img_path))
plt.title(f'Predictions: {labels[predicted_index]}')
plt.axis('off')
plt.show()
