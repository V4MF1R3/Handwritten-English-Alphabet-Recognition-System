import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Dictionary for label conversion
word_dict = { 0:'A',1:'B',2:'C',3:'D',4:'E',5:'F',6:'G',7:'H',8:'I',9:'J',10:'K',11:'L',12:'M',13:'N',14:'O',15:'P',16:'Q',17:'R',18:'S',19:'T',20:'U',21:'V',22:'W',23:'X', 24:'Y',25:'Z' }

# Function to classify the input image
def classify_image():
    global label_packed
    filepath = filedialog.askopenfilename()
    image = Image.open(filepath)
    image = image.resize((28, 28))
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1)
    pred = model.predict([image])[0]
    result = np.argmax(pred)
    label.configure(text = f"Predicted Alphabet: {word_dict[result]}")

# Creating the main window
root = tk.Tk()
root.geometry("300x300")
root.title("Alphabet Recognition System")

# Button to select an image
button = tk.Button(root, text="Select Image", command=classify_image)
button.pack(side=tk.BOTTOM, pady=50)

# Label to display the prediction
label = tk.Label(root, text="Predicted Alphabet: ", font=("Arial", 20))
label.pack()

# Running the GUI application
root.mainloop()