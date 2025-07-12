*Bonus Task: Deploy  MNIST Classifier* 

*Bonus Task: Web App Deployment with Streamlit*

We’ll use **Streamlit** to create a web interface that lets users upload a digit image and get a prediction from trained CNN.

✅ Step 1: Install Streamlit

In environment or Colab:

```bash
pip install streamlit
```

> ⚠️ Streamlit doesn’t run inside Colab directly. Use it locally for deployment.

✅ Step 2: Save Trained MNIST Model

After training in TensorFlow, save the model:

```python
model.save('mnist_cnn.h5')
```

✅ Step 3: Streamlit App Code

Create a file called `mnist_app.py`:

```python
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load the model
model = tf.keras.models.load_model('mnist_cnn.h5')

st.title("MNIST Digit Classifier")
st.write("Upload a digit image (28x28 grayscale) and the model will predict it.")

uploaded_file = st.file_uploader("Choose a digit image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert colors (white digit on black bg)
    image = image.resize((28, 28))
    st.image(image, caption="Uploaded Image", width=150)

    img_array = np.array(image)
    img_array = img_array / 255.0  # Normalize
    img_array = img_array.reshape(1, 28, 28)

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.success(f"Predicted Digit: **{predicted_digit}**")
```

✅ Step 4: Run Locally

From your terminal:

```bash
streamlit run mnist_app.py
```

✅ Step 5: Deploy Online (Optional)

You can deploy app via:

* [Streamlit Cloud](https://share.streamlit.io/)
* GitHub + Streamlit (just push your `mnist_app.py` and `mnist_cnn.h5` to GitHub)

#📸 Screenshot

Take a screenshot of:

* The running Streamlit app.
* A prediction result from your uploaded digit.

# 🔗 Add This in Report & GitHub README

> *Bonus Task: Streamlit Web App*
>
> * Built a simple interface for digit prediction using Streamlit.
> * Allows image upload and returns predictions using our trained TensorFlow CNN.
