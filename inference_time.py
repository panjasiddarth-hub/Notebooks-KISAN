import time
import numpy as np
import tensorflow as tf

MODEL_PATH = r"artifacts\disease_model.keras"
IMG_SIZE = (224, 224)

model = tf.keras.models.load_model(MODEL_PATH)

# One image-shaped input (timing does not depend on the picture's content)
x = np.random.rand(1, IMG_SIZE[0], IMG_SIZE[1], 3).astype("float32") * 255

# Warm-up runs (the first calls are slower and should not be counted)
for _ in range(5):
    model.predict(x, verbose=0)

# Timed runs
N = 100
start = time.time()
for _ in range(N):
    model.predict(x, verbose=0)
ms = (time.time() - start) / N * 1000

print("Inference time: %.1f ms per image (CPU)" % ms)