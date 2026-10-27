import os, time
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

# ---- settings (change DATA_DIR only) ----
DATA_DIR  = r"C:\Users\panja\OneDrive\Desktop\kisan ai docunmentation\KisanAI_Colab_Notebooks\color"   # same path you used in the last run
IMG_SIZE  = (224, 224)
BATCH     = 32
SEED      = 123
VAL_SPLIT = 0.2
MODEL_PATH = r"artifacts\disease_model.keras"
# -----------------------------------------

# NOTE: no shuffle=False here, so the split matches training
val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR, validation_split=VAL_SPLIT, subset="validation",
    seed=SEED, image_size=IMG_SIZE, batch_size=BATCH, label_mode="int")
class_names = val_ds.class_names

model = tf.keras.models.load_model(MODEL_PATH)

# Collect labels and predictions in the SAME pass
y_true, y_pred = [], []
for x, y in val_ds:
    y_true.extend(y.numpy())
    y_pred.extend(np.argmax(model.predict(x, verbose=0), axis=1))
y_true, y_pred = np.array(y_true), np.array(y_pred)
print("\nAccuracy check (should be about 0.946):", round((y_true == y_pred).mean(), 4))

# 1) Confused classes
cm = confusion_matrix(y_true, y_pred)
np.fill_diagonal(cm, 0)
print("\nTop 8 confusions (true -> predicted : count)")
for idx in np.argsort(cm, axis=None)[::-1][:8]:
    i, j = np.unravel_index(idx, cm.shape)
    print(f"{class_names[i]} -> {class_names[j]} : {cm[i, j]}")

# 2) Model size
print("\nModel size: %.2f MB" % (os.path.getsize(MODEL_PATH) / 1e6))

# 3) Inference time (single image, CPU)
x1 = next(iter(val_ds))[0][:1]
model.predict(x1, verbose=0)  # warm-up
t = time.time()
for _ in range(50):
    model.predict(x1, verbose=0)
print("Inference: %.1f ms per image" % ((time.time() - t) / 50 * 1000))