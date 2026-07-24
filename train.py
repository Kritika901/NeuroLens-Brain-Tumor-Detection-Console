import json
import os

import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from preprocess import CLASS_NAMES, NUM_CLASSES, test_generator, train_generator, validation_generator
from utils import plot_confusion_matrix, plot_training_history, print_classification_report

EPOCHS = 50
MODEL_PATH = "models/brain_tumor_model.keras"

os.makedirs("models", exist_ok=True)
os.makedirs("images", exist_ok=True)

# Save the class name -> index mapping so predict.py / app.py can
# turn model output back into readable labels.
with open("models/class_indices.json", "w") as f:
    json.dump(train_generator.class_indices, f)

print(f"Classes found: {CLASS_NAMES}")
print("Class labels saved to models/class_indices.json")

# --------------------------------------------------
# Build the model (MobileNetV2 base + custom head)
# --------------------------------------------------

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3),
)
base_model.trainable = False  

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.3)(x)
predictions = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# --------------------------------------------------
# Callbacks
# --------------------------------------------------

callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
    ModelCheckpoint(filepath=MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
]

# --------------------------------------------------
# Train
# --------------------------------------------------

history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=EPOCHS,
    callbacks=callbacks,
)

print("\nTraining completed. Best model saved at:", MODEL_PATH)

# --------------------------------------------------
# Evaluate on the test set
# --------------------------------------------------

test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
print(f"\nTest Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

test_generator.reset()
predictions = model.predict(test_generator, verbose=1)
y_pred = np.argmax(predictions, axis=1)
y_true = test_generator.classes

plot_training_history(history)
print_classification_report(y_true, y_pred, CLASS_NAMES)
plot_confusion_matrix(y_true, y_pred, CLASS_NAMES)

print("\nDone. Run 'streamlit run app.py' to try the model out.")
