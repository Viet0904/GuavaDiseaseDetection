import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import datetime
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
)
from tensorflow.keras.callbacks import Callback

# Define necessary parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 8
EPOCHS = 100

# Get current time
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class DetailedLoggingCallback(Callback):
    def __init__(self, valid_data, test_data, file_prefix="CNN_optAdam_lr0.001_bs32"):
        super(DetailedLoggingCallback, self).__init__()
        self.valid_data = valid_data
        self.test_data = test_data
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.detail_file_path = f"{file_prefix}_{current_time}_details.txt"
        self.Confusion_Matrix_path = f"{file_prefix}_{current_time}_confusion_matrix"
        self.report_path = f"{file_prefix}_{current_time}_report"
        # Initialize file and write header with tab as separator
        with open(self.detail_file_path, "w") as f:
            f.write(
                "Epoch\tTrain Loss\tTrain Accuracy\tValid Loss\tValid Accuracy\tTest Loss\tTest Accuracy\tTest Precision\tTest Recall\tTest F1-Score\tTest MCC\tTest CMC\tValid Precision\tValid Recall\tValid F1-Score\tValid MCC\tValid CMC\n"
            )
        with open(f"{self.Confusion_Matrix_path}_test.txt", "w") as f:
            f.write("Confusion Matrix Test\n")
        with open(f"{self.Confusion_Matrix_path}_valid.txt", "w") as f:
            f.write("Confusion Matrix Valid\n")
        with open(f"{self.report_path}_test.txt", "w") as f:
            f.write("Classification Report Test\n")
        with open(f"{self.report_path}_valid.txt", "w") as f:
            f.write("Classification Report Valid\n")
        self.epoch_logs = []
        self.epoch_cm_logs = []
        self.epoch_report = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        test_results = self.model.evaluate(self.test_data, verbose=0)
        test_loss, test_accuracy = test_results[0], test_results[1]
        y_pred_valid = np.argmax(self.model.predict(self.valid_data), axis=1)
        y_true_valid = self.valid_data.classes
        cm_valid = confusion_matrix(y_true_valid, y_pred_valid)
        report_valid = classification_report(
            y_true_valid, y_pred_valid, digits=5, output_dict=True
        )
        precision_valid = precision_score(y_true_valid, y_pred_valid, average="macro")
        recall_valid = recall_score(y_true_valid, y_pred_valid, average="macro")
        f1_valid = f1_score(y_true_valid, y_pred_valid, average="macro")
        mcc_valid = matthews_corrcoef(y_true_valid, y_pred_valid)
        cmc_valid = cohen_kappa_score(y_true_valid, y_pred_valid)
        valid_accuracy = logs.get("val_accuracy", 0)

        y_pred_test = np.argmax(self.model.predict(self.test_data), axis=1)
        y_true_test = self.test_data.classes
        cm_test = confusion_matrix(y_true_test, y_pred_test)
        report_test = classification_report(
            y_true_test, y_pred_test, digits=5, output_dict=True
        )
        precision_test = precision_score(y_true_test, y_pred_test, average="macro")
        recall_test = recall_score(y_true_test, y_pred_test, average="macro")
        f1_test = f1_score(y_true_test, y_pred_test, average="macro")
        mcc_test = matthews_corrcoef(y_true_test, y_pred_test)
        cmc_test = cohen_kappa_score(y_true_test, y_pred_test)
        test_accuracy = logs.get("accuracy", 0)
        cm_test = confusion_matrix(y_true_test, y_pred_test)
        report_test = classification_report(
            y_true_test, y_pred_test, digits=5, output_dict=True
        )
        print("Confusion Matrix (Validation):")
        print(cm_valid)
        print("Classification Report (Validation):")
        print(report_valid)
        print("Confusion Matrix (Test):")
        print(cm_test)
        print("Classification Report (Test):")
        print(report_test)
        self.epoch_cm_logs.append((epoch + 1, cm_test, cm_valid))
        self.epoch_report.append((epoch + 1, report_test, report_valid))
        # Save information to temporary list with values separated by tab
        self.epoch_logs.append(
            (
                epoch + 1,
                logs.get("loss", 0),
                logs.get("accuracy", 0),
                logs.get("val_loss", 0),
                logs.get("val_accuracy", 0),
                test_loss,
                test_accuracy,
                precision_test,
                recall_test,
                f1_test,
                mcc_test,
                cmc_test,
                precision_valid,
                recall_valid,
                f1_valid,
                mcc_valid,
                cmc_valid,
            )
        )

    def on_train_end(self, logs=None):
        # Save information from each epoch to detail file, using tab as separator
        with open(self.detail_file_path, "a") as f:
            for log in self.epoch_logs:
                f.write(
                    f"{log[0]}\t{log[1]:.5f}\t{log[2]:.5f}\t{log[3]:.5f}\t{log[4]:.5f}\t{log[5]:.5f}\t{log[6]:.5f}\t{log[7]:.5f}\t{log[8]:.5f}\t{log[9]:.5f}\t{log[10]:.5f}\t{log[11]:.5f}\t{log[12]:.5f}\t{log[13]:.5f}\t{log[14]:.5f}\t{log[15]:.5f}\t{log[16]:.5f}\n"
                )
        with open(f"{self.Confusion_Matrix_path}_test.txt", "a") as f:
            for log in self.epoch_cm_logs:
                f.write(f"{log[1]}\n\n")
        with open(f"{self.Confusion_Matrix_path}_valid.txt", "a") as f:
            for log in self.epoch_cm_logs:
                f.write(f"{log[2]}\n\n")
        with open(f"{self.report_path}_test.txt", "a") as f:
            for log in self.epoch_report:
                f.write(f"{log[1]}\n\n")
        with open(f"{self.report_path}_valid.txt", "a") as f:
            for log in self.epoch_report:
                f.write(f"{log[2]}\n\n")


# Create paths to data directories
train_dir = "./images_segmentation/train"
test_dir = "./images_segmentation/test"
valid_dir = "./images_segmentation/valid"

# Create data generators for training set
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)

train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

# Create data generators for test and validation sets
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
detailed_logging_callback = DetailedLoggingCallback(
    valid_data=valid_data, test_data=test_data
)
# Create a CNN model
model = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), input_shape=(224, 224, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(512),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

# Compile the model with new metrics
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

# Train the model with the new callback
history = model.fit(
    train_data,
    epochs=EPOCHS,
    validation_data=valid_data,
    verbose=1,
    callbacks=[detailed_logging_callback],
)

# Save the model in the native Keras format
model.save("./CNN_optAdam_lr0.001_bs32.h5")

# Load the model from the correct path
loaded_model = models.load_model("./CNN_optAdam_lr0.001_bs32.h5")
