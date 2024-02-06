import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
import numpy as np
import datetime
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score,
)


class DetailedLoggingCallback(Callback):
    def __init__(
        self, valid_data, test_data, file_prefix="RESNET50_optAdam_lr0.001_bs32"
    ):
        super(DetailedLoggingCallback, self).__init__()
        self.valid_data = valid_data
        self.test_data = test_data
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.detail_file_path = f"{file_prefix}_{current_time}_details.txt"
        self.summary_file_path = f"{file_prefix}_{current_time}_summary.txt"
        # Initialize file and write header with tab as separator
        with open(self.detail_file_path, "w") as f:
            f.write(
                "Epoch\tTrain Loss\tValid Loss\tTest Loss\tAccuracy\tPrecision\tRecall\tF1-Score\tMCC\tCMC\tTest Accuracy\tValid Accuracy\n"
            )
        self.epoch_logs = []

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

        print("Confusion Matrix (Validation):")
        print(cm_valid)
        print("Classification Report (Validation):")
        print(report_valid)
        print("Confusion Matrix (Test):")
        print(cm_test)
        print("Classification Report (Test):")
        print(report_test)

        # Save information to temporary list with values separated by tab
        self.epoch_logs.append(
            (
                epoch + 1,
                logs.get("loss", 0),
                logs.get("val_loss", 0),
                test_loss,
                logs.get("val_accuracy", 0),
                precision_valid,
                recall_valid,
                f1_valid,
                mcc_valid,
                cmc_valid,
                test_accuracy,
                valid_accuracy,
            )
        )

    def on_train_end(self, logs=None):
        # Save information from each epoch to detail file, using tab as separator
        with open(self.detail_file_path, "a") as f:
            for log in self.epoch_logs:
                f.write(
                    f"{log[0]}\t{log[1]:.5f}\t{log[2]:.5f}\t{log[3]:.5f}\t{log[4]:.5f}\t{log[5]:.5f}\t{log[6]:.5f}\t{log[7]:.5f}\t{log[8]:.5f}\t{log[9]:.5f}\t{log[10]:.5f}\t{log[11]:.5f}\n"
                )
        # Calculate summary information
        avg_logs = np.mean(np.array(self.epoch_logs), axis=0)[
            1:
        ]  # Exclude epoch column
        # Write summary file
        with open(self.summary_file_path, "w") as f:
            f.write(
                "Average Train Loss\tAverage Valid Loss\tAverage Test Loss\tAverage Accuracy\tAverage Precision\tAverage Recall\tAverage F1-Score\tAverage MCC\tAverage CMC\tAverage Test Accuracy\tAverage Valid Accuracy\n"
            )
            f.write("\t".join([f"{avg:.5f}" for avg in avg_logs]))


# Define necessary parameters

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 8
EPOCHS = 50
# Create paths to data directories
train_dir = "./images_segmentation/train"
valid_dir = "./images_segmentation/valid"
test_dir = "./images_segmentation/test"

# Load the ResNet50 model pre-trained weights
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(*IMG_SIZE, 3))

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
model = models.Sequential(
    [
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

# Compile the model
model.compile(
    optimizer=Adam(lr=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy", tf.keras.metrics.Precision(), tf.keras.metrics.Recall()],
)

# Data preprocessing and augmentation
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

valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load data
train_data = train_datagen.flow_from_directory(
    train_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
valid_data = valid_datagen.flow_from_directory(
    valid_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
# Đánh giá mô hình trên tập test
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load tập test
test_data = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

detailed_logging_callback = DetailedLoggingCallback(
    valid_data=valid_data, test_data=test_data
)
# Train the model
history = model.fit(
    train_data,
    epochs=EPOCHS,  # Adjust epochs based on your needs
    validation_data=valid_data,
    verbose=1,
    callbacks=[detailed_logging_callback],
)

model.save("./my_resnet50_model.keras")

loaded_model = load_model("./my_resnet50_model.keras")
# đánh giá mô hình
test_results = loaded_model.evaluate(test_data)
test_loss, test_accuracy = test_results[0], test_results[1]
valid_results = loaded_model.evaluate(valid_data)
valid_loss, valid_accuracy = valid_results[0], valid_results[1]
print(f"Test accuracy: {test_accuracy}")
print(f"Valid accuracy: {valid_accuracy}")
print(f"Test loss: {test_loss}")
print(f"Valid loss: {valid_loss}")
results_file_path = "./evaluation_results.txt"
with open(results_file_path, "w") as file:
    file.write(f"Test accuracy: {test_accuracy}\n")
    file.write(f"Valid accuracy: {valid_accuracy}\n")
    file.write(f"Test loss: {test_loss}\n")
    file.write(f"Valid loss: {valid_loss}")

# sử dụng ResNet50
