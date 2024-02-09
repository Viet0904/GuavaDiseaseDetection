import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import ResNet50
import os
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


# Define necessary parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 8
EPOCHS = 100

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class DetailedLoggingCallback(Callback):
    def __init__(
        self,
        valid_data,
        test_data,
        file_prefix="ResNet50_segmentation_optAdam_lr0.001_bs32",
    ):
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
        with open(f"{self.Confusion_Matrix_path}_test.txt ", "w") as f:
            f.write("Confusion Matrix Test\n")
        with open(f"{self.Confusion_Matrix_path}_valid.txt ", "w") as f:
            f.write("Confusion Matrix Valid\n")
        with open(f"{self.report_path}_test.txt", "w") as f:
            f.write("Classification Report Test\n")
        with open(f"{self.report_path}_valid.txt", "w") as f:
            f.write("Classification Report Valid\n")
        self.epoch_logs = []
        self.epoch_cm_logs = []
        self.epoch_report = []

    def extract_labels(self, dataset):
        all_labels = []
        for images, labels in dataset:
            all_labels.append(labels.numpy())
        return np.concatenate(all_labels, axis=0)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        test_results = self.model.evaluate(self.test_data, verbose=0)
        test_loss, test_accuracy = test_results[0], test_results[1]
        y_pred_valid_probs = self.model.predict(self.valid_data)
        y_pred_valid = np.argmax(y_pred_valid_probs, axis=1)
        y_true_valid = self.extract_labels(self.valid_data)
        y_true_valid = np.argmax(y_true_valid, axis=1)
        cm_valid = confusion_matrix(y_true_valid, y_pred_valid)
        report_valid = classification_report(
            y_true_valid, y_pred_valid, digits=5, output_dict=True
        )
        precision_valid = precision_score(y_true_valid, y_pred_valid, average="macro")
        recall_valid = recall_score(y_true_valid, y_pred_valid, average="macro")
        f1_valid = f1_score(y_true_valid, y_pred_valid, average="macro")
        mcc_valid = matthews_corrcoef(y_true_valid, y_pred_valid)
        cmc_valid = cohen_kappa_score(y_true_valid, y_pred_valid)

        # Dự đoán nhãn cho dữ liệu test
        y_pred_test_probs = self.model.predict(self.test_data)
        y_pred_test = np.argmax(y_pred_test_probs, axis=1)
        y_true_test = self.extract_labels(self.test_data)
        y_true_test = np.argmax(y_true_test, axis=1)
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
        # Lưu confusion matrix và classification report vào file tương ứng
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


def load_and_preprocess_image(img_path, label_path):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, IMG_SIZE)
    img /= 255.0  # Normalize to [0,1]

    label = tf.io.read_file(label_path)
    label = tf.strings.split(tf.expand_dims(label, axis=-1), sep=" ")
    label = tf.strings.to_number(label.values[0], out_type=tf.int32)
    label = tf.one_hot(label, NUM_CLASSES)  # Áp dụng one-hot encoding

    return img, label


def create_dataset(img_dir, label_dir):
    img_paths = [os.path.join(img_dir, fname) for fname in os.listdir(img_dir)]
    label_paths = [
        os.path.join(label_dir, fname.replace(".jpg", ".txt"))
        for fname in os.listdir(img_dir)
    ]

    dataset = tf.data.Dataset.from_tensor_slices((img_paths, label_paths))
    dataset = dataset.map(lambda img, label: load_and_preprocess_image(img, label))

    return dataset


# Tạo datasets
train_data = create_dataset("./images/train/images", "./images/train/labels")
train_data = train_data.shuffle(1000).batch(32)

test_data = create_dataset("./images/test/images", "./images/test/labels")
test_data = test_data.batch(32)

valid_data = create_dataset("./images/valid/images", "./images/valid/labels")
valid_data = valid_data.batch(32)

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
model.save(f"./resnet50_segmentation_{current_time}.keras")

loaded_model = load_model(f"./resnet50_segmentation_{current_time}.keras")


# sử dụng ResNet50
