import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.callbacks import Callback

# Định nghĩa các thông số cần thiết
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 8
EPOCHS = 50
# Giả sử đây là các siêu tham số bạn muốn bao gồm
algorithm_name = "CNN"
optimizer = "Adam"
learning_rate = 0.001

# Lấy thời gian hiện tại
current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


class DetailedLoggingCallback(Callback):
    def __init__(self, valid_data, file_prefix="CNN_optAdam_lr0.001_bs32"):
        super(DetailedLoggingCallback, self).__init__()
        self.valid_data = valid_data
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.detail_file_path = f"{file_prefix}_{current_time}_details.txt"
        self.summary_file_path = f"{file_prefix}_{current_time}_summary.txt"
        # Khởi tạo file và viết tiêu đề với tab làm dấu phân cách
        with open(self.detail_file_path, "w") as f:
            f.write(
                "Epoch\tTrain Loss\tValid Loss\tAccuracy\tPrecision\tRecall\tF1-Score\n"
            )
        self.epoch_logs = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        y_pred = np.argmax(self.model.predict(self.valid_data), axis=1)
        y_true = self.valid_data.classes
        precision = precision_score(y_true, y_pred, average="macro")
        recall = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")

        # Lưu thông tin vào list tạm thời với các giá trị được phân cách bằng tab
        self.epoch_logs.append(
            (
                epoch + 1,
                logs.get("loss"),
                logs.get("val_loss"),
                logs.get("val_accuracy"),
                precision,
                recall,
                f1,
            )
        )

    def on_train_end(self, logs=None):
        # Lưu thông tin từ mỗi epoch vào file chi tiết, sử dụng tab làm dấu phân cách
        with open(self.detail_file_path, "a") as f:
            for log in self.epoch_logs:
                f.write(
                    f"{log[0]}\t{log[1]:.5f}\t{log[2]:.5f}\t{log[3]:.5f}\t{log[4]:.5f}\t{log[5]:.5f}\t{log[6]:.5f}\n"
                )
        # Tính toán thông tin tổng hợp
        avg_logs = np.mean(np.array(self.epoch_logs), axis=0)[1:]  # Bỏ qua cột epoch
        # Ghi file tổng hợp
        with open(self.summary_file_path, "w") as f:
            f.write(
                "Average Train Loss\tAverage Valid Loss\tAverage Accuracy\tAverage Precision\tAverage Recall\tAverage F1-Score\n"
            )
            f.write("\t".join([f"{avg:.5f}" for avg in avg_logs]))


# Tạo các đường dẫn đến các thư mục chứa dữ liệu
train_dir = "./images_segmentation/train"
test_dir = "./images_segmentation/test"
valid_dir = "./images_segmentation/valid"

# Tạo data generators cho tập huấn luyện
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

# Tạo data generators cho tập kiểm thử và validation
test_datagen = ImageDataGenerator(rescale=1.0 / 255)
valid_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_datagen.flow_from_directory(
    test_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)

valid_data = valid_datagen.flow_from_directory(
    valid_dir, target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode="categorical"
)
detailed_logging_callback = DetailedLoggingCallback(valid_data=valid_data)
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

# Compile mô hình với các metric mới
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
model.save("./CNN_optAdam_lr0.001_bs32.keras")

# Load the model from the correct path
loaded_model = models.load_model("./CNN_optAdam_lr0.001_bs32.keras")

# Sủ dụng CNN - Deep learning
