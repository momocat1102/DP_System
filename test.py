import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from pre_processing import *

from tensorflow.keras import layers

# IMG_SIZE = 180
# batch_size = 32
# AUTOTUNE = tf.data.AUTOTUNE


# # (train_datasets, val_ds, test_ds), metadata = tfds.load(
# #     'tf_flowers',
# #     split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
# #     with_info=True,
# #     as_supervised=True,
# # )
# # print(type(train_datasets))
# train_datasets, val_ds = img_train_dataset('img')

# def resize_and_rescale(image, label):
#   image = tf.cast(image, tf.float32)
#   image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
#   image = (image / 255.0)
#   return image, label

# def augment(image_label, seed):
#   image, label = image_label
#   image, label = resize_and_rescale(image, label)
#   image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6)
#   # Make a new seed.
#   new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
#   # Random crop back to the original size.
#   image = tf.image.stateless_random_crop(
#       image, size=[IMG_SIZE, IMG_SIZE, 3], seed=seed)
#   # Random brightness.
#   image = tf.image.stateless_random_brightness(
#       image, max_delta=0.5, seed=new_seed)
#   image = tf.clip_by_value(image, 0, 1)
#   return image, label

# # 1
# counter = tf.data.experimental.Counter()
# train_ds = tf.data.Dataset.zip((train_datasets, (counter, counter)))
# # test_ds = tf.data.Dataset.zip((test_ds, (counter, counter)))
# # val_ds = tf.data.Dataset.zip((val_ds, (counter, counter)))
# # 2


# train_ds = (
#     train_ds
#     .shuffle(1000)
#     .map(augment, num_parallel_calls=AUTOTUNE)
#     .batch(batch_size)
#     .prefetch(AUTOTUNE)
# )

# val_ds = (
#     val_ds
#     .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
#     .batch(batch_size)
#     .prefetch(AUTOTUNE)
# )

# # test_ds = (
# #     test_ds
# #     .map(resize_and_rescale, num_parallel_calls=AUTOTUNE)
# #     .batch(batch_size)
# #     .prefetch(AUTOTUNE)
# # )
# # import glob
# # import tensorflow as tf

# # def get_image_path():
# #     train_image_path=glob.glob('./img/*/*.jpg')
# #     test_image_path=glob.glob('./data/dc_2000/test/*/*.jpg')
# #     train_image_label=[1 if elem.split('\\')[1]=='dog' else 0 for elem in train_image_path]
# #     test_image_label=[1 if elem.split('\\')[1]=='dog' else 0 for elem in train_image_path]
# #     train_image_label=tf.reshape(train_image_label,[1])
# #     test_image_label=tf.reshape(test_image_label,[1])
	
# # def load_process_file(filepath):
# #     image = tf.io.read_file(filepath)
# #     image = tf.image.decode_jpeg(image,channels=3)
# #     image = tf.image.resize(image,[256,256])
# #     image=tf.cast(image,tf.float32)/255
# #     # image = tf.image.convert_image_dtype #次函数会将图片格式转化为float32，并执行归一化，如果原数据类型是float32，则不会进行数据归一化的操作


# # train_image_path = "img"
# # train_ds=tf.data.Dataset.from_tensor_slices((train_image_path,train_image_label))
# # train_ds.map(load_process_file,num_parallel_calls=tf.data.experimental.AUTOTUNE) #使用多线程，线程数自适应
# # # test_ds = tf.data.Dataset.from_tensor_slices((test_image_path, test_image_label))
# # # test_ds.map(load_process_file, num_parallel_calls=tf.data.experimental.AUTOTUNE)  # 使用多线程，线程数自适应
# # BATCH_SIZE=32
# # train_count=len(train_image_path)
# # # test_count=len(test_image_path)
# # train_ds=train_ds.repeat().shuffle(train_count).batch(BATCH_SIZE)
# # # test_ds=test_ds.batch(BATCH_SIZE)
# # train_ds=train_ds.prefetch(tf.data.experimental.AUTOTUNE)
# # # test_ds=test_ds.prefetch(tf.data.experimental.AUTOTUNE)

# # imgs,labels =next(iter(train_ds))
# from tensorflow import keras
# import tensorflow as tf
# from tensorflow.keras import layers

# # Create a data augmentation stage with horizontal flipping, rotations, zooms
# HEIGHT = 224
# WITH = 224
# SEED = 123
# data_augmentation = keras.Sequential(
#     [
#         # tf.keras.layers.RandomCrop(height=HEIGHT, width=WITH, seed=SEED),
#         tf.keras.layers.RandomFlip(
#             mode="horizontal",              # {"vertical", "horizontal_and_vertical"}
#             seed=SEED),
#         tf.keras.layers.RandomTranslation(
#             height_factor=0.2,  
#             width_factor=0.2, 
#             fill_mode="reflect",            # {"constant", "wrap", "nearest"}
#             interpolation="bilinear",       # {"nearest"}
#             seed=SEED,
#             fill_value=0.0), 
#         tf.keras.layers.RandomRotation(
#             factor=0.2,
#             fill_mode="reflect",            # {"constant", "wrap", "nearest"}
#             interpolation="bilinear",       # {"nearest"}
#             seed=SEED,
#             fill_value=0.0),
#         tf.keras.layers.RandomZoom(
#             height_factor=0.2,
#             width_factor=None,
#             fill_mode="reflect",            # {"constant", "wrap", "nearest"}
#             interpolation="bilinear",       # {"nearest"}
#             seed=SEED,
#             fill_value=0.0),
#         # tf.keras.layers.RandomHeight(
#         #     factor=0.2,
#         #     interpolation="bilinear",       # {"nearest", "bicubic", "area", "lanczos3", "lanczos5", "gaussian", "mitchellcubic"}
#         #     seed=SEED),
#         # tf.keras.layers.RandomWidth(
#         #     factor=0.2,
#         #     interpolation="bilinear",       # {"nearest", "bicubic", "area", "lanczos3", "lanczos5", "gaussian", "mitchellcubic"}
#         #     seed=SEED),
#         tf.keras.layers.RandomContrast(
#             factor=0.2,                     # (x - mean) * factor + mean
#             seed=SEED),
#     ]
# )

# train_dataset, valid_dataset = img_train_dataset('img')

# import tensorflow as tf
# import tensorflow_hub as hub

# num_classes = 10

# 使用 hub.KerasLayer 组件待训练模型
# new_model = tf.keras.Sequential([
#     inputs = Input(shape)
#     hub.KerasLayer("https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4", trainable=False),
#     tf.keras.layers.Dense(num_classes, activation='softmax')
# ])
# new_model.build([None, 299, 299, 3])

# # 输出模型结构
# new_model.summary()


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers

# data = np.array([[0.1, 0.2, 0.3], [0.8, 0.9, 1.0], [1.5, 1.6, 1.7],])
# layer = layers.Normalization()
# layer.adapt(data)
# normalized_data = layer(data)

# print("Features mean: %.2f" % (normalized_data.numpy().mean()))
# print("Features std: %.2f" % (normalized_data.numpy().std()))


import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)

batch_size = 32
img_height = 180
img_width = 180

train_ds = tf.keras.utils.image_dataset_from_directory(
    "img",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "img",
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
  for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

plt.show()


import numpy as np
import tensorflow as tf
from tensorflow import keras

# 設定參數
batch_size = 32
img_height = 224
img_width = 224
data_dir = '/path/to/your/data/'
mask_dir = '/path/to/your/mask/'
validation_split = 0.2

# 讀取圖片和遮罩，並將它們轉換為 numpy 數組
x = []
y = []
for filename in os.listdir(data_dir):
    img = keras.preprocessing.image.load_img(
        os.path.join(data_dir, filename), target_size=(img_height, img_width)
    )
    img = keras.preprocessing.image.img_to_array(img)
    x.append(img)
    
    mask = keras.preprocessing.image.load_img(
        os.path.join(mask_dir, filename), target_size=(img_height, img_width)
    )
    mask = keras.preprocessing.image.img_to_array(mask)
    mask = np.mean(mask, axis=-1)  # 將遮罩轉換為灰度圖
    mask = mask.astype('uint8')
    y.append(mask)

x = np.array(x)
y = np.array(y)

# 將圖片和遮罩分成訓練集和驗證集
num_samples = len(x)
indices = np.arange(num_samples)
np.random.shuffle(indices)
split = int(num_samples * validation_split)
train_indices = indices[split:]
val_indices = indices[:split]
x_train, y_train = x[train_indices], y[train_indices]
x_val, y_val = x[val_indices], y[val_indices]

# 將圖片和遮罩組成訓練集和驗證集的 tf.data.Dataset 對象
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(batch_size).repeat()

val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(batch_size).repeat()

# 創建 Unet 模型並進行訓練
inputs = keras.layers.Input(shape=(img_height, img_width, 3))
x = inputs

# 編碼器部分
downsamples = []
for filters in [64, 128, 256, 512]:
    x = keras.layers.Conv2D(filters, 3, strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Conv2
