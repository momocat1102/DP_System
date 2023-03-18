from library import *

os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
# Create a data augmentation stage with horizontal flipping, rotations, zooms
HEIGHT = 224
WITH = 224
SEED = 123
def data_augmentation():
    data_aug = tf.keras.Sequential(
        [
            # tf.keras.layers.RandomCrop(height=HEIGHT, width=WITH, seed=SEED),
            tf.keras.layers.RandomFlip(
                mode="horizontal",              # {"vertical", "horizontal_and_vertical"}
                seed=SEED),
            tf.keras.layers.RandomTranslation(
                height_factor=0.2,  
                width_factor=0.2, 
                fill_mode="reflect",            # {"constant", "wrap", "nearest"}
                interpolation="bilinear",       # {"nearest"}
                seed=SEED,
                fill_value=0.0), 
            tf.keras.layers.RandomRotation(
                factor=0.2,
                fill_mode="reflect",            # {"constant", "wrap", "nearest"}
                interpolation="bilinear",       # {"nearest"}
                seed=SEED,
                fill_value=0.0),
            tf.keras.layers.RandomZoom(
                height_factor=0.2,
                width_factor=None,
                fill_mode="reflect",            # {"constant", "wrap", "nearest"}
                interpolation="bilinear",       # {"nearest"}
                seed=SEED,
                fill_value=0.0),
            # tf.keras.layers.RandomHeight(
            #     factor=0.2,
            #     interpolation="bilinear",       # {"nearest", "bicubic", "area", "lanczos3", "lanczos5", "gaussian", "mitchellcubic"}
            #     seed=SEED),
            # tf.keras.layers.RandomWidth(
            #     factor=0.2,
            #     interpolation="bilinear",       # {"nearest", "bicubic", "area", "lanczos3", "lanczos5", "gaussian", "mitchellcubic"}
            #     seed=SEED),
            tf.keras.layers.RandomContrast(
                factor=0.2,                     # (x - mean) * factor + mean
                seed=SEED),
        ]
    )
    return data_aug

def get_model(size=(224, 224, 3), classes=10):
    url = "https://tfhub.dev/google/imagenet/efficientnet_v2_imagenet1k_s/classification/2"

    inputs = tf.keras.Input(shape=size)
    retrain = hub.KerasLayer(handle=url, trainable=False, output_shape=size)(inputs)


    # representation = GroupNormalization(epsilon=1e-6)(retrain)
    # representation = tf.keras.layers.GlobalAveragePooling2D()(retrain)
    representation = tf.keras.layers.Dropout(0.5)(retrain) 
    # print(representation.shape)

    # Classify outputs.
    activation = tf.keras.layers.Activation(tf.nn.softmax)(representation)
    x = tf.keras.layers.Dense(classes)(activation)
    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

def resnet_model(size=(224, 224, 3), classes=10):
    model = ResNet50(weights='imagenet', include_top=False, input_shape=size)

    x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=model.input, outputs=output)
    return model


def train_model(model, train_dataset, valid_dataset, epochs=10, save_path="save"):
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                            patience=1, mode='auto', verbose=0, cooldown=0,
                            min_lr=1e-7)
    mod_check = tf.keras.callbacks.ModelCheckpoint(filepath=save_path + "/weights.{epoch:02d}-{val_loss:.2f}.hdf5",
                            save_weights_only=True,
                            monitor='val_loss',
                            mode='max',
                            verbose=0,
                            save_best_only=True)
    model.compile(
        optimizer=tf.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
    )
    history = model.fit(
            train_dataset,
            epochs=epochs,
            callbacks=[
                reduce_lr,
                mod_check],
            validation_data=valid_dataset
    )

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    return model

def plot_confusion_matrix(y_true, y_pred):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    # 轉換成百分比
    cnf_matrix=cnf_matrix.astype('float')/cnf_matrix.sum(axis=1)
    # print(cnf_matrix.shape)
    plt.figure(figsize=(10,10))
    plt.imshow(cnf_matrix, cmap='Blues')
    plt.colorbar()
    plt.xticks([i for i in range(10)])
    plt.yticks([i for i in range(10)])

    thresh = cnf_matrix.max() / 2.
    for i in range(10):
        for j in range(10):
            # 繪製文字於座標格中並置中
            plt.text(j, i,format(cnf_matrix[i, j], '.2f')+'%',horizontalalignment="center",color="white" if cnf_matrix[i, j] > thresh else"black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout() # 自動調整子圖參數以指定填充
    plt.show()