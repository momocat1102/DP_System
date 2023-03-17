from pre_processing import *
from model_lib import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32')
    # class_names = train_dataset.class_names # 類別名稱
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

    # 將標籤數據轉換成 one-hot 編碼形式
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)

    # 生成資料集
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=10000, seed=1234)
    train_dataset = train_dataset.batch(batch_size=1)

    data_visualization(train_dataset, class_names) # 資料視覺化

    # 切分資料集
    num_samples = len(train_dataset)
    val_size = int(num_samples * 0.1)
    train_dataset = train_dataset.skip(val_size)
    valid_dataset = train_dataset.take(val_size)
    print("Train dataset size:", train_dataset.cardinality().numpy())
    print("Validation dataset size:", valid_dataset.cardinality().numpy())
    
    # 資料擴增
    data_aug = data_augmentation()
    augdata_times = 9
    augmented_dataset = train_dataset.repeat(augdata_times).map(lambda x, y: (data_aug(x), y))
    train_dataset = train_dataset.concatenate(augmented_dataset)
    train_dataset = train_dataset.map(lambda x, y: (tf.squeeze(x, axis=0), tf.squeeze(y, axis=0)))
    print("augmentation train dataset size:", train_dataset.cardinality().numpy())
    train_dataset = train_dataset.shuffle(1000).batch(32)

    # 擴增資料視覺化
    visualize_data_for_dataset(train_dataset, class_names, num_samples=8, img_in_one_line=4)

    # 資料前處理
    train_dataset = train_dataset.map(preprocess_image)
    valid_dataset = valid_dataset.map(preprocess_image)

    # 建立模型
    model = resnet_model(size=(32, 32, 3), classes=10)
    # model = get_model(size=(32, 32, 3), classes=10)
    model.summary()
    tf.keras.utils.plot_model(model, "model.png", show_shapes=True)

    # 訓練模型
    model = train_model(model, train_dataset, valid_dataset, epochs=40, save_path="save")

    # 預測
    y_pred = model.predict(valid_dataset)
    y_pred = np.argmax(y_pred, axis=-1)
    _, y_true = dataset_to_x_y(valid_dataset)
    plot_confusion_matrix(y_true, y_pred)

    

if __name__ == '__main__':
    main()