from pre_processing import *
from model_lib import *


def main():
    # 生成資料集
    train_dataset, test_dataset = img_train_dataset('img', 1)
    class_names = train_dataset.class_names # 類別名稱
    # data_visualization(train_dataset) # 資料視覺化

    # 切分資料集
    num_samples = len(train_dataset)
    val_size = int(num_samples * 0.1)
    train_dataset = train_dataset.skip(val_size)
    valid_dataset = train_dataset.take(val_size)
    print("Train dataset size:", train_dataset.cardinality().numpy())
    print("Validation dataset size:", valid_dataset.cardinality().numpy())
    
    # 資料擴增
    data_aug = data_augmentation()
    augmented_dataset = train_dataset.repeat(10).map(lambda x, y: (data_aug(x), y))
    train_dataset = train_dataset.concatenate(augmented_dataset)
    train_dataset = train_dataset.map(lambda x, y: (tf.squeeze(x, axis=0), tf.squeeze(y, axis=0)))
    train_dataset = train_dataset.shuffle(1000).batch(32)
    print("augmentation train dataset size:", train_dataset.cardinality().numpy())

    # 擴增資料視覺化
    visualize_data_for_dataset(train_dataset, class_names, num_samples=8, img_in_one_line=4)
    # 資料前處理
    train_dataset = train_dataset.map(preprocess_image)
    valid_dataset = valid_dataset.map(preprocess_image)
    # 建立模型
    model = get_model(classes=10)
    model.summary()
    tf.keras.utils.plot_model(model, "model.png", show_shapes=True)
    # 訓練模型
    model = train_model(model, train_dataset, valid_dataset)
    # 預測
    y_pred = model.predict(valid_dataset)
    y_pred = np.argmax(y_pred, axis=-1)
    _, y_true = dataset_to_x_y(valid_dataset)
    plot_confusion_matrix(y_true, y_pred)

    

if __name__ == '__main__':
    main()