from model_lib import *
from pre_processing import *
from sklearn.metrics import classification_report

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test.astype('float32')
    class_names = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]

    # 將標籤數據轉換成 one-hot 編碼形式
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

    # 生成資料集
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(batch_size=32)

    # 資料前處理
    test_dataset = test_dataset.map(preprocess_image)

    # 建立模型
    model = resnet_model(size=(32, 32, 3), classes=10)
    model.load_weights("save/model.h5")

    # 預測
    y_pred = model.predict(test_dataset)
    y_pred = np.argmax(y_pred, axis=-1)
    print(classification_report(y_test, y_pred, target_names=class_names))

    plot_confusion_matrix(y_test, y_pred)



if __name__ == "__main__":
    main()