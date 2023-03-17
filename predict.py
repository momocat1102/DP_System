from model_lib import *
from pre_processing import *

train_dataset, valid_dataset = img_train_dataset('img')
target_names = valid_dataset.class_names
train_dataset = train_dataset.map(preprocess_image)
valid_dataset = valid_dataset.map(preprocess_image)
x_true, y_true = dataset_to_x_y(valid_dataset)


model = get_model(classes=10)
model.load_weights('save/model.tf')
y_pred = model.predict(valid_dataset)
y_pred = np.argmax(y_pred, axis=-1)
print(y_pred.shape, y_true.shape)
print(target_names)

from sklearn.metrics import classification_report

# 假设 y_true 和 y_pred 是测试集上的真实标签和模型预测标签

print(classification_report(y_true, y_pred, target_names=target_names))


# 使用mnist的預測結果，y_true和y_predict
# cnf_matrix = plot_confusion_matrix(y_true, y_pred)
