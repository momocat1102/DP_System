from library import *
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# ----------------------------生成資料集-------------------------------
# train_generator
def create_train_generator(path, batch_size = 16, target_size = (224, 224), from_dataframe=False, dataframe = None):
    tf.random.set_seed(151416)
    train_datagen = ImageDataGenerator( 
                                    rescale=1/255,
                                    validation_split=0.1,
                                    rotation_range=15,
                                    width_shift_range=[-0.12,0.12],
                                    zoom_range= 0.1,
                                    height_shift_range=[-0.12,0.12],
                                    shear_range= 0.1,
                                    horizontal_flip=True,
                                    fill_mode="constant", cval=0
                                    ) # fill_mode : {"constant", "nearest", "reflect" or "wrap"}
    test_datagen = ImageDataGenerator(
                                      rescale=1/255,
                                      validation_split=0.1
                                      )
    if from_dataframe:
        train_generator = train_datagen.flow_from_dataframe(dataframe,
                                                            seed=132130,
                                                            x_col='img',
                                                            y_col='labels',
                                                            directory=path,
                                                            target_size=target_size,
                                                            batch_size=batch_size,
                                                            interpolation="bilinear",
                                                            #classes=classes,
                                                            #save_to_dir="asd",
                                                            #class_mode=None
                                                            subset="training"
                                                            )
        test_generator = test_datagen.flow_from_dataframe(dataframe,
                                                          x_col='img',
                                                          y_col='labels',
                                                          directory=path,
                                                          seed=132130,
                                                          target_size=target_size,
                                                          batch_size=batch_size,
                                                          interpolation="bilinear",
                                                          subset="validation"
                                                          )
        # print(train_generator.class_indices)
        # print(test_generator.class_indices)
    else:
        train_generator = train_datagen.flow_from_directory(path,
                                                            seed=132130,
                                                            target_size=target_size,
                                                            batch_size=batch_size,
                                                            interpolation="bilinear",
                                                            #classes=classes,
                                                            #save_to_dir="asd",
                                                            #class_mode=None
                                                            subset="training"
                                                            )
        test_generator = test_datagen.flow_from_directory(path,
                                                        seed=132130,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        interpolation="bilinear",
                                                        subset="validation"
                                                        )
    return train_generator, test_generator

# test_generator
def create_test_generator(path, set_classes=None, batch_size=32, target_size=(224, 224)):
    test_datagen = ImageDataGenerator(rescale=1/255,                    
                                    rotation_range=12,
                                    width_shift_range=[-0.12,0.12],
                                    height_shift_range=[-0.12,0.12],
                                    shear_range= 0.1,
                                    zoom_range=0.1,
                                    horizontal_flip=True,
                                    fill_mode="constant", cval=0
                                    ) #fill_mode : {"constant", ")
    test_generator = test_datagen.flow_from_directory(path,
                                                    target_size=target_size,
                                                    batch_size=batch_size,
                                                    interpolation="bilinear",
                                                    #save_to_dir = "./asd",
                                                    shuffle=False)  
    return test_generator 

def img_train_dataset(directory, batch_size=32, image_size=(224, 224), validation_split=0.1, seed=123):
    '''
    to build train dataset,  label_mode is categorical, color_mode is rgb
    '''

    train_dataset = tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",                  # {None, list/tuple of integer labels}
        label_mode="categorical",           # {int, binary, None}
        class_names=None,                   # Only valid if "labels" is "inferred", list, to control order
        color_mode="rgb",                   # {"grayscale", "rgba"}
        batch_size=batch_size,  
        image_size=image_size,  
        shuffle=True,   
        seed=seed,                          # Optional random seed for shuffling and transformations.
        validation_split=validation_split,
        subset="training",                # Only valid if validation_split is not None, One of "training" or "validation"
        interpolation="bilinear",           # {nearest, bicubic, area, lanczos3, lanczos5, gaussian, mitchellcubic}
        crop_to_aspect_ratio=False          # 使圖像不失真
    )
    if validation_split != None:
        valid_dataset = tf.keras.utils.image_dataset_from_directory(
            directory,
            labels="inferred",                  # {None, list/tuple of integer labels}
            label_mode="categorical",           # {int, binary, None}
            class_names=None,                   # Only valid if "labels" is "inferred", list, to control order
            color_mode="rgb",                   # {"grayscale", "rgba"}
            batch_size=batch_size,  
            image_size=image_size,  
            shuffle=True,   
            seed=seed,                          # Optional random seed for shuffling and transformations.
            validation_split=validation_split,
            subset="validation",                # Only valid if validation_split is not None, One of "training" or "validation"
            interpolation="bilinear",           # {nearest, bicubic, area, lanczos3, lanczos5, gaussian, mitchellcubic}
            crop_to_aspect_ratio=False          # 使圖像不失真
        )
        return train_dataset, valid_dataset
    else:
        return train_dataset

def preprocess_image(image, label):
    # 将图片归一化到 [0, 1] 区间内
    image = tf.cast(image, tf.float32) / 255.0
    # label = tf.cast(label, tf.float32) / 255.0
    return image, label

def dataset_to_x_y(dataset, is_categorical=True):
    y_true = []
    x_true = []
    for x, y in dataset:
        y_true.append(y)
        x_true.append(x)
    y_true = np.concatenate(y_true, axis=0)
    x_true = np.concatenate(x_true, axis=0)
    if is_categorical:
        y_true = np.argmax(y_true, axis=-1)

    return x_true, y_true

def data_visualization(dataset, class_names=None):
    x_true, y_true = dataset_to_x_y(dataset)
    print(x_true.shape, y_true.shape)
    data_y = pd.DataFrame({"label": y_true})
    # print(y_true.shape, x_true.shape)
    if class_names == None:
        class_names = dataset.class_names
    # print(class_names)

    # 設定 視覺化風格
    plt.style.use("tableau-colorblind10")
    # 以下程式碼從全域性設定字型為SimHei（黑體），解決顯示中文問題【Windows】
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    # 解決中文字型下座標軸負數的負號顯示問題
    plt.rcParams["axes.unicode_minus"] = False

    # plot the num of every class
    Label = data_y.groupby(['label']).size().to_list()
    plt.grid(zorder=0)
    plt.barh(class_names, Label)
    plt.show()

    # plot top 25 images in dataset
    plt.figure(figsize=(10,10))
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(x_true[i].astype("uint8"))
        plt.xlabel(class_names[y_true[i]])
    plt.show()
    
def visualize_data_for_dataset(dataset, class_names, num_samples=8, img_in_one_line=4):
    if num_samples % img_in_one_line != 0:
        raise ValueError("num_samples must be divisible by img_in_one_line")
    fig, axs = plt.subplots(int(num_samples / img_in_one_line), img_in_one_line, figsize=(12, 6))
    axs = axs.flatten()
    for i, (x, y) in enumerate(dataset.take(num_samples)):
        print(x.shape, y.shape)
        y = y[0]
        axs[i].imshow(x[0].numpy().astype("uint8"))
        axs[i].set_title(f"Class: {class_names[tf.argmax(y).numpy()]}")
        axs[i].axis("off")
    plt.show()

# train_dataset, valid_dataset = img_train_dataset('img')
# train_dataset, valid_dataset = create_train_generator("img")
# print(type(train_dataset))
# train_dataset.prefetch(buffer_size=32)