

DIR = r"C:\Users\HAZAL\Desktop\facemask_detection-main\dataset"
recognize_set = ["masked", "unmasked"]
print("Images are being loaded...")

listing = []
tagging = []

for i in recognize_set:
    join_dir = os.path.join(DIR, i)
    directories = os.listdir(join_dir)

    for x in directories:
        pictures = os.path.join(join_dir, x)
        pics = load_img(pictures, target_size=(200, 200))
        pics = img_to_array(pics)
        pics = preprocess_input(pics)

        listing.append(pics)
        tagging.append(i)

imgGenerate = ImageDataGenerator(shear_range=0.15, horizontal_flip=True, fill_mode="nearest",
                                 rotation_range=20, zoom_range=0.15, width_shift_range=0.2,
                                 height_shift_range=0.2)

binary = LabelBinarizer()
tagging = binary.fit_transform(tagging)
tagging = to_categorical(tagging)

listing = np.array(listing, dtype="float32")
tagging = np.array(tagging)

baseModel = MobileNetV2(input_tensor=Input(shape=(200, 200, 3)), weights="imagenet", include_top=False)

(train_X, test_X, train_Y, test_Y) = train_test_split(listing, tagging, test_size=0.20, stratify=tagging,
                                                      random_state=42)

create_head = baseModel.output
create_head = AveragePooling2D(pool_size=(7, 7))(create_head)
create_head = Flatten(name="flatten")(create_head)
create_head = Dense(128, activation="relu")(create_head)
create_head = Dropout(0.5)(create_head)
create_head = Dense(2, activation="softmax")(create_head)

actual_model = Model(inputs=baseModel.input, outputs=create_head)

for layer in baseModel.layers:
    layer.trainable = False

learn_rate = 1e-4
epoch = 20
size = 32

print("compiling...")
adam_opt = Adam(lr=learn_rate, decay=learn_rate / epoch)

actual_model.compile(loss="binary_crossentropy", optimizer=adam_opt,
                     metrics=["accuracy"])

print("training...")

HEAD = actual_model.fit(imgGenerate.flow(train_X, train_Y, batch_size=32),
                        steps_per_epoch=len(train_X) // size,
                        validation_data=(test_X, test_Y),
                        validation_steps=len(test_X) // size,
                        epochs=epoch)

print(" completing...")
predictions = actual_model.predict(test_X, batch_size=32)
predictions = np.argmax(predictions, axis=1)
print(classification_report(test_Y.argmax(axis=1), predictions, target_names=binary.classes_))

print("Model is being saved...")
actual_model.save("model_mask.model", save_format="h5")
