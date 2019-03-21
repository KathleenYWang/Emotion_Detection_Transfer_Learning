# Emotion Detection with Transfer Learning

### Overview
Accurately identifying human emotions is a challenging task even for humans. With the help of deep neural networks, significant progress has been made in training algorithms to identify emotions. Here, we explore using transfer learning to retrain and refine complex deep neural networks with pre-trained weights.

### Data
Extensive research has been done in this area and increasing amount of data have been used by researchers.
In this project, we trained our base model on the FER 2013 dataset and further evaluated the results agaisnt the SFEE DATASET.

### Tested Models

### 1 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
and a logistic layer -- for this example we have 7 classes
predictions = Dense(7, activation='softmax')(x)
this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)

train_datagen = ImageDataGenerator(
rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

Epoch 1/100
1795/1795 [==============================] - 950s 529ms/step - loss: 1.7152 - acc: 0.3039

Epoch 00001: acc improved from -inf to 0.30376, saving model to inception_v3
Epoch 2/100
1795/1795 [==============================] - 950s 529ms/step - loss: 1.7091 - acc: 0.3057

Epoch 00002: acc improved from 0.30376 to 0.30567, saving model to inception_v3
Epoch 3/100
1795/1795 [==============================] - 949s 529ms/step - loss: 1.7059 - acc: 0.3101

Epoch 00003: acc improved from 0.30567 to 0.31020, saving model to inception_v3
Epoch 4/100
1795/1795 [==============================] - 948s 528ms/step - loss: 1.7053 - acc: 0.3151

Epoch 00004: acc improved from 0.31020 to 0.31480, saving model to inception_v3
Epoch 5/100
1795/1795 [==============================] - 951s 530ms/step - loss: 1.6946 - acc: 0.3209

Epoch 00005: acc improved from 0.31480 to 0.32107, saving model to inception_v3
Epoch 6/100
1795/1795 [==============================] - 953s 531ms/step - loss: 1.6950 - acc: 0.3195

Epoch 00006: acc did not improve from 0.32107
Epoch 7/100
1795/1795 [==============================] - 953s 531ms/step - loss: 1.6931 - acc: 0.3167

Epoch 00007: acc did not improve from 0.32107
Epoch 8/100
1795/1795 [==============================] - 952s 530ms/step - loss: 1.7003 - acc: 0.3206

Epoch 00008: acc did not improve from 0.32107
Epoch 9/100
1795/1795 [==============================] - 955s 532ms/step - loss: 1.6873 - acc: 0.3205

Epoch 00009: acc did not improve from 0.32107
Epoch 10/100
1795/1795 [==============================] - 959s 534ms/step - loss: 1.6898 - acc: 0.3228

Epoch 00010: acc improved from 0.32107 to 0.32295, saving model to inception_v3
Epoch 11/100
1795/1795 [==============================] - 956s 533ms/step - loss: 1.6954 - acc: 0.3220

Epoch 00011: acc did not improve from 0.32295
Epoch 12/100
1795/1795 [==============================] - 957s 533ms/step - loss: 1.6839 - acc: 0.3259

Epoch 00012: acc improved from 0.32295 to 0.32591, saving model to inception_v3
Epoch 13/100
1795/1795 [==============================] - 956s 533ms/step - loss: 1.6887 - acc: 0.3247

Epoch 00013: acc did not improve from 0.32591
Epoch 14/100
1795/1795 [==============================] - 953s 531ms/step - loss: 1.6957 - acc: 0.3194

Epoch 00014: acc did not improve from 0.32591
Epoch 15/100
1795/1795 [==============================] - 952s 531ms/step - loss: 1.6919 - acc: 0.3220

Epoch 00015: acc did not improve from 0.32591
Epoch 16/100
1795/1795 [==============================] - 960s 535ms/step - loss: 1.6890 - acc: 0.3194

Epoch 00016: acc did not improve from 0.32591
Epoch 17/100
1795/1795 [==============================] - 965s 538ms/step - loss: 1.6851 - acc: 0.3287

Epoch 00017: acc improved from 0.32591 to 0.32867, saving model to inception_v3
Epoch 18/100
1795/1795 [==============================] - 957s 533ms/step - loss: 1.6934 - acc: 0.3258

Epoch 00018: acc did not improve from 0.32867
Epoch 19/100
1795/1795 [==============================] - 956s 533ms/step - loss: 1.6831 - acc: 0.3249

Epoch 00019: acc did not improve from 0.32867
Epoch 20/100
1795/1795 [==============================] - 957s 533ms/step - loss: 1.6879 - acc: 0.3284

Epoch 00020: acc did not improve from 0.32867
Epoch 21/100
1795/1795 [==============================] - 958s 533ms/step - loss: 1.6934 - acc: 0.3249

Epoch 00021: acc did not improve from 0.32867
Epoch 22/100
1795/1795 [==============================] - 963s 537ms/step - loss: 1.6818 - acc: 0.3305

Epoch 00022: acc improved from 0.32867 to 0.33055, saving model to inception_v3
Epoch 23/100
1795/1795 [==============================] - 955s 532ms/step - loss: 1.6910 - acc: 0.3255

Epoch 00023: acc did not improve from 0.33055
Epoch 24/100
1795/1795 [==============================] - 956s 532ms/step - loss: 1.6815 - acc: 0.3302

Epoch 00024: acc did not improve from 0.33055
Epoch 25/100
1795/1795 [==============================] - 951s 530ms/step - loss: 1.6896 - acc: 0.3281

Epoch 00025: acc did not improve from 0.33055
Epoch 26/100
1795/1795 [==============================] - 952s 531ms/step - loss: 1.6869 - acc: 0.3291

Epoch 00026: acc did not improve from 0.33055
Epoch 27/100
1795/1795 [==============================] - 951s 530ms/step - loss: 1.6915 - acc: 0.3258

Epoch 00027: acc did not improve from 0.33055
Epoch 28/100
1795/1795 [==============================] - 951s 530ms/step - loss: 1.6880 - acc: 0.3274

Epoch 00028: acc did not improve from 0.33055
Epoch 29/100
1795/1795 [==============================] - 958s 534ms/step - loss: 1.6826 - acc: 0.3261

Epoch 00029: acc did not improve from 0.33055
Epoch 30/100
1795/1795 [==============================] - 956s 533ms/step - loss: 1.6927 - acc: 0.3229

Epoch 00030: acc did not improve from 0.33055
Epoch 31/100
1795/1795 [==============================] - 956s 533ms/step - loss: 1.6879 - acc: 0.3286

Epoch 00031: acc did not improve from 0.33055
Epoch 32/100
1795/1795 [==============================] - 953s 531ms/step - loss: 1.6848 - acc: 0.3290

Epoch 00032: acc did not improve from 0.33055
Epoch 00032: early stopping


### 1 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
### 2 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
and a logistic layer -- for this example we have 7 classes
x = model.output
x = Flatten()(x)
x = Dropout(0.5)(x)
x = Dense(1024, activation="relu")(x)
predictions = Dense(7, activation='softmax')(x)
this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)




Epoch 1/100
1795/1795 [==============================] - 448s 250ms/step - loss: 1.8905 - acc: 0.2376

Epoch 00001: acc improved from -inf to 0.23768, saving model to inception_v3
Epoch 2/100
1795/1795 [==============================] - 442s 246ms/step - loss: 1.8162 - acc: 0.2502

Epoch 00002: acc improved from 0.23768 to 0.25015, saving model to inception_v3
Epoch 3/100
1795/1795 [==============================] - 446s 248ms/step - loss: 1.8105 - acc: 0.2507

Epoch 00003: acc improved from 0.25015 to 0.25071, saving model to inception_v3
Epoch 4/100
1795/1795 [==============================] - 445s 248ms/step - loss: 1.8070 - acc: 0.2504

Epoch 00004: acc did not improve from 0.25071
Epoch 5/100
1795/1795 [==============================] - 444s 247ms/step - loss: 1.7996 - acc: 0.2507

Epoch 00005: acc improved from 0.25071 to 0.25081, saving model to inception_v3
Epoch 6/100
1795/1795 [==============================] - 444s 248ms/step - loss: 1.7968 - acc: 0.2507

Epoch 00006: acc did not improve from 0.25081
Epoch 7/100
1795/1795 [==============================] - 445s 248ms/step - loss: 1.7925 - acc: 0.2547

Epoch 00007: acc improved from 0.25081 to 0.25447, saving model to inception_v3
Epoch 8/100
1795/1795 [==============================] - 443s 247ms/step - loss: 1.7923 - acc: 0.2547

Epoch 00008: acc improved from 0.25447 to 0.25450, saving model to inception_v3
Epoch 9/100
1795/1795 [==============================] - 444s 247ms/step - loss: 1.7899 - acc: 0.2550

Epoch 00009: acc improved from 0.25450 to 0.25516, saving model to inception_v3
Epoch 10/100
1795/1795 [==============================] - 444s 248ms/step - loss: 1.7924 - acc: 0.2562

Epoch 00010: acc improved from 0.25516 to 0.25631, saving model to inception_v3
Epoch 11/100
 908/1795 [==============>...............] - ETA: 3:39 - loss: 1.7931 - acc: 0.2541

### 3 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Epoch 1/50
717/717 [==============================] - 47s 65ms/step - loss: 1.8064 - acc: 0.3106 - val_loss: 1.7123 - val_acc: 0.3453

Epoch 00001: val_loss improved from inf to 1.71225, saving model to /data/emotion_models/fer2013_mini_XCEPTION.01-0.35.hdf5
Epoch 2/50
717/717 [==============================] - 42s 59ms/step - loss: 1.5491 - acc: 0.4186 - val_loss: 1.6067 - val_acc: 0.3932

Epoch 00002: val_loss improved from 1.71225 to 1.60671, saving model to /data/emotion_models/fer2013_mini_XCEPTION.02-0.39.hdf5
Epoch 3/50
717/717 [==============================] - 42s 59ms/step - loss: 1.4238 - acc: 0.4713 - val_loss: 1.5781 - val_acc: 0.4310

Epoch 00003: val_loss improved from 1.60671 to 1.57814, saving model to /data/emotion_models/fer2013_mini_XCEPTION.03-0.43.hdf5
Epoch 4/50
717/717 [==============================] - 43s 60ms/step - loss: 1.3481 - acc: 0.4966 - val_loss: 1.4967 - val_acc: 0.4673

Epoch 00004: val_loss improved from 1.57814 to 1.49667, saving model to /data/emotion_models/fer2013_mini_XCEPTION.04-0.47.hdf5
Epoch 5/50
717/717 [==============================] - 42s 59ms/step - loss: 1.2982 - acc: 0.5168 - val_loss: 1.3704 - val_acc: 0.4855

Epoch 00005: val_loss improved from 1.49667 to 1.37040, saving model to /data/emotion_models/fer2013_mini_XCEPTION.05-0.49.hdf5
Epoch 6/50
717/717 [==============================] - 43s 59ms/step - loss: 1.2597 - acc: 0.5280 - val_loss: 1.3118 - val_acc: 0.5162

Epoch 00006: val_loss improved from 1.37040 to 1.31179, saving model to /data/emotion_models/fer2013_mini_XCEPTION.06-0.52.hdf5
Epoch 7/50
717/717 [==============================] - 43s 60ms/step - loss: 1.2297 - acc: 0.5395 - val_loss: 1.3198 - val_acc: 0.5166


### 3 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
