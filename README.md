# 

### Tested Models

# 1 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
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


# 1 $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$