```python
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.applications import vgg16
from keras.applications import mobilenet
from keras.optimizers import Adam, SGD
from keras.metrics import categorical_crossentropy
from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from keras.models import Model
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
%matplotlib inline

```


```python
from google.colab import drive
drive.mount('/content/drive')

train_path = '/content/drive/MyDrive/datasets/data/dogs_and_cats/train'
valid_path = '/content/drive/MyDrive/datasets/data/dogs_and_cats/valid'
test_path = '/content/drive/MyDrive/datasets/data/dogs_and_cats/test'

train_batches = ImageDataGenerator().flow_from_directory(train_path, target_size=(224,224), batch_size=10)
valid_batches = ImageDataGenerator().flow_from_directory(valid_path, target_size=(224,224),batch_size=30)
test_batches = ImageDataGenerator().flow_from_directory(test_path, target_size=(224,224),batch_size=50, shuffle=False)

```

    Mounted at /content/drive
    Found 202 images belonging to 2 classes.
    Found 103 images belonging to 2 classes.
    Found 451 images belonging to 2 classes.



```python
base_model = vgg16.VGG16(weights="imagenet", include_top=False,input_shape=(224,224,3))
for layer in base_model.layers:
    layer.trainable = False

last_layer = base_model.get_layer("block5_pool")
last_output = last_layer.output

x = Flatten()(last_output)
x = Dense(64,activation = "relu")(x)
x = BatchNormalization()(x)
x = Dropout(0.5)(x)
x = Dense(2,activation = "softmax")(x)

new_model = Model(inputs=base_model.input, outputs=x)

new_model.summary()
```

    Model: "model_3"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     input_4 (InputLayer)        [(None, 224, 224, 3)]     0         
                                                                     
     block1_conv1 (Conv2D)       (None, 224, 224, 64)      1792      
                                                                     
     block1_conv2 (Conv2D)       (None, 224, 224, 64)      36928     
                                                                     
     block1_pool (MaxPooling2D)  (None, 112, 112, 64)      0         
                                                                     
     block2_conv1 (Conv2D)       (None, 112, 112, 128)     73856     
                                                                     
     block2_conv2 (Conv2D)       (None, 112, 112, 128)     147584    
                                                                     
     block2_pool (MaxPooling2D)  (None, 56, 56, 128)       0         
                                                                     
     block3_conv1 (Conv2D)       (None, 56, 56, 256)       295168    
                                                                     
     block3_conv2 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_conv3 (Conv2D)       (None, 56, 56, 256)       590080    
                                                                     
     block3_pool (MaxPooling2D)  (None, 28, 28, 256)       0         
                                                                     
     block4_conv1 (Conv2D)       (None, 28, 28, 512)       1180160   
                                                                     
     block4_conv2 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_conv3 (Conv2D)       (None, 28, 28, 512)       2359808   
                                                                     
     block4_pool (MaxPooling2D)  (None, 14, 14, 512)       0         
                                                                     
     block5_conv1 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv2 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_conv3 (Conv2D)       (None, 14, 14, 512)       2359808   
                                                                     
     block5_pool (MaxPooling2D)  (None, 7, 7, 512)         0         
                                                                     
     flatten_1 (Flatten)         (None, 25088)             0         
                                                                     
     dense_6 (Dense)             (None, 64)                1605696   
                                                                     
     batch_normalization_3 (Batc  (None, 64)               256       
     hNormalization)                                                 
                                                                     
     dropout_3 (Dropout)         (None, 64)                0         
                                                                     
     dense_7 (Dense)             (None, 2)                 130       
                                                                     
    =================================================================
    Total params: 16,320,770
    Trainable params: 1,605,954
    Non-trainable params: 14,714,816
    _________________________________________________________________



```python
new_model.compile(Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

new_model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=2,epochs=20, verbose=2)
```

    Epoch 1/20


    /usr/local/lib/python3.10/dist-packages/keras/optimizers/legacy/adam.py:117: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      super().__init__(name, **kwargs)
    <ipython-input-15-d3f88811f438>:3: UserWarning: `Model.fit_generator` is deprecated and will be removed in a future version. Please use `Model.fit`, which supports generators.
      new_model.fit_generator(train_batches, steps_per_epoch=4, validation_data=valid_batches, validation_steps=2,epochs=20, verbose=2)


    4/4 - 51s - loss: 1.3599 - accuracy: 0.5250 - val_loss: 1.7523 - val_accuracy: 0.6500 - 51s/epoch - 13s/step
    Epoch 2/20
    4/4 - 58s - loss: 0.5970 - accuracy: 0.6750 - val_loss: 1.4224 - val_accuracy: 0.7167 - 58s/epoch - 14s/step
    Epoch 3/20
    4/4 - 41s - loss: 0.3723 - accuracy: 0.9000 - val_loss: 0.6876 - val_accuracy: 0.7667 - 41s/epoch - 10s/step
    Epoch 4/20
    4/4 - 37s - loss: 0.3581 - accuracy: 0.8438 - val_loss: 0.6825 - val_accuracy: 0.7833 - 37s/epoch - 9s/step
    Epoch 5/20
    4/4 - 57s - loss: 0.5087 - accuracy: 0.7000 - val_loss: 0.4652 - val_accuracy: 0.8500 - 57s/epoch - 14s/step
    Epoch 6/20
    4/4 - 56s - loss: 0.4861 - accuracy: 0.8750 - val_loss: 0.4500 - val_accuracy: 0.8500 - 56s/epoch - 14s/step
    Epoch 7/20
    4/4 - 58s - loss: 0.2614 - accuracy: 0.8750 - val_loss: 0.3202 - val_accuracy: 0.8833 - 58s/epoch - 14s/step
    Epoch 8/20
    4/4 - 57s - loss: 0.2272 - accuracy: 0.9000 - val_loss: 0.2335 - val_accuracy: 0.8833 - 57s/epoch - 14s/step
    Epoch 9/20
    4/4 - 58s - loss: 0.2380 - accuracy: 0.9000 - val_loss: 0.2264 - val_accuracy: 0.8833 - 58s/epoch - 15s/step
    Epoch 10/20
    4/4 - 57s - loss: 0.2418 - accuracy: 0.9250 - val_loss: 0.1797 - val_accuracy: 0.9333 - 57s/epoch - 14s/step
    Epoch 11/20
    4/4 - 48s - loss: 0.1278 - accuracy: 0.9750 - val_loss: 0.2600 - val_accuracy: 0.9000 - 48s/epoch - 12s/step
    Epoch 12/20
    4/4 - 40s - loss: 0.0774 - accuracy: 0.9750 - val_loss: 0.3299 - val_accuracy: 0.8833 - 40s/epoch - 10s/step
    Epoch 13/20
    4/4 - 57s - loss: 0.2434 - accuracy: 0.9250 - val_loss: 0.1822 - val_accuracy: 0.9167 - 57s/epoch - 14s/step
    Epoch 14/20
    4/4 - 40s - loss: 0.1638 - accuracy: 0.9000 - val_loss: 0.2498 - val_accuracy: 0.8833 - 40s/epoch - 10s/step
    Epoch 15/20
    4/4 - 57s - loss: 0.1250 - accuracy: 0.9250 - val_loss: 0.3479 - val_accuracy: 0.8667 - 57s/epoch - 14s/step
    Epoch 16/20
    4/4 - 53s - loss: 0.1771 - accuracy: 0.9062 - val_loss: 0.3905 - val_accuracy: 0.8500 - 53s/epoch - 13s/step
    Epoch 17/20
    4/4 - 41s - loss: 0.1733 - accuracy: 0.9500 - val_loss: 0.3286 - val_accuracy: 0.8500 - 41s/epoch - 10s/step
    Epoch 18/20
    4/4 - 40s - loss: 0.1236 - accuracy: 0.9000 - val_loss: 0.3008 - val_accuracy: 0.8833 - 40s/epoch - 10s/step
    Epoch 19/20
    4/4 - 41s - loss: 0.0804 - accuracy: 0.9750 - val_loss: 0.2636 - val_accuracy: 0.9000 - 41s/epoch - 10s/step
    Epoch 20/20
    4/4 - 61s - loss: 0.1681 - accuracy: 0.9250 - val_loss: 0.3174 - val_accuracy: 0.8667 - 61s/epoch - 15s/step





    <keras.callbacks.History at 0x7f853d3edae0>




```python
new_model.evaluate(test_batches)
```

    10/10 [==============================] - 181s 18s/step - loss: 0.1977 - accuracy: 0.9180





    [0.1976603865623474, 0.9179601073265076]




```python

```
