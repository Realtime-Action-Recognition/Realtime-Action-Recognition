# ==============================================================================================
# Title:   three_stream_model.py
# Contact: realtimeactionrecognition@gmail.com
# ==============================================================================================

# ==============================================================================================
#
#   Points to Note:
#   1. This sample code uses DenseNet121, whereas the final model uses DenseNet-BC 
#      (101 layers)
#   2. The image size considered here is (224, 224)
#   3. This is done for the sake of code portability, should the paper be accepted, 
#      the entire code will be released.
#
# ==============================================================================================

import numpy as np
# Import to load pre-trained model
from keras.applications.densenet import DenseNet121

# Imports for callbacks during training
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Imports for the layers used
from keras.layers import Concatenate, Flatten, Dense
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Import to create the model
from keras.models import Model

# Import Optimizers for the model
from keras.optimizers import Adam, SGD

# Import for real-time plotting of the Training Loss and Training Accuracy
from livelossplot.keras import PlotLossesCallback

# Save in current Folder by default
model_save_path = "."



class ThreeStreamModel:
    '''
    Class to provide an interface to the to the three-stream algorithm
    used for action recognition in real-time
    '''

    # Constructor
    def __init__(self, output_classes, input_learning_rate, input_epochs):
        '''
        Initialize the the three-stream model, with a specified number of classes
        
        Parameters:
        @output_classes : Number of output classes
        '''

        self.densenet121_hog_stream = DenseNet121(include_top=False,
                        weights='imagenet', pooling=None, classes=output_classes,
                        input_shape=(224, 224, 3))
        self.densenet121_temporal_stream = DenseNet121(include_top=False, 
                        weights='imagenet',  pooling=None, classes=output_classes,
                        input_shape=(224, 224, 3))
        self.densenet121_spatial_stream = DenseNet121(include_top=False,
                        weights='imagenet', pooling=None, classes=output_classes, 
                        input_shape=(224, 224, 3))

        self.rename_layers()

        concatenated_layers = keras.layers.concatenate([self.densenet121_hog_stream.output, 
                                                        self.densenet121_temporal_stream.output, 
                                                        self.densenet121_spatial_stream.output], 
                                                        axis=1)

        merged_streams_flatten_layer = Flatten()(concatenated_layers)
        merged_streams_first_fc = Dense(64, activation='relu')(merged_streams_flatten_layer)
        merged_streams_softmax = Dense(8, activation='softmax')(merged_streams_first_fc)

        self.fusion_model = Model(inputs=[self.densenet121_hog_stream.input, 
                                     self.densenet121_temporal_stream.input, 
                                     self.densenet121_spatial_stream.input], 
                                     outputs=[merged_streams_softmax])

        model_optimizer = self.adam_optimizer()

        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

        filepath = model_save_path + "/DenseNet3S_8_classes_{epoch:02d}_{val_acc:.2f}.h5"

        self.fusion_model.compile(loss='sparse_categorical_crossentropy', optimizer=model_optimizer, 
                    metrics=['accuracy'])

        mcp_save = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')



    def adam_optimizer(self, learning_rate = 0.00001, beta_1 = 0.9, beta_2 = 0.999, 
                        epsilon = None, decay = 0.0, amsgrad = True):
        '''
        Initialize the the Adam optimizer with the specified parameters.

        Adam calculates an exponential moving average of the gradient and the squared gradient, 
        and the parameters beta1 and beta2 control the decay rates of these moving averages.
        
        Parameters:
        @learning_rate  : The learning rate of the model
        @beta_1         : The exponential decay rate for the first moment estimates
        @beta_2         : The exponential decay rate for the second moment estimates
        @epsilon        : Very small number to prevent the division by 0 exception
        @decay          : Factor of Learning rate decay
        @amsgrad        : Is the AMSGrad variant of Adam being used or not?
        '''
        adam_optimizer = Adam(lr = learning_rate, beta_1 = beta_1, beta_2 = beta_2, 
                        epsilon = None, decay = decay, amsgrad = amsgrad)

        return adam_optimizer



    def rename_layers(self):
        '''
        Function to rename the layers of the three difference streams to allow concatenation.
        
        If the layers of each stream is not renamed, model compilation will not be allowed
        as multiple layers cannot have the same name.
        '''
        for i, layer in enumerate(self.densenet121_hog_stream.layers):
            layer.name = 'hog_' + str(i)

        for i, layer in enumerate(self.densenet121_temporal_stream.layers):
            layer.name = 'tmeporal_' + str(i)

        for i, layer in enumerate(self.densenet121_spatial_stream.layers):
            layer.name = 'spatial_' + str(i)



    def train_model(self, X_train_hog, X_train_temporal, X_train_spatial, trainLabels, 
                X_test_hog, X_test_temporal, X_test_spatial, testLabels, input_epochs,
                input_batch_size):
        '''
        Function to train the model that was built.

        Parameters:
        @X_train_hog     :
        @X_train_temporal:
        @X_train_spatial :
        @trainLabels     :
        @X_test_hog      :
        @X_test_temporal :
        @X_test_spatial  :
        @testLabels      :
        '''
        earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')

        filepath = "UCF_Trained_Model_for_Paper_1929_{epoch:02d}.h5"
        mcp_save = ModelCheckpoint(filepath, save_best_only=True, monitor='val_loss', mode='min')
        
        model_history = self.fusion_model.fit([np.array(X_train_hog), 
                                                np.array(X_train_temporal), 
                                                np.array(X_train_spatial)],
                                                np.array(trainLabels),
                                                validation_data=([np.array(X_test_hog), 
                                                np.array(X_test_temporal), 
                                                np.array(X_test_spatial)], 
                                                np.array(testLabels)),
                                                epochs=input_epochs, 
                                                callbacks=[earlyStopping, mcp_save],
                                                batch_size=input_batch_size)



    def test_model(self, X_test_hog, X_test_temporal, X_test_spatial, testLabels):
        '''
        Function to train the model that was built.

        Parameters:
        @X_test_hog      :
        @X_test_temporal :
        @X_test_spatial  :
        @testLabels      :
        '''
        results = self.fusion_model.evaluate([np.array(X_test_hog), 
                                            np.array(X_test_temporal),
                                            np.array(X_test_spatial)],
                                            testLabels,
                                            batch_size=32)
        print('Model:', model_name, 'Test Loss:', results[0], 'Test Accuracy:', results[1])



    def test_pretrained_model(self, model_path, X_test_hog, X_test_temporal, X_test_spatial, testLabels):
        '''
        Function to train the model that was built.

        Parameters:
        @X_test_hog      :
        @X_test_temporal :
        @X_test_spatial  :
        @testLabels      :
        '''
        model_to_test = load_model(model_path)
        results = model_to_test.evaluate([np.array(X_test_hog),
                                np.array(X_test_temporal),
                                np.array(X_test_spatial)],
                                testLabels,
                                batch_size=32)
        print('Model:', model_name, 'Test Loss:', results[0], 'Test Accuracy:', results[1])