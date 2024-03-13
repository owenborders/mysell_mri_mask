class Architectures():
    def hybrid_dilated_model(self,activ_func : str, width : ,height) -> tf.keras.models.Model:
        """
        Model that combines dilated convolutional layers with maxpooling layers.

        Parameters
        ---------------
        activ_func : str
            activation function 
        width: int
            width of input and output
        height : int
            height of input and output

        Returns
        --------------
        model : tf.keras.models.Model
            The complete model defined in the function.

        """
        
        def dilation_block(input):
            output = layers.Conv2D(64, (3, 3), dilation_rate=1, padding='same')(input)
            output = layers.BatchNormalization()(output)
            output = layers.Activation(activ_func)(output)
            for dilation_rate in range(1, 16, 1):
                print(dilation_rate)
                filter_size = 64
                output = layers.Conv2D(filter_size, (3, 3),\
                dilation_rate=dilation_rate, padding='same')(output)
                output = layers.BatchNormalization()(output)
                output = layers.Activation(activ_func)(output)
            return output

        def max_pooling_block(input):
            output = layers.Conv2D(64, (3, 3), dilation_rate=1, padding='same')(input)
            output = layers.BatchNormalization()(output)
            output = layers.Activation(activ_func)(output)
            output = layers.concatenate([output,dilated_output])
            output = layers.MaxPooling2D((2, 2))(output)
            output = layers.Conv2D(64, (3, 3), dilation_rate=1, padding='same')(output)
            output = layers.BatchNormalization()(output)
            output = layers.Activation(activ_func)(output)
            output = layers.Conv2D(64, (3, 3), dilation_rate=1, padding='same')(output)
            output = layers.BatchNormalization()(output)
            output = layers.Activation(activ_func)(output)
            output = layers.Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(output)
            output = layers.BatchNormalization()(output)
            output = layers.Activation(activ_func)(output)
            output = layers.Conv2D(64, (3, 3), dilation_rate=1, padding='same')(output)
            output = layers.BatchNormalization()(output)
            output = layers.Activation(activ_func)(output)

            return output

        def conv_block(input,filter_size):
            output = layers.Conv2D(64,filter_size, dilation_rate=1, padding='same')(input)
            output = layers.BatchNormalization()(output)
            output = layers.Activation(activ_func)(output)
            output = layers.concatenate([output,dilated_output,max_pooling_output])
            output = layers.Conv2D(128,filter_size, dilation_rate=1, padding='same')(output)
            output = layers.BatchNormalization()(output)
            output = layers.Activation(activ_func)(output)
            output = layers.Conv2D(64,filter_size, dilation_rate=1, padding='same')(output)
            output = layers.BatchNormalization()(output)
            output = layers.Activation(activ_func)(output)

            return output

        inputs = layers.Input(shape=(width, height, 1))

        dilated_output = dilation_block(inputs)
        max_pooling_output = max_pooling_block(inputs)
        small_filter_output = conv_block(inputs,(3,3))
        merged_outputs = layers.concatenate(\
        [small_filter_output,dilated_output,max_pooling_output])

        final_layer = layers.Conv2D(64, (3, 3), padding='same')(merged_outputs)

        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(final_layer)
        model = models.Model(inputs=[inputs], outputs=[outputs])
        return model

