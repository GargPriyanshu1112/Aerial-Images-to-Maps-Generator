# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, Input
from tensorflow.keras.models import Model

from generator_model import define_generator
from discriminator_model import define_discriminator


def define_gan(inp_shape):
    # Get the Generator model
    g_model = define_generator(inp_shape)
    # Get the Discriminator model
    d_model = define_discriminator(inp_shape)

    # Make weights in the discriminator not trainable
    for layer in d_model.layers:
        if not isinstance(layer, BatchNormalization):
            layer.trainable = False
    
    src_img = Input(shape=inp_shape)
    # Supply source image as input to the generator and receive the generated image
    gen_img = g_model(src_img)
    # Supply source and generated image as inputs to the discriminator
    dis_output = d_model([src_img, gen_img])
    # Instantiate the model with source image as input, fake image generated
    # by the generator and discriminator output (0 or 1) as as outputs
    gan_model = Model(inputs=src_img, outputs=[dis_output, gen_image])

    # Compile the gan model
    gan_model.compile(loss=["binary_crossentropy", "mae"],
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.999),
                      loss_weights=[1, 100]) # Authors suggested weighting BCE vs L1 as 1:100

    return g_model, d_model, gan_model
