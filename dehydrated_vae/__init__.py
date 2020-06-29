from keras.layers import Dense, Lambda, Input
from keras.models import Model
from keras.losses import mean_squared_error
import keras.backend as K

def build_vae(sub_encoder, decoder, latent_size, base_loss=mean_squared_error, kl_scale=1.):
  encoder_inputs = Input(sub_encoder.input.shape[1:])
  x = encoder_inputs
  x = sub_encoder(x)
  z_mean = Dense(latent_size)(x)
  z_log_sigma = Dense(latent_size)(x)

  def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_size),
      mean=0., stddev=1.)
    return z_mean + K.exp(z_log_sigma) * epsilon

  z = Lambda(sampling, output_shape=(latent_size,))([z_mean, z_log_sigma])

  encoder = Model(encoder_inputs, z_mean)

  x = decoder(z)
  vae = Model(encoder.inputs, x)

  def loss(x, x_decoded_mean):
    recon_loss = base_loss(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=-1)
    kl_loss = K.mean(kl_loss)
    return recon_loss + kl_loss * kl_scale

  return [vae, encoder, decoder, loss]
