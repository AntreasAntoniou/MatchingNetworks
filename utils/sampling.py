import scipy.misc
import numpy as np

def unstack(np_array):
    new_list = []
    for i in range(np_array.shape[0]):
        temp_list = np_array[i]
        new_list.append(temp_list)
    return new_list

def sample_generator(num_generations, sess, same_images, inputs, dropout_rate, dropout_rate, data, batch_size,
                     file_name, input_a, training_phase, z_input, z_vectors, save_to_drive=False, gdrive=None):

    input_images, generated = sess.run(same_images, feed_dict={input_a: inputs, dropout_rate: dropout_rate,
                                                                  training_phase: False,
                                                                  z_input: batch_size*[z_vectors[0]]})
    input_images_list = np.zeros(shape=(batch_size, num_generations, input_images.shape[-3], input_images.shape[-2],
                                        input_images.shape[-1]))
    generated_list = np.zeros(shape=(batch_size, num_generations, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))
    height = generated.shape[-3]
    for i in range(num_generations):
        input_images, generated = sess.run(same_images, feed_dict={z_input: batch_size*[z_vectors[i]],
                                                                      input_a: inputs,
                                                                      training_phase: False, dropout_rate:
                                                                      dropout_rate})
        input_images_list[:, i] = input_images
        generated_list[:, i] = generated


    input_images, generated = data.dataset.reconstruct_original(input_images_list), \
                              data.dataset.reconstruct_original(generated_list)

    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)
    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)
    line = np.zeros(shape=(batch_size, 1, generated.shape[-3], generated.shape[-2],
                                     generated.shape[-1]))

    generated = unstack(generated)
    generated = np.concatenate((generated), axis=1)
    generated = unstack(generated)
    generated = np.concatenate((generated), axis=1)

    image = np.concatenate((input_images, generated), axis=1)
    image = np.squeeze(image)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = image * 255
    image = image[:, (num_generations-1)*height:]
    scipy.misc.imsave(file_name, image)
    if save_to_drive == True:
        gdrive.save_in_sample_storage(file_to_save=file_name)

def sample_two_dimensions_generator(sess, same_images, inputs,
                                    dropout_rate, dropout_rate, data,
                                    batch_size, file_name, input_a,
                                    training_phase, z_input, z_vectors, save_to_drive=False, gdrive=None):

    num_generations = z_vectors.shape[0]
    num_gpus, batch_size, im_height, im_width, im_channels = inputs.shape

    generated_list = np.zeros(shape=(batch_size, num_generations, im_height,
                                     im_width,
                                     im_channels))
    height = generated_list.shape[-3]

    for i in range(num_generations):
        _, generated = sess.run(same_images, feed_dict={z_input: batch_size*[z_vectors[i]],
                                                                      input_a: inputs,
                                                                      training_phase: False, dropout_rate:
                                                                      dropout_rate})
        generated_list[:, i] = generated

    canvas_side = int(np.ceil(np.sqrt(num_generations)))
    midpoint = int(np.ceil(canvas_side / 2))
    input_images = np.zeros(shape=(batch_size, canvas_side, im_height, im_width, im_channels))
    input_images[:, midpoint-1] = inputs[0]
    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)
    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)

    # line = np.zeros(shape=(input_images.shape))
    generated = unstack(generated_list)
    generated = np.concatenate((generated), axis=1)
    generated = unstack(generated)
    generated = np.concatenate((generated), axis=1)
    # image = np.concatenate((input_images, line, generated), axis=1)
    
    full_image = np.concatenate((input_images, generated), axis=1)

    for i in range(batch_size):
        single_sample_interpolation = full_image[i * im_height:(i + 1) * im_width]

        single_sample_interpolation = np.array(
            [single_sample_interpolation[:, im_width * sample_idx:im_width * (sample_idx + 1)]
             for sample_idx in range(num_generations)])

        # side, side, height, width, channel
        canvas_image = np.zeros(shape=((canvas_side + 1) * canvas_side, im_height, im_width, im_channels))
        canvas_image[:num_generations] = single_sample_interpolation
        canvas_image = np.reshape(a=canvas_image,
                                  newshape=(canvas_side + 1, canvas_side, im_height, im_width, im_channels))
        canvas_image = unstack(canvas_image)  # side * [side, height, width, channel)
        canvas_image = np.concatenate((canvas_image), axis=1)  # side, side*height, width, channel
        canvas_image = unstack(canvas_image)  # side * [side*height, width, channel]
        canvas_image = np.concatenate((canvas_image), axis=1)
        image = np.squeeze(canvas_image)
        filepath = "{}_{}.png".format(file_name, i)
        scipy.misc.imsave(filepath, image)
        if save_to_drive == True:
            gdrive.save_in_sample_storage(file_to_save=filepath)
#import matplotlib.pyplot as plt
def sample_two_dimensions_test(num_generations, filename="test/test_image"):

    num_gpus, batch_size, im_height, im_width, im_channels = 1, 32, 32, 32, 3

    generated_list = np.zeros(shape=(batch_size, num_generations, im_height,
                                     im_width,
                                     im_channels))
    inputs = np.ones(shape=(1, batch_size, im_height,
                                     im_width,
                                     im_channels))
    height = generated_list.shape[-3]

    for i in range(num_generations):
        generated_list[:, i] = np.random.normal(size=(batch_size, im_height,
                                        im_width,
                                         im_channels), scale=i)

    canvas_side = int(np.ceil(np.sqrt(num_generations)))
    midpoint = int(np.ceil(canvas_side / 2))
    input_images = np.zeros(shape=(batch_size, canvas_side, im_height, im_width, im_channels))
    input_images[:, midpoint - 1] = inputs[0]
    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)
    input_images = unstack(input_images)
    input_images = np.concatenate((input_images), axis=1)

    # line = np.zeros(shape=(input_images.shape))
    generated = unstack(generated_list)
    generated = np.concatenate((generated), axis=1)
    generated = unstack(generated)
    generated = np.concatenate((generated), axis=1)
    # image = np.concatenate((input_images, line, generated), axis=1)

    print(input_images.shape, generated.shape)
    full_image = np.concatenate((input_images, generated), axis=1)
    plt.imshow(full_image)
    plt.show()
    for i in range(batch_size):
        single_sample_interpolation = full_image[i * im_height:(i + 1) * im_height]

        single_sample_interpolation = np.array(
            [single_sample_interpolation[:, im_width * sample_idx:im_width * (sample_idx + 1)]
             for sample_idx in range(num_generations)])

        # side, side, height, width, channel
        canvas_image = np.zeros(shape=((canvas_side + 1) * canvas_side, im_height, im_width, im_channels))
        canvas_image[:num_generations] = single_sample_interpolation
        canvas_image = np.reshape(a=canvas_image,
                                  newshape=(canvas_side + 1, canvas_side, im_height, im_width, im_channels))
        canvas_image = unstack(canvas_image)  # side * [side, height, width, channel)
        canvas_image = np.concatenate((canvas_image), axis=1)  # side, side*height, width, channel
        canvas_image = unstack(canvas_image)  # side * [side*height, width, channel]
        canvas_image = np.concatenate((canvas_image), axis=1)
        image = np.squeeze(canvas_image)
        filepath = "{}_{}.png".format(filename, i)
        scipy.misc.imsave(filepath, image)

# sample_two_dimensions_test(num_generations=15, filename="test")


