import math
import os 
import tensorflow as tf

RUTA_DATOS_ENTRENAMIENTO = "./DataNous/Caso12/Entrenamiento/*/*.png"
RUTA_DATOS_VALIDACION = "./DataNous/Caso12/Validacion/"
PORCENTAJE_DATOS_VALIDACION = 0.1

if __name__ == "__main__":
    train_data_set = tf.data.Dataset.list_files([RUTA_DATOS_ENTRENAMIENTO], shuffle=True)
    data_set_length = len(list(train_data_set))

    validation_data_set_length = math.ceil(data_set_length * PORCENTAJE_DATOS_VALIDACION)
    validation_data_set = train_data_set.take(validation_data_set_length)

    tf.print("Total datos validación: %d" % validation_data_set_length)

    for element in validation_data_set:
        parts = tf.strings.split(element, os.path.sep)

        folder = parts[-2]
        name = parts[-1]
        full_name = folder + "/" + name

        img = tf.io.read_file(element)
        tf.io.write_file(RUTA_DATOS_VALIDACION + full_name, img)
        tf.io.gfile.remove(element.numpy())