# en este módulo definimos la interfaz para interactuar con el código
# del clasificador.
# Deseamos realizar dos tareas distintas:
# - dada una foto, decir quién es la persona.
# - dada una carpeta con fotos, reentrenar el modelo con las nuevas fotos.


def eval_face(input_dir):
    """
    function that recognizes the face on a picture.
    input: location of the picture. (str)
    output: predicted label of the picture. (str)
    errors:
    + picture_name does not exist/is not and image - throw type error
    + no positive match to any of the labels - throw nonexistent error
    """
    pass


def add_face(input_dir):
    """
    function that retrains the model to add a new face.
    input: location of the images. (str)
    output: error code, 0 otherwise. (exception, none)
    errors:
    + pictures_folder is not a folder - throw type error
    + pictures_folder contains a non-image - fail silently and continue
      for the rest
    + pictures_folder does not have [enough] pictures - throw exception and
      exit
    + pictures_folder contains pictures that are not from the same people -
      this kills the model.
    """
    pass


def new_model(input_dir):
    """
    function that generates a new model from a folder full of pictures.
    """
    pass
