import os


def latex_file_cleanup(file_name, extension_list=[]):
    """
    Function to remove the non_pdf files following latex file generation
    We look to remove aux, txt, and tex file

    :param file_name: name of the file
    :param extension_list: The extensions of all the variations of the file we want to remove (e.g. [.csv])
    """

    for extension in extension_list:
        file_name_with_extension = f'{file_name}.{extension}'
        file_path = os.path.join(os.getcwd(), file_name_with_extension)
        os.remove(file_path)