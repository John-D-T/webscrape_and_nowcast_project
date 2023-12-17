import os


def latex_file_cleanup(file_name):
    """
    Function to remove the non_pdf files following latex file generation
    We look to remove aux, txt, and tex file

    TODO - Look to make this more dynamic in the future
    """
    extension_list = ['aux', 'log', 'tex']

    for extension in extension_list:

        file_name_with_extension = f'{file_name}.{extension}'

        file_path = os.path.join(os.getcwd(), file_name_with_extension)

        os.remove(file_path)

