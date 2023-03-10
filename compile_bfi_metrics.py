
import os

def rename_excel_files():
    bfi_subfolder = '/bfi_data'
    folder_location = os.path.join(os.getcwd(), bfi_subfolder)

    for file_name in os.listdir(folder_location):

        # check if the file is an .xls file
        if file_name.endswith('.xls'):
            # case 1 - file name contains 'bfi-uk-box-office' (e.g. bfi-uk-box-office-5-7-April-2013)
            if 'bfi-uk-box-office' in file_name:

            # case 2 - file name contains 'bfi-weekend-box-office-report' (e.g. bfi-weekend-box-office-report-01-03-august-2014)
            elif 'bfi-weekend-box-office-report' in file_name:
                pass
            # case 3 - file name contains 'uk-film-council-box-office-report' (e.g. uk-film-council-box-office-report-aug-5-aug-7-2011)
            elif 'uk-film-council-box-office-report' in file_name:
                pass
            else:
                print('%s could not be categorized' % file_name)


def combine_and_clean_bfi_csv():


if __name__ == '__main__':

    rename_excel_files()
    #combine_and_clean_bfi_csv()