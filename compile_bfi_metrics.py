
import os
from datetime import datetime

def rename_excel_files():
    bfi_subfolder = 'bfi_data'
    folder_location = os.path.join(os.getcwd(), bfi_subfolder)

    month_dict = {'feb-': 'february-', 'mar-': 'march-', 'apr-': 'april-', 'may-': 'may-', 'jun-': 'june-', 'jul-': 'july-',
    'aug-': 'august-', 'sep-': 'september-', 'oct-': 'october', 'nov-': 'november', 'dec-': 'december'}

    for file_name in os.listdir(folder_location):

        # check if the file is an .xls file
        if file_name.endswith('.xls'):
            # TODO - see if the file content is consistent across each type:
            # case 1 - file name contains 'bfi-uk-box-office' (e.g. bfi-uk-box-office-5-7-April-2013)
            if 'bfi-uk-box-office' in file_name:
                # todo - might need to edit file contains before renaming

                # making sure all months not abbreviated:
                for i, j in month_dict.items():
                    file_name = file_name.replace(i, j)

                # renaming first half of file:
                new_file_name = file_name.replace('bfi-uk-box-office', 'bfi_box_office')

                # renaming second half of file:
                # original string
                original_date = "-".join(file_name.split('-')[4:]).split('.')[0]

                # a check for files of a different naming convention - e.g. bfi-box-office-30-november-2-december-2012
                if len(file_name.split('-')[5:]) > 4:
                    modified_date = "-".join(file_name.split('-')[6:]).split('.')[0]

                    # convert to datetime object
                    new_date = datetime.strptime(modified_date, '%d-%B-%Y')

                    # format as string with leading zeros for day
                    new_date = new_date.strftime('%d-%m-%Y')

                    new_file_name = new_file_name.replace(original_date, new_date)
                else:
                    modified_date = "-".join(file_name.split('-')[5:]).split('.')[0]

                    # convert to datetime object
                    new_date = datetime.strptime(modified_date, '%d-%B-%Y')

                    # format as string with leading zeros for day
                    new_date = new_date.strftime('%d-%m-%Y')

                    new_file_name = new_file_name.replace(original_date, new_date)

                # renaming file
                os.rename(os.path.join(folder_location, file_name), os.path.join(folder_location, new_file_name))

            # case 2 - file name contains 'bfi-weekend-box-office-report' (e.g. bfi-weekend-box-office-report-01-03-august-2014)
            elif 'bfi-weekend-box-office-report' in file_name:
                # todo - might need to edit file contains before renaming

                # making sure all months not abbreviated:
                for i, j in month_dict.items():
                    file_name = file_name.replace(i, j)

                # renaming first half of file:
                new_file_name = file_name.replace('bfi-weekend-box-office-report', 'bfi_box_office')

                # renaming second half of file:
                # original string
                original_date = "-".join(file_name.split('-')[5:]).split('.')[0]

                # a check for file names which are of a different naming convention -  e.g. uk-film-council-box-office-report-2007-01
                if len(file_name.split('-')[6:]) > 3:
                    modified_date = "-".join(file_name.split('-')[6:]).split('.')[0]

                    # convert to datetime object
                    new_date = datetime.strptime(modified_date, '%d-%B-%Y')

                    # format as string with leading zeros for day
                    new_date = new_date.strftime('%d-%m-%Y')

                    new_file_name = new_file_name.replace(original_date, new_date)

                # renaming file
                os.rename(os.path.join(folder_location, file_name), os.path.join(folder_location, new_file_name))

            # case 3 - file name contains 'uk-film-council-box-office-report' (e.g. uk-film-council-box-office-report-aug-5-aug-7-2011)
            elif 'uk-film-council-box-office-report' in file_name:
                # todo - might need to edit file contains before renaming

                # making sure all months not abbreviated:
                for i, j in month_dict.items():
                    file_name = file_name.replace(i, j)

                # renaming first half of file:
                new_file_name = file_name.replace('uk-film-council-box-office-report', 'bfi_box_office')

                # renaming second half of file:
                # original string
                original_date = "-".join(file_name.split('-')[6:]).split('.')[0]

                if len(file_name.split('-')[5:]) > 4:
                    modified_date = "-".join(file_name.split('-')[6:]).split('.')[0]

                    # convert to datetime object
                    new_date = datetime.strptime(modified_date, '%d-%B-%Y')

                    # format as string with leading zeros for day
                    new_date = new_date.strftime('%d-%m-%Y')

                    new_file_name = new_file_name.replace(original_date, new_date)
                else:
                    modified_date = "-".join(file_name.split('-')[8:]).split('.')[0]

                    # convert to datetime object
                    new_date = datetime.strptime(modified_date, '%B-%d-%Y')

                    # format as string with leading zeros for day
                    new_date = new_date.strftime('%d-%m-%Y')

                    new_file_name = new_file_name.replace(original_date, new_date)

                # renaming file
                os.rename(os.path.join(folder_location, file_name), os.path.join(folder_location, new_file_name))
            else:
                print('%s could not be categorized' % file_name)


def combine_and_clean_bfi_csv():
    pass

if __name__ == '__main__':

    rename_excel_files()
    #combine_and_clean_bfi_csv()