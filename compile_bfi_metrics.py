
import os
from datetime import datetime
import pandas as pd
import xlrd

"""
pip install xlrd
pip install Jinja
"""

def creating_clean_df_using_excel():
    bfi_subfolder = 'bfi_data'
    folder_location = os.path.join(os.getcwd(), bfi_subfolder)
    new_folder_location = os.path.join(os.getcwd(), bfi_subfolder.replace('data', 'data_refined'))

    month_dict = {'jan-': 'january-', 'feb-': 'february-', 'mar-': 'march-', 'apr-': 'april-', 'may-': 'may-', 'jun-': 'june-', 'jul-': 'july-',
    'aug-': 'august-', 'sep-': 'september-', 'oct-': 'october', 'nov-': 'november', 'dec-': 'december'}

    for file_name in os.listdir(folder_location):
        try:
            # check if the file is an .xls file
            print('processing: ' + str(file_name))
            if file_name.endswith('.xls'):
                # TODO - see if the file content is consistent across each type:
                # case 1 - file name contains 'bfi-uk-box-office' (e.g. bfi-uk-box-office-5-7-April-2013)
                if 'bfi-uk-box-office' in file_name:
                    # loading important information into a df
                    file_path = os.path.join(folder_location, file_name)
                    box_office_df = pd.read_excel(file_path, skiprows=[0])
                    box_office_df = box_office_df[
                        ['Title', 'Weekend Gross', 'Number of cinemas', 'Site average', 'Distributor']]

                    # creating a df for the broken down box office
                    subset_box_office_df = box_office_df.iloc[0:15, :]
                    # creating a df for the total box office
                    total_box_office_df = box_office_df.iloc[15:16, :]

                    # making sure all months not abbreviated:
                    for i, j in month_dict.items():
                        file_name = file_name.replace(i, j)

                    # renaming first half of file:
                    new_file_name = file_name.replace('bfi-uk-box-office', 'bfi_box_office').replace('xls', 'csv')

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

                    # loading dataframe to renamed csv
                    total_new_file_name = 't_%s' % new_file_name
                    subset_box_office_df.to_csv(os.path.join(new_folder_location, new_file_name), encoding='utf-8', sep=',')
                    total_box_office_df.to_csv(os.path.join(new_folder_location, total_new_file_name), encoding='utf-8', sep=',')

                    # os.rename(os.path.join(folder_location, file_name), os.path.join(folder_location, new_file_name))

                # case 2 - file name contains 'bfi-weekend-box-office-report' (e.g. bfi-weekend-box-office-report-01-03-august-2014)
                elif 'bfi-weekend-box-office-report' in file_name:
                    # loading important information into a df
                    file_path = os.path.join(folder_location, file_name)
                    box_office_df = pd.read_excel(file_path, skiprows=[0])
                    try:
                        box_office_df = box_office_df[['Title', 'Weekend Gross', 'Number of cinemas', 'Site average', 'Distributor']]
                    except:
                        box_office_df = box_office_df[['Film', 'Weekend Gross', 'Number of cinemas', 'Site average', 'Distributor']]
                        box_office_df = box_office_df.rename({'Film': 'Title'})

                    # creating a df for the broken down box office
                    subset_box_office_df = box_office_df.iloc[0:15, :]
                    # creating a df for the total box office
                    total_box_office_df = box_office_df.iloc[15:16, :]


                    # making sure all months not abbreviated:
                    for i, j in month_dict.items():
                        file_name = file_name.replace(i, j)

                    # renaming first half of file:
                    new_file_name = file_name.replace('bfi-weekend-box-office-report', 'bfi_box_office').replace('xls', 'csv')

                    # renaming second half of file:
                    # original string
                    original_date = "-".join(file_name.split('-')[5:]).split('.')[0]

                    # a check for file names which are of a different naming convention
                    # eg. bfi-weekend-box-office-report-2017-09-01-03
                    if len(str(file_name.split('-')[5])) == 4:
                        if len(file_name.split('-')[5:]) == 4:
                            date_list = file_name.split('-')[5:]
                            del date_list[2]
                            modified_date = "-".join(date_list).split('.')[0]

                            # convert to datetime object
                            new_date = datetime.strptime(modified_date, '%Y-%m-%d')

                            # format as string with leading zeros for day
                            new_date = new_date.strftime('%d-%m-%Y')

                            new_file_name = new_file_name.replace(original_date, new_date)
                        # e.g. bfi-weekend-box-office-report-2017-03-31-04-02
                        elif len(file_name.split('-')[5:]) == 5:
                            date_list = file_name.split('-')[5:]
                            del date_list[1:2]
                            modified_date = "-".join(date_list).split('.')[0]

                            # convert to datetime object
                            new_date = datetime.strptime(modified_date, '%Y-%m-%d')

                            # format as string with leading zeros for day
                            new_date = new_date.strftime('%d-%m-%Y')

                            new_file_name = new_file_name.replace(original_date, new_date)
                    # e.g. bfi-weekend-box-office-report-31-october-2-November-2014
                    elif len(file_name.split('-')[6:]) > 4:
                        modified_date = "-".join(file_name.split('-')[7:]).split('.')[0]

                        # convert to datetime object
                        new_date = datetime.strptime(modified_date, '%d-%B-%Y')

                        # format as string with leading zeros for day
                        new_date = new_date.strftime('%d-%m-%Y')

                        new_file_name = new_file_name.replace(original_date, new_date)
                    # e.g. bfi-weekend-box-office-report-29-31-january-2015
                    else:
                        modified_date = "-".join(file_name.split('-')[6:]).split('.')[0]

                        # convert to datetime object
                        new_date = datetime.strptime(modified_date, '%d-%B-%Y')

                        # format as string with leading zeros for day
                        new_date = new_date.strftime('%d-%m-%Y')

                        new_file_name = new_file_name.replace(original_date, new_date)

                    # loading dataframe to renamed csv
                    total_new_file_name = 't_%s' % new_file_name
                    subset_box_office_df.to_csv(os.path.join(new_folder_location, new_file_name), encoding='utf-8', sep=',')
                    total_box_office_df.to_csv(os.path.join(new_folder_location, total_new_file_name), encoding='utf-8', sep=',')
                    # os.rename(os.path.join(folder_location, file_name), os.path.join(folder_location, new_file_name))

                # case 3 - file name contains 'uk-film-council-box-office-report' (e.g. uk-film-council-box-office-report-aug-5-aug-7-2011)
                elif 'uk-film-council-box-office-report' in file_name:
                    # loading important information into a df
                    file_path = os.path.join(folder_location, file_name)
                    box_office_df = pd.read_excel(file_path, skiprows=[0])
                    box_office_df = box_office_df[
                        ['Title', 'Weekend Gross', 'Number of cinemas', 'Site average', 'Distributor']]

                    # creating a df for the broken down box office
                    subset_box_office_df = box_office_df.iloc[0:15, :]
                    # creating a df for the total box office
                    total_box_office_df = box_office_df.iloc[15:16, :]

                    # making sure all months not abbreviated:
                    for i, j in month_dict.items():
                        file_name = file_name.replace(i, j)

                    # renaming first half of file:
                    new_file_name = file_name.replace('uk-film-council-box-office-report', 'bfi_box_office').replace('xls', 'csv')

                    # renaming second half of file:
                    # original string
                    original_date = "-".join(file_name.split('-')[6:]).split('.')[0]

                    # a check for file names which are of a different naming convention
                    # e.g. uk-film-council-box-office-report-may-23-may-25-2008
                    if len(file_name.split('-')[6:]) > 4:
                        modified_date = "-".join(file_name.split('-')[8:]).split('.')[0]

                        # convert to datetime object
                        new_date = datetime.strptime(modified_date, '%B-%d-%Y')

                        # format as string with leading zeros for day
                        new_date = new_date.strftime('%d-%m-%Y')

                        new_file_name = new_file_name.replace(original_date, new_date)
                    #  e.g. uk-film-council-box-office-report-2007-01
                    elif len(file_name.split('-')[6:]) == 3:
                        modified_date = "-".join(file_name.split('-')[7:]).split('.')[0]

                        # convert to datetime object
                        new_date = datetime.strptime(modified_date, '%B%d-%Y')

                        # format as string with leading zeros for day
                        new_date = new_date.strftime('%d-%m-%Y')

                        new_file_name = new_file_name.replace(original_date, new_date)
                    else:
                        print('no matching format for file: %s' % file_name)

                    # loading dataframe to renamed csv
                    total_new_file_name = 't_%s' % new_file_name
                    subset_box_office_df.to_csv(os.path.join(new_folder_location, new_file_name), encoding='utf-8', sep=',')
                    total_box_office_df.to_csv(os.path.join(new_folder_location, total_new_file_name), encoding='utf-8', sep=',')
                    # os.rename(os.path.join(folder_location, file_name), os.path.join(folder_location, new_file_name))
                else:
                    print('%s could not be categorized' % file_name)
        except Exception as e:
            print('issue loading file %s due to %s' % (file_name, e))
            continue


if __name__ == '__main__':

    creating_clean_df_using_excel()