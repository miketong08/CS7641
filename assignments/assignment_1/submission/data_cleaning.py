# python3
# coding: utf-8

"""
CS7641 - Data cleaning scripts for Assignment 1

Mike Tong


Created: JAN 2019
"""


from pathlib import Path

import pandas as pd

from ancillary import prep_housing, prep_student


def create_dummies(df, cols):
	"""Converts a list of categorical columns into dummy variables
	"""

	df_dummies = pd.concat([pd.get_dummies(df[col]) for col in cols], axis=1)
	df_c = pd.concat([df, df_dummies], axis=1)
	df_c.drop(columns=cols, axis=1, inplace=True)

	return df_c

def clean_housing_data(DATA_DIR, save_csv=False, prep=True):
	"""Cleans the Housing Data CSV files from DATA_DIR and optionally saves it as a csv
	:param DATA_DIR: Path to the directory with all csvs
	:param save_csv: Path to save location of the cleaned dataframe as a csv, will not save if not a string

	:return: A cleaned and prepared pandas.DataFrame for analysis
	"""

	paths = Path(DATA_DIR)

	df_address = pd.read_csv(next(paths.glob('raw_address*')))
	df_residential = pd.read_csv(next(paths.glob('raw_residential*')))

	df_address.drop(columns=['POLDIST', 'ROC', 'PSA'], axis=1, inplace=True)
	df_residential.drop(columns=['GIS_LAST_MOD_DTTM'], axis=1, inplace=True)

	df_address_column_stats = df_address.nunique().to_frame('n_unique').join(
	    df_address.isna().sum().to_frame('nans')
	    )
	df_address_cols = [i for i,v in df_address_column_stats.iterrows() if v['n_unique'] < 40 and v['nans'] < 4000]  # ambiguious thresholds
	df_address_cols.append('SSL')  # join col
	df_address_c = df_address[df_address_cols]

	df_residential_c = df_residential.dropna(subset=['PRICE'])  # target columns
	df_residential_c['YR_RMDL'] = df_residential['YR_RMDL'].fillna(-1).astype(int)

	sale_date_split = df_residential_c['SALEDATE'].str.split('-', expand=True)  # split yr and mo
	df_residential_c['YR_SALE'] = sale_date_split[0].astype(int)
	df_residential_c['MO_SALE'] = sale_date_split[1].astype(int)
	df_residential_c = df_residential_c.drop(columns=['SALEDATE'], axis=1)

	has_D = [i.split('_D')[0] for i in df_residential_c.columns if i.endswith('_D')]  # string of integer representation
	df_residential_cols = [i for i in df_residential_c.columns if i not in has_D]
	df_residential_cols.append('SSL') # join column

	df_residential_c = df_residential_c[df_residential_cols]

	df_m = df_address_c.join(df_residential_c, lsuffix='SSL', rsuffix='SSL')
	df_m = df_m.drop(columns='SSLSSL', axis=1)

	df_m_categorical = [i for i,v in df_m.dtypes.iteritems() if v == object]  # track categoricals to convert

	df_m = create_dummies(df_m, df_m_categorical)

	df_m.dropna(inplace=True)
	df_m.drop(columns=['OBJECTID'],
        axis=1, inplace=True)

	if prep:
		df_m = prep_housing(df_m)

	if type(save_csv) == str:
		df_m.to_csv(save_csv, index=False)

	return df_m

def clean_student_data(DATA_PATH, save_csv=False, prep=True):
	"""Cleans the StudentPerformance CSV file from DATA_DIR and optionally saves it as a csv
	:param DATA_DIR: Path to the directory with all csvs
	:param save_csv: Path to save location of the cleaned dataframe as a csv, will not save if not a string

	:return: A cleaned and prepared pandas.DataFrame for analysis
	"""

	paths = Path(DATA_PATH)
	df_m = pd.read_csv(next(paths.glob('StudentsPerformance.csv')))
	df_categorical = [i for i,v in df_m.dtypes.iteritems() if v==object]

	df_m = create_dummies(df_m, df_categorical)

	if prep:
		df_m = prep_student(df_m)

	if type(save_csv) == str:
		df_m.to_csv(save_csv, index=False)

	return df_m

def main():
	from argparse import ArgumentParser
	from os.path import join as Join

	parser = ArgumentParser()
	parser.add_argument('--dataset', default='all')
	parser.add_argument('--DATA_DIR', default='./raw_data')
	parser.add_argument('--save_path_housing',
		default='./cleaned_housing_data.csv')
	parser.add_argument('--save_path_student',
		default='./cleaned_student_data.csv')
	parser.add_argument('--prep', default=True)
	args = parser.parse_args()

	if args.dataset == 'all':
		clean_housing_data(Join(args.DATA_DIR, 'dc-residential-properties'), args.save_path_housing, prep=args.prep)
		clean_student_data(Join(args.DATA_DIR, 'Student_Performance'), args.save_path_student, prep=args.prep)

	elif args.dataset == 'housing':
		clean_housing_data(Join(args.DATA_DIR, 'dc-residential-properties'), args.save_path_housing, prep=args.prep)

	elif args.dataset == 'student':
		clean_student_data(Join(args.DATA_DIR, 'Student_Performance'), args.save_path_student, prep=args.prep)

	else:
		print('incorrect dataset argument')

if __name__ == "__main__":
	main()
