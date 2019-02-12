CS7641 - Assignment 1
Michael Tong (903357308)
---

Raw Data and Project scripts can be found at the respective locations:
[Raw Data (not needed):](https://gtvault-my.sharepoint.com/:f:/g/personal/mtong31_gatech_edu/EpgtfeUruDlJsmtEde-PE_QBa7sxvdDPm3V54cZ1f0Y24g?e=WaNigB)

[Project files (contains cleaned data):](https://gtvault-my.sharepoint.com/:f:/g/personal/mtong31_gatech_edu/En88tAbxoItOn4ywMoqI2QwBA0fVfmSoELM4eqTx90reTg?e=vBFvgF)

* In the project folder, run `pip install -r requirements.txt` if desired; packages are matplotlib, numpy, Pandas, Scikit-learn, and XGBoost.

To run analysis, enter `python3 -m process_file` in a terminal. This will run every classification on the _smaller dataset_ (Student). If you want to run analysis on the larger (Housing) dataset, pass the argument --dataset=housing, or --dataset=all for both. _The larger dataset takes about 18 hours to run on a 2017 MacBook Pro_.

To run data cleaning and preprocessing on the raw data scripts, download and copy the raw_data folder into the Project folder. Then run `python3 -m data_cleaning`, which will create the cleaned datasets into the project folder.


Have a wonderful day :)
