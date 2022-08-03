import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from scipy.stats import shapiro
from scipy.stats import ttest_ind
import matplotlib.pyplot as plt

def process(df):
    '''This function uses the dataframe as input and creates a dictionary from each row.
    Using this function, it becomes possible to isolate the numerical data for further analyses.
    input:
    - df: dataframe of dataset with non-numerical data.
    output:
    - values_dict: dictionary with targets as keys and pressures across subjects as values. '''
    values = df.iloc[:,2:]
    values_dict = dict()
    for row in values.index:
        values_dict[df.iloc[row,1]] = pd.Series.tolist(np.round(values.iloc[row],3))
    return values_dict

def compare_two(dict1, dict2):
    '''Using this function, two dictionaries of mean pressure values from two mattrasses can be compared 
    using the Mann-Whitney U test. Targets must be the same and in the same order. Returns p-values of 
    Mann-Whitney U test.
    inputs: 
    - dict1: dictionary of targets of dataset 1 with the pressure values of all subjects
    - dict2: dictionary of targets of dataset 2 with the pressure values of all subjects
    outputs:
    - p_values: dictionary of targets with the p-values from Mann-Whitney U test.'''
    p_values = dict()
    for ROI in dict1:
        _, p = mannwhitneyu(dict1[ROI], dict2[ROI]) # Perform the Mann-Whitney U test
        p_values[ROI] = np.round(p, 3)
    
    return p_values

def compare_ttest(dict1, dict2):
    '''Using this function, two dictionaries of mean pressure values from two mattrasses can be compared 
    using the independent two-sample t-test. Targets must be the same and in the same order. Returns p-values of 
    independent two-sample t-test.
    inputs: 
    - dict1: dictionary of targets of dataset 1 with the pressure values of all subjects
    - dict2: dictionary of targets of dataset 2 with the pressure values of all subjects
    outputs:
    - p_values: dictionary of targets with the p-values from independent two-sample t-test.'''
    p_values = dict()
    for ROI in dict1:
        _, p = ttest_ind(dict1[ROI], dict2[ROI]) # Perform the independent two-sample t-test
        p_values[ROI] = np.round(p, 3)
    
    return p_values


def test_normality(df):
    '''This function tests the normality of each row by performing the shapiro-wilk test for normality
    It returns the p_value in a dataframe. The original Shapiro-Wilk test is suitable for small sample sizes between 5-30. 
    https://www.real-statistics.com/tests-normality-and-symmetry/statistical-tests-normality-symmetry/shapiro-wilk-test/
    input:
    - df: dataframe that you want test normality in. The function investigates normality for each row.
    output:
    - new_df: returns a dataframe with the same number of rows as the input containing the p-values for each row. p<0.05 
                means normality is rejected.
    '''
    new_df = pd.DataFrame(columns = ['p-value'])
    for index,row in df.iloc[:,2:].iterrows():
        _, p = shapiro(row.array)
        new_df.at[index,'p-value'] = p
    return new_df

def make_plots(accella, accumax, flexi, dolphin, target):
    '''This function makes a box and whisker plot for a specific target (string). Returns the figure.
    For optimal use, use the order of 'accella', 'accumax', 'flexi', 'dolphin'. 
    inputs: 
    - accella: dictionary of accella data. Key should be targets and values the pressure measurements.
    - accumax: dictionary of accumax data. Key should be targets and values the pressure measurements.
    - flexi: dictionary of flexi data. Key should be targets and values the pressure measurements.
    - dolphin: dictionary of target data. Key should be targets and values the pressure measurements.
    - target: string of target of interest.
    output:
    - box and whisker plot of the requested target. '''
    fig = plt.figure(figsize = (10,7))
    ax = fig.add_subplot(111)
    ax.set_xticklabels(['accella','accumax', 'flexi', 'dolphin'])
    colors = ['#5C6599','#FBF2E7','#456D75','#5EB9ED'] # standard Clinical Technology colors :) 
    bplot1= plt.boxplot([accella[target], accumax[target], flexi[target], 
                        dolphin[target]], patch_artist = True)
    for patch, color in zip(bplot1['boxes'],colors):
        patch.set_facecolor(color)

    ax.set_title(f'{target} pressure distribution over all mattresses')    
    ax.set_ylabel('Pressure (mmHg)')
    return ax

#### IMPORT DATA ####

# Datasets for the point analyses of all four mattrasses
df_accella = pd.read_excel(r'C:\Users\linda\Dropbox\TM\Stagedocumenten\Q4 2021-2022\meetronde 2 20-06\analyse_accella.xlsx', sheet_name ='point')
df_accumax = pd.read_excel(r'C:\Users\linda\Dropbox\TM\Stagedocumenten\Q4 2021-2022\meetronde 2 20-06\analyse_accumax.xlsx', sheet_name ='point')
df_dolphin = pd.read_excel(r'C:\Users\linda\Dropbox\TM\Stagedocumenten\Q4 2021-2022\meetronde 2 20-06\analyse_dolphin.xlsx', sheet_name ='point')
df_flexi = pd.read_excel(r'C:\Users\linda\Dropbox\TM\Stagedocumenten\Q4 2021-2022\meetronde 2 20-06\analyse_flexi.xlsx', sheet_name ='point')

# Datasets for the ROI analyses of all four mattrasses
df_accella_roi = pd.read_excel(r'C:\Users\linda\Dropbox\TM\Stagedocumenten\Q4 2021-2022\meetronde 2 20-06\analyse_accella.xlsx', sheet_name ='roi')
df_accumax_roi  = pd.read_excel(r'C:\Users\linda\Dropbox\TM\Stagedocumenten\Q4 2021-2022\meetronde 2 20-06\analyse_accumax.xlsx', sheet_name ='roi')
df_dolphin_roi  = pd.read_excel(r'C:\Users\linda\Dropbox\TM\Stagedocumenten\Q4 2021-2022\meetronde 2 20-06\analyse_dolphin.xlsx', sheet_name ='roi')
df_flexi_roi = pd.read_excel(r'C:\Users\linda\Dropbox\TM\Stagedocumenten\Q4 2021-2022\meetronde 2 20-06\analyse_flexi.xlsx', sheet_name ='roi')

# Make dictionaries from the datasets for further analyses
accella_dict = process(df_accella)
accumax_dict = process(df_accumax)
dolphin_dict = process(df_dolphin)
flexi_dict = process(df_flexi)

accella_roi_dict = process(df_accella_roi)
accumax_roi_dict = process(df_accumax_roi)
dolphin_roi_dict = process(df_dolphin_roi)
flexi_roi_dict = process(df_flexi_roi)

#### TEST NORMALITY ####
# Point measurements
normal_df_point = pd.DataFrame(columns = ['ROI','accella_point','accumax_point','dolphin_point','flexi_point'])
normal_df_point['ROI'] = df_accella['ROI']
normal_df_point['accella_point'] = test_normality(df_accella)
normal_df_point['accumax_point'] = test_normality(df_accumax)
normal_df_point['dolphin_point'] = test_normality(df_dolphin)
normal_df_point['flexi_point'] = test_normality(df_flexi)

#normal_df_point.to_excel("normal_point.xlsx") # Uncomment to export excel

# ROI measurements
normal_df_roi = pd.DataFrame(columns = ['ROI','accella_roi','accumax_roi','dolphin_roi','flexi_roi'])
normal_df_roi['ROI'] = df_accella_roi['ROI']
normal_df_roi['accella_roi'] = test_normality(df_accella_roi)
normal_df_roi['accumax_roi'] = test_normality(df_accumax_roi)
normal_df_roi['dolphin_roi'] = test_normality(df_dolphin_roi)
normal_df_roi['flexi_roi'] = test_normality(df_flexi_roi)

#normal_df_roi.to_excel("normal_roi.xlsx") # Uncomment to export excel

#### COMPARE POINT MEASUREMENTS USING MANN WHITNEY U TEST ####
flexi_vs_dolphin = compare_two(flexi_dict, dolphin_dict)
flexi_vs_accumax = compare_two(flexi_dict, accumax_dict)
flexi_vs_accella = compare_two(flexi_dict, accella_dict)
accumax_vs_dolphin = compare_two(accumax_dict, dolphin_dict)
accumax_vs_accella = compare_two(accumax_dict, accella_dict)
accella_vs_dolphin = compare_two(accella_dict, dolphin_dict)

# Make a dataframe with the p-values from Mann Whitney U test comparisons
point_vs = pd.DataFrame(columns = ['target','flexi_vs_accella','flexi_vs_accumax','flexi_vs_dolphin',
                                    'accumax_vs_dolphin','accumax_vs_accella','accella_vs_dolphin'])
point_vs['target'] = df_accella.iloc[:,1]
point_vs['flexi_vs_accella'] = list(flexi_vs_accella.values())
point_vs['flexi_vs_accumax'] = list(flexi_vs_accumax.values())
point_vs['flexi_vs_dolphin'] = list(flexi_vs_dolphin.values())
point_vs['accumax_vs_dolphin'] = list(accumax_vs_dolphin.values())
point_vs['accumax_vs_accella'] = list(accumax_vs_accella.values())
point_vs['accella_vs_dolphin']=list(accella_vs_dolphin.values())
#print(point_vs)
#point_vs.to_excel("point_vs.xlsx") # Uncomment to export excel

#### COMPARE ROI MEASUREMENTS USING MANN WHITNEY U TEST ####
# Comparisons of ROI measurements
flexi_vs_dolphin_r = compare_two(flexi_roi_dict, dolphin_roi_dict)
flexi_vs_accumax_r = compare_two(flexi_roi_dict, accumax_roi_dict)
flexi_vs_accella_r = compare_two(flexi_roi_dict, accella_roi_dict)
accumax_vs_dolphin_r = compare_two(accumax_roi_dict, dolphin_roi_dict)
accumax_vs_accella_r = compare_two(accumax_roi_dict, accella_roi_dict)
accella_vs_dolphin_r = compare_two(accella_roi_dict, dolphin_roi_dict)

# Make a dataframe with the p-values from Mann Whitney U test comparisons
roi_vs = pd.DataFrame(columns = ['target','flexi_vs_accella','flexi_vs_accumax','flexi_vs_dolphin',
                                    'accumax_vs_dolphin','accumax_vs_accella','accella_vs_dolphin'])
roi_vs['target'] = df_accella_roi.iloc[:,1]
roi_vs['flexi_vs_accella'] = list(flexi_vs_accella_r.values())
roi_vs['flexi_vs_accumax'] = list(flexi_vs_accumax_r.values())
roi_vs['flexi_vs_dolphin'] = list(flexi_vs_dolphin_r.values())
roi_vs['accumax_vs_dolphin'] = list(accumax_vs_dolphin_r.values())
roi_vs['accumax_vs_accella'] = list(accumax_vs_accella_r.values())
roi_vs['accella_vs_dolphin']=list(accella_vs_dolphin_r.values())
#print(roi_vs)
#roi_vs.to_excel("roi_vs.xlsx") # Uncomment to export to excel

#### COMPARE POINT MEASUREMENTS USING T-TEST ####
tflexi_vs_dolphin = compare_ttest(flexi_dict, dolphin_dict)
tflexi_vs_accumax = compare_ttest(flexi_dict, accumax_dict)
tflexi_vs_accella = compare_ttest(flexi_dict, accella_dict)
taccumax_vs_dolphin = compare_ttest(accumax_dict, dolphin_dict)
taccumax_vs_accella = compare_ttest(accumax_dict, accella_dict)
taccella_vs_dolphin = compare_ttest(accella_dict, dolphin_dict)

# Make a dataframe with the p-values from Mann Whitney U test comparisons
tpoint_vs = pd.DataFrame(columns = ['target','flexi_vs_accella','flexi_vs_accumax','flexi_vs_dolphin',
                                    'accumax_vs_dolphin','accumax_vs_accella','accella_vs_dolphin'])
tpoint_vs['target'] = df_accella.iloc[:,1]
tpoint_vs['flexi_vs_accella'] = list(tflexi_vs_accella.values())
tpoint_vs['flexi_vs_accumax'] = list(tflexi_vs_accumax.values())
tpoint_vs['flexi_vs_dolphin'] = list(tflexi_vs_dolphin.values())
tpoint_vs['accumax_vs_dolphin'] = list(taccumax_vs_dolphin.values())
tpoint_vs['accumax_vs_accella'] = list(taccumax_vs_accella.values())
tpoint_vs['accella_vs_dolphin']=list(taccella_vs_dolphin.values())
#tpoint_vs.to_excel("tpoint_vs.xlsx") # Uncomment to export excel

#### COMPARE ROI MEASUREMENTS USING T-TEST ####
tflexi_vs_dolphin_r = compare_ttest(flexi_roi_dict, dolphin_roi_dict)
tflexi_vs_accumax_r = compare_ttest(flexi_roi_dict, accumax_roi_dict)
tflexi_vs_accella_r = compare_ttest(flexi_roi_dict, accella_roi_dict)
taccumax_vs_dolphin_r = compare_ttest(accumax_roi_dict, dolphin_roi_dict)
taccumax_vs_accella_r = compare_ttest(accumax_roi_dict, accella_roi_dict)
taccella_vs_dolphin_r = compare_ttest(accella_roi_dict, dolphin_roi_dict)

# Make a dataframe with the p-values from t-test comparisons
troi_vs = pd.DataFrame(columns = ['target','flexi_vs_accella','flexi_vs_accumax','flexi_vs_dolphin',
                                    'accumax_vs_dolphin','accumax_vs_accella','accella_vs_dolphin'])
troi_vs['target'] = df_accella_roi.iloc[:,1]
troi_vs['flexi_vs_accella'] = list(tflexi_vs_accella_r.values())
troi_vs['flexi_vs_accumax'] = list(tflexi_vs_accumax_r.values())
troi_vs['flexi_vs_dolphin'] = list(tflexi_vs_dolphin_r.values())
troi_vs['accumax_vs_dolphin'] = list(taccumax_vs_dolphin_r.values())
troi_vs['accumax_vs_accella'] = list(taccumax_vs_accella_r.values())
troi_vs['accella_vs_dolphin']=list(taccella_vs_dolphin_r.values())
#troi_vs.to_excel("troi_vs.xlsx") # Uncomment to export excel

#### Make box and whisker plots ####
# These were the targets I thought were interesting, change the string in order to look at other targets.

# sacrum_S = make_plots(accella_dict, accumax_dict, flexi_dict, dolphin_dict, 'Sacrum (S)')
# plt.show()
# occiput_B = make_plots(accella_dict, accumax_dict, flexi_dict, dolphin_dict, 'Occiput (B)')
# plt.show()
# cheek_L = make_plots(accella_dict, accumax_dict, flexi_dict, dolphin_dict, 'Cheek and ear (L)')
# plt.show()

# Make plots for sacrum_B
# sacrum_B = make_plots(accella_dict, accumax_dict, flexi_dict, dolphin_dict, 'Sacrum (B)')
# plt.show()

# Greater trochanter L point
# greater_trochanter_L = make_plots(accella_dict, accumax_dict, flexi_dict, dolphin_dict, 'Greater trochanter (L)')
# plt.show()

# # Now make plots for ROIs I think are interesting (ROI measurements)
# cheek_P_roi = make_plots(accella_roi_dict, accumax_roi_dict, flexi_roi_dict, dolphin_roi_dict, 'Cheek and ear (P)')
# plt.show()

# cheek_L_roi = make_plots(accella_roi_dict, accumax_roi_dict, flexi_roi_dict, dolphin_roi_dict, 'Cheek and ear (L)')
# plt.show()

# sacrum_B_roi = make_plots(accella_roi_dict, accumax_roi_dict, flexi_roi_dict, dolphin_roi_dict, 'Sacrum (B)')
# plt.show()

# sacrum_S_roi = make_plots(accella_roi_dict, accumax_roi_dict, flexi_roi_dict, dolphin_roi_dict, 'Sacrum (S)')
# plt.show()

# heel_S = make_plots(accella_roi_dict,accumax_roi_dict,flexi_roi_dict,dolphin_roi_dict,'Heel (S)')
# plt.show()

#### Check if there is a significant difference between new and old measurement mats #####
# Datasets for new vs old analyses
df_new = pd.read_excel(r'C:\Users\linda\Dropbox\TM\Stagedocumenten\Q4 2021-2022\oud v new 14-06\analyse_oudvnew.xlsx', sheet_name ='new2')
df_old = pd.read_excel(r'C:\Users\linda\Dropbox\TM\Stagedocumenten\Q4 2021-2022\oud v new 14-06\analyse_oudvnew.xlsx', sheet_name ='old2')

old_dict = process(df_old)
new_dict = process(df_new)
old_vs_new = compare_two(old_dict, new_dict)

# Make a dataframe with the p-values from Mann Whitney U test comparisons
old_vs_new_df = pd.DataFrame(columns = ['ROI','old_vs_new'])
old_vs_new_df['ROI'] = df_new.iloc[:,1]
old_vs_new_df['old_vs_new'] = list(old_vs_new.values())
# old_vs_new_df.to_excel("old_vs_new.xlsx") # Uncomment to export to excel

# NB: extremely small sample size (n=2) so I'm not going to use the results from this statistical test.

