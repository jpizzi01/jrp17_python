import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy import linalg


def nan_chopper(data, percent_thresh):
    # kill rows with value zero in designated columns, like if mass is listed as 0
    indexes = data.index
    columns = data.columns
    # removing vars with greater than 60% nan presence
    nan_bad = []
    nan_good = []
    for i in columns:
        working = data[i]
        nan_count = sum(working.isna())
        n = len(indexes)
        nan_percent = nan_count / n
        if nan_percent > percent_thresh:
            nan_bad.append(i)
        else:
            nan_good.append(i)
    for i in nan_bad:
        data = data.drop(i, axis=1)
    # determining distribution of each variable, gaussian or skewed
    columns = data.columns
    alpha = 0.05
    column_dist = []
    for i in columns:
        test = stats.normaltest(data[i], nan_policy='omit')
        if test[1] > alpha:
            column_dist.append('normal')
        else:
            column_dist.append('skewed')
    # replace NaN in gaussian columns with mean, median for skewed columns
    for i in range(len(data.columns)):
        if column_dist[i] == 'skewed':
            replace = np.nanmedian(data[columns[i]])
            data[columns[i]] = data[columns[i]].fillna(replace)
        if column_dist[i] == 'normal':
            replace = np.nanmean(data[columns[i]])
            data[columns[i]] = data[columns[i]].fillna(replace)
    return data


def explained_var(data):
    cov = np.cov(data, rowvar=False)
    ev = np.linalg.det(cov)
    return ev


def scat_coef(data):
    # converting to df to make column deletion easier with .drop()
    data = pd.DataFrame(data)
    # removing columns with only one unique values, then converting df back to ndarray
    dfdata = data.drop(data.columns[data.nunique() == 1], axis=1)
    new_feats = dfdata.columns
    data = np.array(dfdata)
    corr = np.corrcoef(data, rowvar=False)
    scat = np.linalg.det(corr)
    return scat, corr, new_feats, dfdata


def eig_eval(data):
    (scat, corr, new_feats, dfdata) = scat_coef(data)
    # symmetrizing correlation matrix, becomes asymmetrical due to floating point rounding errors
    corr = (corr + np.transpose(corr))/2
    # only proceed if the correlation matrix is symmetric, def= matrix - matrix''' == [bunch of zeros]
    check = corr - np.transpose(corr)
    if len(check[check != 0]) > 0:
        return print('The correlation matrix is not symmetrical')
    # calculate eigenvalues of the symmetrical correlation matrix
    eig_val = np.array(linalg.eigvalsh(corr))
    # some eigenvalues are negative, but with magnitude 1*10^-13
    # these are zero, but treated as otherwise by the algorithm, so make them actually zero
    eig_val[eig_val < 0] = 0
    # sort eigenvalues in descending order
    eig_val = eig_val[::-1]
    # Gauge PCA usefulness with metrics that can tolerate eigenvalues of 0
    # calculate psi index, where def=sum((eig-1)^2)
    psi = np.sum((eig_val-1)**2)
    print('The psi index of this dataset is %.4f' % psi)
    # calculate information statistic, where def=-0.5*sum(ln(eig))
    # first, prep data for log operation
    eig_val[eig_val == 0] = 10**-10
    infostat = -(1/2)*np.sum(np.log(eig_val))
    print('The information statistic of this dataset is %.4f' % infostat)
    return


def multi_collinearity_check(data, clean):
    # calculate correlation matrix and its scatter coefficient, along with the modified dataframe w/ var=0 columns chop
    scat, corr, new_feats, dfdata = scat_coef(data)
    print('The scatter coefficient of the data is %.9f' % scat)
    # next, decide which features are highly correlated, and could be candidates for removal for techniques like logit
    vif_scores = np.linalg.inv(corr).diagonal()
    vif_series = pd.Series(vif_scores, index=new_feats)
    if clean:
        super_columns = vif_series[vif_series > 10.0].index
        low_vif_data = data.drop(super_columns, axis=1)
        return low_vif_data
    else:
        return vif_series
