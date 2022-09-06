import hashlib
import os
import tarfile
import zipfile
import matplotlib.pyplot as plt
import requests
import seaborn as sns
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.model_selection import cross_val_score
import pandas as pd
import tensorflow as tf
from d2l import tensorflow as d2l
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import LabelEncoder                    #标签编码
from sklearn.preprocessing import RobustScaler, StandardScaler    #去除异常值与数据标准化
from sklearn.pipeline import Pipeline, make_pipeline              #构建管道
from scipy.stats import skew                                 #偏度
from scipy.special import boxcox1p       
from sklearn.linear_model import LogisticRegression
                    # box-cox变换
#from sklearn.preprocessing import Imputer
from sklearn.decomposition import PCA
from scipy.stats import norm, skew     
Imputer = SimpleImputer(strategy="median")
DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'
def custom_coding(x):
    if(x=='Ex'):
        r = 0
    elif(x=='Gd'):
        r = 1
    elif(x=='TA'):
        r = 2
    elif(x=='Fa'):
        r = 3
    elif(x=='None'):
        r = 4
    else:
        r = 5
    return r

def null_hist(data):                             # 自定义缺失值比例绘图函数
    cols = list(data.columns)
    the_null_percents = []
    for col in cols:
        # print(col)
        the_null_percent = np.uint8(data[col].isnull()).sum()/len(data[col].values)       # 计算特征列的缺失值比例
        # print(the_na_percent)

        the_null_percents.append(the_null_percent)

    plot_data = pd.Series(the_null_percents, index = cols)    # 将比例值和对应的特征名称构建成一个Series
    plot_data = plot_data[plot_data.values>0.001]              # 只取出比例大于0.05的数据绘制直方图
    plot_data = plot_data.sort_values(ascending=False)        # 对数据进行降序排列
    plot_data = plot_data[plot_data.values != 0]              # 取出其中比例值不等于o的数据作为最终用于绘图的数据

    ## 绘制直方图 
    plot_x = plot_data.index
    plot_y = plot_data.values

    plt.figure(figsize=(10,6))
    # plt.axis(ymin=0,ymax=1)

    plt.bar(plot_x, plot_y, color='g')

    plt.rcParams['font.sans-serif'] = ['SimHei']     # 图中可以显示中文
    plt.rcParams['axes.unicode_minus'] = False       # 图中可以显示负号

    plt.title("缺失比例大于0.001的特征列数据缺失程度",fontsize=20)
    plt.xlabel("特征名称",fontsize=15)
    plt.ylabel("数据缺失比例",fontsize=15)

    plt.tick_params(labelsize = 15)         #设置坐标轴数字的大小
    plt.xticks(rotation = 90)               #设置坐标轴轴上注记字发生旋转

    for a,b in zip(plot_x,plot_y.round(3)):                    #在柱子上注释数字
        plt.text(a,b+0.01,b,ha='center',va='bottom',fontsize=10) 

        #a指示x的位置，b+50指示y的位置，第二个b为条柱上的注记数字,ha表示注记数字的对齐方式，fontsize表示注释数字字体大小
        #va表示条柱位于注释数字底部还是顶部

    plt.show() 
def download(name, cache_dir=os.path.join('..', 'data')):  #@save
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    return fname
DATA_HUB['kaggle_house_train'] = (  #@save
    DATA_URL + 'kaggle_house_pred_train.csv',
    '585e9cc93e70b39160e7921475f9bcd7d31219ce')

DATA_HUB['kaggle_house_test'] = (  #@save
    DATA_URL + 'kaggle_house_pred_test.csv',
    'fa19780a7b011d9b009e8bff8e99922a8ee2eb90')
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))
print(train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
all_features.shape
n_train = train_data.shape[0]
train_features = tf.constant(all_features[:n_train].values, dtype=tf.float32)
test_features = tf.constant(all_features[n_train:].values, dtype=tf.float32)
train_labels = tf.constant(
    train_data.SalePrice.values.reshape(-1, 1), dtype=tf.float32)
train_data.drop(train_data[(train_data['OverallQual']<5) & (train_data['SalePrice']>200000)].index,inplace=True)
train_data.drop(train_data[(train_data['YearBuilt']<1900) & (train_data['SalePrice']>400000)].index,inplace=True)
train_data.drop(train_data[(train_data['YearBuilt']>1980) & (train_data['SalePrice']>700000)].index,inplace=True)
train_data.drop(train_data[(train_data['TotalBsmtSF']>6000) & (train_data['SalePrice']<200000)].index,inplace=True)
train_data.drop(train_data[(train_data['GrLivArea']>4000) & (train_data['SalePrice']<200000)].index,inplace=True)
## 将异常值点去除
## 重置索引，使得索引值连续
train_data.reset_index(drop=True, inplace=True)

## 数据里面的ID列与数据分析和模型训练无关，在此先删除
train_id = train_data['Id']
test_id = test_data['Id']
train_data.drop("Id", axis = 1, inplace = True)
test_data.drop("Id", axis = 1, inplace = True)
train_data["SalePrice"] = np.log1p(train_data["SalePrice"]) # 对数变换 
all_data=pd.concat([train_data,test_data],axis=0)
all_data.reset_index(drop=True, inplace=True)     # 重置索引，使得索引值连续
null_hist(all_data) 
str_cols = ["PoolQC" , "MiscFeature", "Alley", "Fence", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond",  \
            "BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "MasVnrType", "MSSubClass"]
for col in str_cols:
    all_data[col].fillna("None",inplace=True)
    
del str_cols, col
num_cols=["BsmtUnfSF","TotalBsmtSF","BsmtFinSF2","BsmtFinSF1","BsmtFullBath","BsmtHalfBath", \
          "MasVnrArea","GarageCars","GarageArea","GarageYrBlt"]
for col in num_cols:
    all_data[col].fillna(0, inplace=True)
del num_cols, col
all_data = all_data.drop(["Utilities"], axis=1)##因为都有，分析没有意义
all_data["Functional"] = all_data["Functional"].fillna("Typ")
count=all_data.isnull().sum().sort_values(ascending=False)
ratio=count/len(all_data)
nulldata=pd.concat([count,ratio],axis=1,keys=['count','ratio'])
del count, ratio
nulldata[nulldata.ratio>0]
other_cols = ["MSZoning", "Electrical", "KitchenQual", "Exterior1st", "Exterior2nd", "SaleType"]
for col in other_cols:
    all_data[col].fillna(all_data[col].mode()[0], inplace=True)
    
del other_cols, col
all_data["LotFrontage"] = all_data.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
count=all_data.isnull().sum().sort_values(ascending=False)
ratio=count/len(all_data)
nulldata=pd.concat([count,ratio],axis=1,keys=['count','ratio'])
del count, ratio
nulldata[nulldata.ratio>0]
## 年份等特征的标签编码
str_cols = ["YearBuilt", "YearRemodAdd", 'GarageYrBlt', "YrSold", 'MoSold']
for col in str_cols:
    all_data[col] = LabelEncoder().fit_transform(all_data[col])

## 为了后续构建有意义的其他特征而进行标签编码
lab_cols = ['Heating','BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', \
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold', \
            'MSZoning','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','Exterior1st','MasVnrType',\
            'Foundation', 'GarageType','SaleType','SaleCondition']

for col in lab_cols:
    new_col = "labfit_" + col
    all_data[new_col] = LabelEncoder().fit_transform(all_data[col]) 
        
del col,str_cols,lab_cols,new_col
## 顺序变量特征编码
cols = ['BsmtCond','BsmtQual','ExterCond','ExterQual','FireplaceQu','GarageCond','GarageQual','HeatingQC','KitchenQual','PoolQC']
for col in cols:
    all_data[col] = all_data[col].apply(custom_coding)
    
del cols, col
cols = ['MSSubClass', 'YrSold', 'MoSold', 'OverallCond', "MSZoning", "BsmtFullBath", "BsmtHalfBath", "HalfBath",\
        "Functional", "Electrical", "KitchenQual","KitchenAbvGr", "SaleType", "Exterior1st", "Exterior2nd", "YearBuilt", \
        "YearRemodAdd", "GarageYrBlt","BedroomAbvGr","LowQualFinSF"]
for col in cols:
    all_data[col] = all_data[col].astype(str)    
del cols, col
## 年份等特征的标签编码
str_cols = ["YearBuilt", "YearRemodAdd", 'GarageYrBlt', "YrSold", 'MoSold']
for col in str_cols:
    all_data[col] = LabelEncoder().fit_transform(all_data[col])

## 为了后续构建有意义的其他特征而进行标签编码
lab_cols = ['Heating','BsmtFinType1', 'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope', \
            'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 'YrSold', 'MoSold', \
            'MSZoning','Neighborhood','Condition1','Condition2','BldgType','HouseStyle','Exterior1st','MasVnrType',\
            'Foundation', 'GarageType','SaleType','SaleCondition']

for col in lab_cols:
    new_col = "labfit_" + col
    all_data[new_col] = LabelEncoder().fit_transform(all_data[col]) 
        
del col,str_cols,lab_cols,new_col
all_data['TotalHouseArea'] = all_data['TotalBsmtSF'] + all_data['1stFlrSF'] + all_data['2ndFlrSF']
all_data['YearsSinceRemodel'] = all_data['YrSold'].astype(int) - all_data['YearRemodAdd'].astype(int)
all_data['Total_Home_Quality'] = all_data['OverallQual'].astype(int) + all_data['OverallCond'].astype(int)
all_data['HasWoodDeck'] = (all_data['WoodDeckSF'] == 0) * 1
all_data['HasOpenPorch'] = (all_data['OpenPorchSF'] == 0) * 1
all_data['HasEnclosedPorch'] = (all_data['EnclosedPorch'] == 0) * 1
all_data['Has3SsnPorch'] = (all_data['3SsnPorch'] == 0) * 1
all_data['HasScreenPorch'] = (all_data['ScreenPorch'] == 0) * 1
all_data["TotalAllArea"] = all_data["TotalHouseArea"] + all_data["GarageArea"]                 # 房屋总面积加车库面积
all_data["TotalHouse_and_OverallQual"] = all_data["TotalHouseArea"] * all_data["OverallQual"]  # 房屋总面积和房屋材质指标乘积
all_data["GrLivArea_and_OverallQual"] = all_data["GrLivArea"] * all_data["OverallQual"]        # 地面上居住总面积和房屋材质指标乘积
all_data["LotArea_and_OverallQual"] = all_data["LotArea"] * all_data["OverallQual"]            # 地段总面积和房屋材质指标乘积
all_data["MSZoning_and_TotalHouse"] = all_data["labfit_MSZoning"] * all_data["TotalHouseArea"] # 一般区域分类与房屋总面积的乘积
all_data["MSZoning_and_OverallQual"] = all_data["labfit_MSZoning"] + all_data["OverallQual"]   # 一般区域分类指标与房屋材质指标之和
all_data["MSZoning_and_YearBuilt"] = all_data["labfit_MSZoning"] + all_data["YearBuilt"]       # 一般区域分类指标与初始建设年份之和
## 地理邻近环境位置指标与总房屋面积之积
all_data["Neighborhood_and_TotalHouse"] = all_data["labfit_Neighborhood"] * all_data["TotalHouseArea"]
all_data["Neighborhood_and_OverallQual"] = all_data["labfit_Neighborhood"] + all_data["OverallQual"]  
all_data["Neighborhood_and_YearBuilt"] = all_data["labfit_Neighborhood"] + all_data["YearBuilt"]
all_data["BsmtFinSF1_and_OverallQual"] = all_data["BsmtFinSF1"] * all_data["OverallQual"]      # 1型成品的面积和房屋材质指标乘积
## 家庭功能评级指标与房屋总面积的乘积
all_data["Functional_and_TotalHouse"] = all_data["labfit_Functional"] * all_data["TotalHouseArea"]
all_data["Functional_and_OverallQual"] = all_data["labfit_Functional"] + all_data["OverallQual"]
all_data["TotalHouse_and_LotArea"] = all_data["TotalHouseArea"] + all_data["LotArea"]
## 房屋与靠近公路或铁路指标乘积系数
all_data["Condition1_and_TotalHouse"] = all_data["labfit_Condition1"] * all_data["TotalHouseArea"]
all_data["Condition1_and_OverallQual"] = all_data["labfit_Condition1"] + all_data["OverallQual"]
all_data["Bsmt"] = all_data["BsmtFinSF1"] + all_data["BsmtFinSF2"] + all_data["BsmtUnfSF"]     # 地下室相关面积总和指标
all_data["Rooms"] = all_data["FullBath"]+all_data["TotRmsAbvGrd"]                              # 地面上全浴室和地面上房间总数量之和
## 开放式门廊、围廊、三季门廊、屏风玄关总面积
all_data["PorchArea"] = all_data["OpenPorchSF"]+all_data["EnclosedPorch"]+ \
                        all_data["3SsnPorch"]+all_data["ScreenPorch"]    
## 全部功能区总面积（房屋、地下室、车库、门廊等）
all_data["TotalPlace"] = all_data["TotalAllArea"] + all_data["PorchArea"]   
num_features = all_data.select_dtypes(include=['int64','float64','int32']).copy()
num_features.drop(['SalePrice'],axis=1,inplace=True)               # 去掉目标值房价列

num_feature_names = list(num_features.columns)

num_features_data = pd.melt(all_data, value_vars=num_feature_names)
g = sns.FacetGrid(num_features_data, col="variable",  col_wrap=5, sharex=False, sharey=False)

skewed_feats = all_data[num_feature_names].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness[skewness["Skew"].abs()>0.75]
skew_cols = list(skewness[skewness["Skew"].abs()>1].index)
for col in skew_cols:
    all_data[col] = boxcox1p(all_data[col], 0.15)                                  # 偏度超过阈值的特征做box-cox变换
    all_data[col] = np.log1p(all_data[col])                                                  # 偏度超过阈值的特征对数变换
    
del num_features, num_feature_names, num_features_data, g, skewed_feats, col, skew_cols      # 清除临时变量
all_data.info()
all_data = pd.get_dummies(all_data)       # 一键独热编码
index_train = train_data.index
def split_data(all_data, index_train):
    cols = list(all_data.columns)
    for col in cols:        # 可能特征工程的过程中会产生极个别异常值（正负无穷大），这里用众数填充
        all_data[col].values[np.isinf(all_data[col].values)] = all_data[col].median()   
    del cols, col

    train_data = all_data[:max(index_train)+1]     # 注意索引值对应关系
    test_data = all_data[max(index_train)+1:]

    y_train = train_data["SalePrice"]
    x_train = train_data.copy()
    x_train.drop(["SalePrice"],axis=1,inplace=True)
    x_test = test_data.copy()
    x_test.drop(["SalePrice"],axis=1,inplace=True)

    # del train_data,test_data
    return y_train, x_train, x_test
y_train, x_train, x_test = split_data(all_data, index_train)
scaler = RobustScaler()
x_train = scaler.fit(x_train).transform(x_train)  #训练样本特征归一化
x_test = scaler.transform(x_test)                 #测试集样本特征归一化  
from sklearn.linear_model import Lasso##运用算法来进行训练集的得到特征的重要性，特征选择的一个作用是，wrapper基础模型
lasso_model=Lasso(alpha=0.001)
lasso_model.fit(x_train,y_train)
Lasso(alpha=0.001, copy_X=True, fit_intercept=True, max_iter=1000,
normalize=False, positive=False, precompute=False, random_state=None,
selection='cyclic', tol=0.0001, warm_start=False)
## 索引和重要性做成dataframe形式
FI_lasso = pd.DataFrame({"Feature Importance":lasso_model.coef_}, index=all_data.drop(["SalePrice"],axis=1).columns) 
## 由高到低进行排序
FI_lasso.sort_values("Feature Importance",ascending=False).round(5)  
FI_lasso[FI_lasso["Feature Importance"] !=0 ].sort_values("Feature Importance").plot(kind="barh",figsize=(12,40), color='g')
plt.xticks(rotation=90)
plt.show()                     ##画图显示

FI_index = FI_lasso.index
FI_val = FI_lasso["Feature Importance"].values
FI_lasso = pd.DataFrame(FI_val, columns = ['Feature Importance'], index = FI_index)
choose_cols = FI_lasso.index.tolist()
choose_cols.append("SalePrice")
choose_data = all_data[choose_cols].copy()
del all_data
import math
def get_mse(records_real, records_predict):
    ## 均方误差 估计值与真值 偏差
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None

def get_rmse(records_real, records_predict):
    ## 均方根误差：是均方误差的算术平方根
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None

#定义交叉验证的策略，以及评估函数
def rmse_cv(model,X,y):
    ## 针对各折数据集的测试结果的均方根误差
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=5))   # cv 代表数据划分的KFold折数
    return rmse
y_train, x_train, x_test = split_data(choose_data, index_train)
scaler = RobustScaler()
x_train = scaler.fit(x_train).transform(x_train)  #训练样本特征归一化
x_test = scaler.transform(x_test)                 #测试集样本特征归一化
y_train = y_train.values.reshape(-1,1)
pca_model = PCA(n_components=375)
x_train = pca_model.fit_transform(x_train)
x_test = pca_model.transform(x_test)
# 定义先验参数网格搜索验证方法
class grid():
    def __init__(self,model):
        self.model = model
    
    def grid_get(self,X,y,param_grid):
        grid_search = GridSearchCV(self.model,param_grid,cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(X,y)
        # 打印最佳参数及对应的评估指标
        print(grid_search.best_params_, np.sqrt(-grid_search.best_score_))
        grid_search.cv_results_['mean_test_score'] = np.sqrt(-grid_search.cv_results_['mean_test_score'])
        
        # 打印单独的各参数组合参数及对应的评估指标
        print(pd.DataFrame(grid_search.cv_results_)[['params','mean_test_score','std_test_score']])
#指定每一个算法的参数
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.svm import SVR, LinearSVR
from sklearn.linear_model import ElasticNet, SGDRegressor, BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
lasso = Lasso(alpha=0.0004,random_state=1,max_iter=10000)
ridge = Ridge(alpha=35)
svr = SVR(gamma= 0.0004,kernel='rbf',C=14,epsilon=0.009)
ker = KernelRidge(alpha=0.4 ,kernel='polynomial',degree=3 , coef0=1.2)
ela = ElasticNet(alpha=0.004,l1_ratio=0.08,random_state=3,max_iter=10000)
bay = BayesianRidge()
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,learning_rate=0.05, max_depth=3,
                   min_child_weight=1.7817, n_estimators=2200,reg_alpha=0.4640, 
                   reg_lambda=0.8571,subsample=0.5213, silent=1,random_state =7, nthread = -1)
lgbm = LGBMRegressor(objective='regression',num_leaves=5,learning_rate=0.05, n_estimators=700,max_bin = 55,
                     bagging_fraction = 0.8,bagging_freq = 5, feature_fraction = 0.25,feature_fraction_seed=9, 
                     bagging_seed=9,min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)
GBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5)
class stacking(BaseEstimator, RegressorMixin, TransformerMixin):
    ##=============== 参数说明 ================##
    # mod --- 堆叠过程的第一层中的算法
    # meta_model --- 堆叠过程的第二层中的算法，也称次学习器
    
    def __init__(self,mod,meta_model):
        self.mod = mod                                                # 首层学习器模型
        self.meta_model = meta_model                                  # 次学习器模型
        self.kf = KFold(n_splits=5, random_state=42, shuffle=True)    # 这就是堆叠的最大特征进行了几折的划分
    
    ## 训练函数
    def fit(self,X,y):
        self.saved_model = [list() for i in self.mod]          # self.saved_model包含所有第一层学习器
        oof_train = np.zeros((X.shape[0], len(self.mod)))      # 维度：训练样本数量*模型数量，训练集的首层预测值
        
        for i,model in enumerate(self.mod):                    #返回的是索引和模型本身
            for train_index, val_index in self.kf.split(X,y):  #返回的是数据分割成分（训练集和验证集对应元素）的索引
                renew_model = clone(model)                     #模型的复制
                renew_model.fit(X[train_index], y[train_index])           #对分割出来的训练集数据进行训练
                self.saved_model[i].append(renew_model)                   #把模型添加进去
                #oof_train[val_index,i] = renew_model.predict(X[val_index]).reshape(-1,1) #用来预测验证集数据
                
                val_prediction = renew_model.predict(X[val_index]).reshape(-1,1)    # 验证集的预测结果，注：结果是没有索引的
                
                for temp_index in range(val_prediction.shape[0]):
                    oof_train[val_index[temp_index],i] = val_prediction[temp_index] #用来预测验证集数据的目标值
                
        self.meta_model.fit(oof_train,y)                       # 次学习器模型训练，这里只是用到了首层预测值作为特征
        return self
    
    ## 预测函数
    def predict(self,X):
        whole_test = np.column_stack([np.column_stack(model.predict(X) for model in single_model).mean(axis=1) 
                                      for single_model in self.saved_model])        #得到的是整个测试集的首层预测值
        return self.meta_model.predict(whole_test)            # 返回次学习器模型对整个测试集的首层预测值特征的最终预测结果              
    
    ## 获取首层学习结果的堆叠特征
    def get_oof(self,X,y,test_X):                 
        oof = np.zeros((X.shape[0],len(self.mod)))                #初始化为0
        test_single = np.zeros((test_X.shape[0],5))               #初始化为0 
        #display(test_single.shape)
        test_mean = np.zeros((test_X.shape[0],len(self.mod)))
        for i,model in enumerate(self.mod):                       #i是模型
            for j, (train_index,val_index) in enumerate(self.kf.split(X,y)):          #j是所有划分好的的数据
                clone_model = clone(model)                                            #克隆模块，相当于把模型复制一下
                clone_model.fit(X[train_index],y[train_index])                        #把分割好的数据进行训练
                
                val_prediction = clone_model.predict(X[val_index]).reshape(-1,1)      # 验证集的预测结果，注：结果是没有索引的
                for temp_index in range(val_prediction.shape[0]):
                    oof[val_index[temp_index],i] = val_prediction[temp_index]         #用来预测验证集数据
                    
                #oof[val_index,i] = clone_model.predict(X[val_index]).reshape(-1,1)    #对验证集进行预测
                # test_single[:,j] = clone_model.predict(test_X).reshape(-1,1)           #对测试集进行预测
                
                test_prediction = clone_model.predict(test_X).reshape(-1,1)           #对测试集进行预测
                
                # display(test_prediction.shape)
                test_single[:,j] = test_prediction[:,0]
            test_mean[:,i] = test_single.mean(axis=1)                                  #测试集算好均值
        return oof, test_mean
stack_model = stacking(mod=[ela,svr,bay,lasso,ridge,ker], meta_model=ker)

x_train = Imputer.fit_transform(x_train)
y_train = Imputer.fit_transform(y_train.reshape(-1,1)).ravel()
score = rmse_cv(stack_model,x_train, y_train)            
x_train_stack, x_test_stack = stack_model.get_oof(x_train,y_train,x_test)
x_train_add = np.hstack((x_train,x_train_stack))
x_test_add = np.hstack((x_test,x_test_stack))
score = rmse_cv(stack_model,x_train_add,y_train)
print(score.mean())
## 指定每一个算法的参数
lasso = Lasso(alpha=0.0004,max_iter=10000)
ridge = Ridge(alpha=35)
svr = SVR(gamma= 0.0004,kernel='rbf',C=15,epsilon=0.009)
ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=1.2)
ela = ElasticNet(alpha=0.0005,l1_ratio=0.08,max_iter=10000)
bay = BayesianRidge()
xgb = XGBRegressor(colsample_bytree=0.4603, gamma=0.0468,learning_rate=0.05, max_depth=3,
                   min_child_weight=1.7817, n_estimators=2200,reg_alpha=0.4640, 
                   reg_lambda=0.8571,subsample=0.5213, silent=1,random_state =7, nthread = -1)
lgbm = LGBMRegressor(objective='regression',num_leaves=5,learning_rate=0.05, n_estimators=700,max_bin = 55,
                     bagging_fraction = 0.8,bagging_freq = 5, feature_fraction = 0.25,feature_fraction_seed=9, 
                     bagging_seed=9,min_data_in_leaf = 6, min_sum_hessian_in_leaf = 11)

GBR = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,max_depth=4, max_features='sqrt',
                                min_samples_leaf=15, min_samples_split=10, loss='huber', random_state =5)

## stack_model定义
stack_model = stacking(mod=[ela,svr,bay,lasso,GBR,ker], meta_model=ker)
last_x_train_stack, last_x_test_stack = stack_model.get_oof(x_train_add,y_train,x_test_add)
param_grid = {'alpha':[0.2,0.3,0.4,0.5], 'kernel':["polynomial"], 'degree':[3],'coef0':[0.8,1,1.2]}#定义好的参数，用字典来表示
grid(KernelRidge()).grid_get(last_x_train_stack, y_train, param_grid)
ker = KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=1.2)
my_model = ker.fit(last_x_train_stack, y_train)
y_pred_stack = np.expm1(my_model.predict(last_x_test_stack))
## 直接用stack_model集成好的类函数拟合并预测数据
stack_model = stacking(mod=[lgbm,ela,svr,ridge,lasso,bay,xgb,GBR,ker], \
                       meta_model=KernelRidge(alpha=0.2 ,kernel='polynomial',degree=3 , coef0=1.2))
stack_model.fit(x_train_add,y_train)
y_pred_stack = np.exp(stack_model.predict(x_test_add))
xgb.fit(last_x_train_stack, y_train)
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
colsample_bynode=1, colsample_bytree=0.4603, gamma=0.0468,
importance_type='gain', learning_rate=0.05, max_delta_step=0,
max_depth=3, min_child_weight=1.7817, missing=None,
n_estimators=2200, n_jobs=1, nthread=-1, objective='reg:linear',
random_state=7, reg_alpha=0.464, reg_lambda=0.8571,
scale_pos_weight=1, seed=None, silent=1, subsample=0.5213,
verbosity=1)

y_pred_xgb = np.expm1(xgb.predict(last_x_test_stack))
y_train_xgb = xgb.predict(last_x_train_stack)
lgbm.fit(last_x_train_stack, y_train)
y_pred_lgbm = np.expm1(lgbm.predict(last_x_test_stack))
y_train_lgbm = xgb.predict(last_x_train_stack)
y_pred = (0.7*y_pred_stack)+(0.15*y_pred_xgb)+(0.15*y_pred_lgbm)
ResultData=pd.DataFrame(np.hstack((test_id.values.reshape(-1,1),y_pred.reshape(-1,1))), index=range(len(y_pred)), \
                        columns=['Id', 'SalePrice'])
ResultData['Id'] = ResultData['Id'].astype('int')
ResultData.to_csv(r'D:\submission.csv',index=False)