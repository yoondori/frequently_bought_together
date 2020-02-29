import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
from mlxtend.frequent_patterns import apriori, association_rules

# online behavior information
obi = pd.read_csv('온라인 행동 정보.csv')
# getting online data with transaction ids(rows with purchases)
buy_online = obi[obi.trans_id.notnull()]

df = pd.DataFrame(buy_online, columns=['clnt_id', 'trans_id', 'sess_id', 'hit_seq',
                'action_type', 'sess_dt', 'hit_tm', 'hit_pss_tm',
                'tot_pag_view_ct', 'tot_sess_hr_v',
                'trfc_src', 'dvc_ctg_nm'])

# tot_pag_view_ct null values according to clint_id's mean values
df['tot_pag_view_ct'] = df['tot_pag_view_ct'].fillna(df.groupby('clnt_id')['tot_pag_view_ct'].transform('mean'))
# rows that don't have values to make mean value out of
mean_pag = round(df['tot_pag_view_ct'].mean(), 0)
df['tot_pag_view_ct'].fillna(mean_pag, inplace=True)
# filling null device names(device category name) to 'unknown'
df['dvc_ctg_nm'] = df['dvc_ctg_nm'].fillna('unknown')

# checking null value
# df[df.isnull().any(axis=1)]

mean_sess_vals = df.groupby('clnt_id')['tot_sess_hr_v'].transform('mean')
df['tot_sess_hr_v'] = df['tot_sess_hr_v'].fillna(mean_sess_vals)
# rows that don't have values to make mean value out of
mean_sess = round(df['tot_sess_hr_v'].mean(), 0)
df['tot_sess_hr_v'].fillna(mean_sess, inplace=True)

#labeled values
df['trfc_src'].replace({'DIRECT': 1, 'PORTAL_1': 2, 'PORTAL_2': 3, 'PORTAL_3': 4,
                        'PUSH': 5, 'WEBSITE': 6, 'unknown': 7}, inplace=True)
df['dvc_ctg_nm'].replace({'PC': 1, 'mobile_app': 2, 'mobile_web': 3, 'unknown': 4}, inplace=True)

'''
behavior time: hour values 
0: 06:00~11:00
1: 12:00~17:00
2: 18:00~23:00
3: 00:00~05:00
'''

df['hit_tm'] = df.hit_tm.str.split(':').str[0]
df['hit_tm'] = pd.to_numeric(df['hit_tm'])
df['hit_tm'].replace({6:0,7:0,8:0,9:0,10:0,11:0,
                     12:1,13:1,14:1,15:1,16:1,17:1,
                     18:2,19:2,20:2,21:2,22:2,23:2,
                     24:3,1:3,2:3,3:3,4:3,5:3}, inplace=True)

# data type converting
df['trans_id'] = df['trans_id'].astype(int)
df['tot_pag_view_ct'] = df['tot_pag_view_ct'].astype(int)
df['tot_sess_hr_v'] = df['tot_sess_hr_v'].astype(int)

b_online = df.copy()
#online behavior counts per client
cnt = b_online['clnt_id'].value_counts().reset_index()
cnt.columns = ['clnt_id', 'cnt']

df1 = b_online.copy().drop(['sess_id', 'hit_seq', 'action_type', 'sess_dt', 'trans_id'], axis=1)
df2 = df1.groupby('clnt_id').mean().reset_index()
df3 = pd.merge(df2, cnt, on='clnt_id')


# trade data
trade = pd.read_csv("거래 정보.csv")
# product names, seem like they've been labeled with numbers
trade['pd_c'].replace({'unknown': 99999}, inplace=True)
trade['pd_c'] = trade['pd_c'].astype(int)

trade['de_tm'] = trade.de_tm.str.split(':').str[0]
trade['de_tm'] = pd.to_numeric(trade['de_tm'])
trade['de_tm'].replace({6:0,7:0,8:0,9:0,10:0,11:0,
                         12:1,13:1,14:1,15:1,16:1,17:1,
                         18:2,19:2,20:2,21:2,22:2,23:2,
                         24:3,1:3,2:3,3:3,4:3,5:3}, inplace=True)

# business unit (data provider informed they can't give us more info than A means online and B means offline)
# online = 0, offline = 1
trade['biz_unit'].replace({'A01':0,'A02':0,'A03':0,'B01':1,'B02':1, 'B03':1}, inplace=True)

trade_tf = trade.copy()
trade_tf = trade_tf.drop(['trans_id', 'trans_seq', 'pd_c', 'de_dt'], axis=1)

# 고객id에 대한 업종단위(온/오프), 구매일자, 구매금액, 구매수량의 평균
tf2 = trade_tf.groupby('clnt_id').mean().reset_index()

#거래 건수
t2 = trade['clnt_id'].value_counts().reset_index().copy() #11284명
t2.columns = ['clnt_id', 'trade_cnt']
t3 = pd.merge(tf2, t2, on='clnt_id')

# 고객 데이터
clnt = pd.merge(df3, t3, on='clnt_id')
clnt2 = clnt.astype(int)
# print(clnt2.dtypes)

#Clustering numeric variables
varclst = ['hit_tm', 'hit_pss_tm', 'tot_pag_view_ct', 'tot_sess_hr_v', 'buy_am',
           'buy_ct', 'de_tm', 'cnt', 'trade_cnt']
cs1 = clnt2[varclst]
cs1.head()

# data scaling
robustScaler = StandardScaler()
robustScaler.fit(cs1)
cs3 = robustScaler.transform(cs1)

# PCA
pca = PCA(n_components=2)
pca.fit(cs3)

X_pca = pca.transform(cs3)
print('원본 데이터 : {}'.format(str(cs3.shape)))
print('축소 데이터 : {}'.format(str(X_pca.shape)))
print('principal component vec: \n', pca.components_.T)

cs2 = X_pca.copy()
X = np.array(cs2)
print(X)

# 스크리도표
distortions = []
K = range(1, 10)
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(X)
    distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1))/ X.shape[0])

plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('the elbow method showing the optimal k')

kmm1 = KMeans(n_clusters=4)
kmm1.fit(X)

clst_label, clst_cust_counts = np.unique(kmm1.labels_, return_counts=True)
# pd.DataFrame({'clst': clst_label, 'cust_count': clst_cust_counts})

predict = pd.DataFrame(kmm1.predict(X))
predict.columns = ['predict']
# [2 3 1 ... 1 1 0]

# 시각화
km = kmm1 = KMeans(n_clusters=4)
y_km = kmm1.fit_predict(X)

plt.scatter(X[y_km==0,0], X[y_km==0,1], c='green', marker='s', s=50, label='클러스터1')
plt.scatter(X[y_km==1,0], X[y_km==1,1], c='orange', marker='o', s=50, label='클러스터2')
plt.scatter(X[y_km==2,0], X[y_km==2,1], c='yellow', marker='v', s=50, label='클러스터3')
plt.scatter(X[y_km==3,0], X[y_km==3,1], c='blue', marker='v', s=50, label='클러스터4')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], c='red', marker='*', s=50, label='클러스터 중점')
# plt.show()

fin_df = pd.DataFrame(np.hstack((predict, X)))
#컬럼명
fin_df.columns = ['group', 'pca1', 'pca2']


final = pd.DataFrame(np.hstack((clnt2, predict)))
final.columns = ['clnt_id', 'hit_tm', 'hit_pss_tm', 'tot_pag_view_ct', 'tot_sess_hr_v',
       'trfc_src', 'dvc_ctg_nm', 'cnt', 'biz_unit', 'de_tm', 'buy_am',
       'buy_ct', 'trade_cnt','group']


def one_hot(x):
    if x <= 0:
        return 0
    if x > 0:
        return 1

# trade = pd.read_csv("거래 정보.csv", sep=',')
prdClass = pd.read_csv('상품분류 정보.csv', sep=',')

df = trade.copy()
df = df[df.pd_c != 'unknown']

group = final[['clnt_id', 'group']].copy()
df1 = pd.merge(df, group, on='clnt_id')
df2 = df1[['group', 'clnt_id', 'trans_id', 'pd_c']].copy()
df2.pd_c = df2.pd_c.astype(int)

prdClass.pd_c = prdClass.pd_c.astype(int)

df3 = pd.merge(df2, prdClass, on='pd_c')
data = df3.copy()

prompt = '> '
print("고객 아이디 입력:")
# 12
clntId = int(input(prompt))
clnt_g = int(data[data.clnt_id == clntId].group.iloc[0])
print("상품추천을 기반할 상품이름 입력:")
# 347
pd_c = int(input(prompt))
#소분류
class_s = prdClass[prdClass.pd_c == pd_c].clac_nm3.to_string(index=False).strip()
# 'Fresh Milk'

#그룹 x에 해당하는 고객의 trans 추출
gx = data[data.group == clnt_g]

#그룹 x에서 상품 y가 존재하는 transaction Id 추출
gx_y_transIds = gx[gx.clac_nm3 == class_s].trans_id
# print(gx_y_transIds)

#해당 transaction id의 모든 trans(다른상품들 포함)들 추출
gx_y_trans = gx[gx.trans_id.isin(gx_y_transIds)].copy()

#구매한 상품은 숫자 1 부여
gx_y_trans['buy_ct'] = 1
# print(gx_y_trans.shape) (30743, 7)

#구매숫자로 unstack
gx_y_toNum = (gx_y_trans.groupby(['trans_id', 'clac_nm3'])['buy_ct'].sum().unstack().reset_index().fillna(0).set_index('trans_id'))

gx_y_encoded = gx_y_toNum.applymap(one_hot)

#model
freq_items = apriori(gx_y_encoded, min_support=0.05, use_colnames = True)
rules = association_rules(freq_items, metric='lift', min_threshold=1)
rules = rules.sort_values(['confidence', 'lift'], ascending=[False, False])

#데이터 부족으로 추천해줄만한게 없는 경우
if len(rules) == 0:
    print("데이터가 부족해 추천해줄만한 상품이 없습니다.")
else:
    # 이 상품을 산 경우만을 기준으로(먼저 선택한 상품으로 설정)
    ante = frozenset({class_s})
    rules = rules[rules.antecedents == ante]
    recs = rules.consequents.reset_index(drop=True)
    #위에 제일 서포트가 높은 다섯개만 자르기
    recs = recs[:5].apply(lambda x: list(x)[0]).astype("unicode").tolist()
    pd_c = str(pd_c)+"(상품 소분류: "+class_s+")"
    st1 = "{}번 고객의 장바구니에 있는 상품 {}에 대한 추천 상품 소분류: \n {}"
    print(st1.format(clntId, pd_c, recs))