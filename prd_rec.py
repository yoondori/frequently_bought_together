import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# int 값으로 변환
def one_hot(x):
    if x <= 0:
        return 0
    if x > 0:
        return 1


def do_work():
    trade = pd.read_csv("거래 정보.csv", sep=',')
    prdClass = pd.read_csv('상품분류 정보.csv', sep=',')
    final = pd.read_csv('final.csv', sep=',')

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


do_work()