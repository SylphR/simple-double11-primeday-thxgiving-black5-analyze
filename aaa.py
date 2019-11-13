import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

double11 = [[1,0.5],[2,9.36],[3,52],[4,191],[5,350],[6,571],[7,912],[8,1207],[9,1683],[10,2135]]
Thanksgiving = [[1,288],[2,318],[3,407],[4,479],[5,633],[6,766],[7,1009],[8,1096],[9,1287]]
Blackfriday = [[1,534],[2,595],[3,648],[4,816],[5,1042],[6,1198],[7,1505],[8,1656],[9,1970]]
PrimeDay = [[1,0.415],[2,0.525],[3,1],[4,4.19]]

list = [double11,Thanksgiving,Blackfriday,PrimeDay]

for l in list:
    t = pd.DataFrame(l, columns=['year', 'sale'])
    X = t['year'].values.reshape(-1, 1)
    y = t['sale']

    pf = PolynomialFeatures(degree=3)
    pf.fit(X)
    Xp = pf.fit_transform(X)

    lr = LinearRegression()
    lr.fit(Xp, y)
    if l == double11:
        test = [[11, 2684]]
        tt = pd.DataFrame(test, columns=['year', 'sale'])
        Xt = tt['year'].values.reshape(-1, 1)
        pred = lr.predict(pf.transform(Xt))
        print('{}{}{}{}{}{}'.format('双十一R2：', lr.score(Xp, y),' 预测 ',pred,' 实际：2684 误差: ',(pred-2684)/2684))
    if l == Thanksgiving:
        test = [[10,1572]]
        tt = pd.DataFrame(test, columns=['year', 'sale'])
        Xt = tt['year'].values.reshape(-1, 1)
        pred = lr.predict(pf.transform(Xt))
        print('{}{}{}{}{}{}'.format('感恩节R2：', lr.score(Xp, y),' 预测 ',pred,' 实际：1572 误差: ',(pred-1572)/1572))
    if l == Blackfriday:
        test = [[10,2360]]
        tt = pd.DataFrame(test, columns=['year', 'sale'])
        Xt = tt['year'].values.reshape(-1, 1)
        pred = lr.predict(pf.transform(Xt))
        print('{}{}{}{}{}{}'.format('黑五节R2：', lr.score(Xp, y),' 预测 ',pred,' 实际：2630 误差: ',(pred-2630)/2630))
    if l == PrimeDay:
        test = [[5,5.8]]
        tt = pd.DataFrame(test, columns=['year', 'sale'])
        Xt = tt['year'].values.reshape(-1, 1)
        pred = lr.predict(pf.transform(Xt))
        print('{}{}{}{}{}{}'.format('亚马逊R2：', lr.score(Xp, y), ' 预测 ', pred, ' 实际：5.8 误差: ', (pred - 5.8) / 5.8))


