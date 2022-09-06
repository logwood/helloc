from random import random
import pandas as pd
from pyecharts.charts import HeatMap
from pyecharts import options as opts
import os
import random
firstly_time=100
end_time=110
retval = os.getcwd()
os.chdir(retval)
a={}
data=[]
df = pd.read_excel(r"1in20.xlsx")
v = df['出生日期'][:50000].values.tolist()
d = df['性别'][:50000].values.tolist()
e = df['手机号'][:50000].values.tolist()
b = [0 for _ in range(50000)]
for i in range(50000):
    v[i] = str(v[i])[0:4]
    iso=v[i]
    if iso[0] > '9' or iso[0] < '0':
        intv=0
        e[i]=0
    else:
       intv=int(iso)
    if intv in a:
         a[intv] += ((int(e[i])/10e8)%100+(int(e[i])/1e7)%1000/10)
    else:
         a[intv] = ((int(e[i])/10e8)%100+(int(e[i])/1e7)%1000/10)
    data.append([random.randint(0,50),random.randint(0,1),a[intv]])
list=set(v)      
x_axis = ["男","女"]
y_axis = list
heatmap = (HeatMap()
                .add_xaxis(x_axis)
                .add_yaxis("series",y_axis,data)
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="热力图"),
                                                                   
                    visualmap_opts=opts.VisualMapOpts(max_=600)
                )
        )
heatmap.render('热力图.html')
