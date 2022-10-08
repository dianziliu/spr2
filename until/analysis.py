import pandas as pd

# datasets = {"ml10m":"data/ML10Mratings.csv",
#             "ml100k": r"E:\设计文件\joint\data\Ml100Krating.csv",
#             "ml1m": r"E:\设计文件\joint\data\ML1Mratings.csv",
#             "r3": r"E:\设计文件\joint\data\YahooR3.csv",
#             "r4": r"E:\设计文件\joint\data\YahooR4.csv"}
# datasets={"mlls":"data/MLLS.csv"}
datasets={"ml10m":"data/ML10Mratings.csv"}


#userId,movieId

for name,path in datasets.items():
    print("dataset {}".format(name))
    totalsU={
        "<32":[0,0],
        "<64":[0,0],
        "<128":[0,0],
        "<256":[0,0],
        ">256":[0,0]
    }
    totalsI={
        "<32":[0,0],
        "<64":[0,0],
        "<128":[0,0],
        "<256":[0,0],
        ">256":[0,0]
    }
    df=pd.read_csv(path,engine="python")
    L=len(df)
    for uid,group in df.groupby(["userId"]):
        l=len(group)
        if l<32:
            p=totalsU["<32"]
        elif l<64:
            p=totalsU["<64"]
            
        elif l<128:
            p=totalsU["<128"]
        elif l<256:
            p=totalsU["<256"]
        else :
            p=totalsU[">256"]
        p[0]+=1
        p[1]+=l
    for iid,group in df.groupby(["movieId"]):
        l=len(group)
        if l<32:
            p=totalsI["<32"]
        elif l<64:
            p=totalsI["<64"]
        elif l<128:
            p=totalsI["<128"]
        elif l<256:
            p=totalsI["<256"]
        else :
            p=totalsI[">256"]
        p[0]+=1
        p[1]+=l
    print("    user totals:")
    for key,value in totalsU.items():
        print("    {}:\t{}\t,{}\t".format(key,value[0],value[1]/L ))

    print("    item totals:")
    for key,value in totalsI.items():
        print("    {}:\t{}\t,{}\t".format(key,value[0],value[1]/L))

"""
dataset ml100k
    user totals:
    <32:        221     ,0.05399        
    <64:        243     ,0.11119
    <128:       196     ,0.18036
    <256:       195     ,0.34997
    >256:       88      ,0.30449
    item totals:
    <32:        899     ,0.08979
    <64:        264     ,0.12021
    <128:       266     ,0.2387
    <256:       191     ,0.34109
    >256:       62      ,0.21021
dataset ml1m
    user totals:
    <32:        860     ,0.021485509528508542   
    <64:        1367    ,0.06234197052815962
    <128:       1425    ,0.13177045997386547
    <256:       1196    ,0.21656073880558963
    >256:       1192    ,0.5678413211638768
    item totals:
    <32:        902     ,0.010723758734424505
    <64:        463     ,0.021358536065962212
    <128:       508     ,0.046740231291660043
    <256:       633     ,0.11645666055794339
    >256:       1200    ,0.8047208133500099
dataset r3
    user totals:
    <32:        5140    ,0.46114148844406766    
    <64:        2565    ,0.3933385408761259
    <128:       329     ,0.0982411585918903
    <256:       41      ,0.025544108010854393
    >256:       14      ,0.02173470407706176
    item totals:
    <32:        0       ,0.0
    <64:        3       ,0.0006626659904265126
    <128:       222     ,0.08710911036165274
    <256:       515     ,0.3326990496851411
    >256:       260     ,0.5795291739627797
dataset r4
    user totals:
    <32:        1555    ,0.23961734257222372
    <64:        1010    ,0.2743265774475558
    <128:       370     ,0.20185242711060677
    <256:       132     ,0.14579146564895176
    >256:       44      ,0.13841218722066198
    item totals:
    <32:        9956    ,0.26635771749535625
    <64:        476     ,0.1332073868859318
    <128:       247     ,0.13751196596910087
    <256:       120     ,0.13720132624144948
    >256:       100     ,0.3257216034081616
dataset ml10m
    user totals:
    <32:        14185   ,0.035193609954506246
    <64:        18743   ,0.08520383989926454
    <128:       15773   ,0.14381112341993355
    <256:       11090   ,0.19979962108204616
    >256:       10087   ,0.5359918056442495
    item totals:
    <32:        2518    ,0.0035734807032042027
    <64:        1370    ,0.006219166416501351
    <128:       1332    ,0.01218623419433535
    <256:       1334    ,0.024440868019312695
    >256:       4123    ,0.9535802506666464


dataset mlls
    user totals:
    <32:        119     ,0.02887857511206315
    <64:        165     ,0.07334682057995161
    <128:       123     ,0.11181522472132968
    <256:       100     ,0.1753937085961363
    >256:       103     ,0.6105656709905193
    item totals:
    <32:        8893    ,0.4400908405728113
    <64:        529     ,0.23209964695148558
    <128:       227     ,0.19699313737157365
    <256:       70      ,0.11584156452060772
    >256:       5       ,0.014974810583521759
"""