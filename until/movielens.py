import pandas as pd

def ml100k():

    p1=r"E:\设计文件\数据集\movielens\ml-100k\ml-100k\u1.base"
    p2=r"E:\设计文件\数据集\movielens\ml-100k\ml-100k\u1.test"

    name=["userId","movieId","rating","time"]


    df1=pd.read_csv(p1,sep="\t",names=name,engine="python")
    df2=pd.read_csv(p2,sep="\t",names=name,engine="python")

    df=pd.concat([df1,df2])
    df.to_csv("Ml100Krating.csv",index=False)

def yahoo3():
    p1=r"E:\设计文件\joint\data\YahooR3\ydata-ymusic-rating-study-v1_0-test.txt"
    p2=r"E:\设计文件\joint\data\YahooR3\ydata-ymusic-rating-study-v1_0-train.txt"

    name=["userId","movieId","rating"]


    df1=pd.read_csv(p1,sep="\t",names=name,engine="python")
    df2=pd.read_csv(p2,sep="\t",names=name,engine="python")
    df=pd.concat([df1,df2])
    new_df=pd.DataFrame()
    for uid,group in df.groupby(["userId"]):
        if len(group)<20:
            continue
        new_df=pd.concat([new_df,group])

    new_df.to_csv(r"E:\设计文件\joint\data\YahooR3.csv",index=False)


def yahoo4():

    p1=r"E:\设计文件\joint\data\YahooR4\ydata-ymovies-user-movie-ratings-train-v1_0.txt"
    p2=r"E:\设计文件\joint\data\YahooR4\ydata-ymovies-user-movie-ratings-test-v1_0.txt" 

    name1=["userId","yahooId","star","rating"]
  
    df1=pd.read_csv(p1,sep="\t",names=name1,engine="python")
    df2=pd.read_csv(p2,sep="\t",names=name1,engine="python")
    
    df=pd.concat([df1,df2])

    ids=df.yahooId.unique()
    movieIds=[i for i in range(len(ids))]
    df3=pd.DataFrame({"yahooId":ids,"movieId":movieIds})
    print(len(ids))
    df=pd.merge(df,df3,on="yahooId")
    df=df.sort_values(by=["userId","movieId"])
    new_df=pd.DataFrame()
    for uid,group in df.groupby(["userId"]):
        if len(group)<20:
            continue
        new_df=pd.concat([new_df,group])
    new_df.to_csv(r"E:\设计文件\joint\data\YahooR4.csv",index=False)

def ml1m():
    df=pd.read_csv("E:\设计文件\joint\data\ML1Mratings.csv",sep="::")
    df.to_csv("E:\设计文件\joint\data\ML1Mratings2.csv",index=False)

if __name__ == "__main__":
    ml1m()