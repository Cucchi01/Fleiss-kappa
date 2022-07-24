import pandas as pd
import numpy
NUMANNOTATORS = 6
NENTRIES = 100 

headers = ['id', 'hs', 'stance', 'irony', 'profile']

df = pd.read_csv('data/data.csv', names=headers, index_col=0, skiprows=1)


def calculateFleissKappa(df, categories, NSub, nRatings, version):
    columnSum = df[categories].sum().to_numpy()
    pj = columnSum/(NSub*nRatings)
    Pi = []
    listRowDF = df[categories].to_numpy()
    for row in listRowDF:
        sum = 0
        for cell in row:
            sum += cell*(cell-1)
        Pi.append(sum/(nRatings*(nRatings-1)))

    meanP = numpy.mean(Pi)
    if(version == 0):
        P_e = SumOfSquare(pj)
    else:
        P_e = 1/len(categories)

    k = (meanP-P_e)/(1-P_e)
    return k


def SumOfSquare(list):
    sum = 0
    for el in list:
        sum += el * el
    return sum


def printFleissKappaValues(df):
    
    replacement_mapping_dict = {
        "no": 0,
        "yes": 1,
        "none": 0,
        "against": -1,
        "favor": 1,
    }

    df_HS_Ir = df.replace(replacement_mapping_dict)

    analysisKappaHS(df_HS_Ir)
    analysisKappaIr(df_HS_Ir)
    analysisKappaStance(df)


def analysisKappaHS(dfAn):
    hsDF = dfAn[["id", "hs"]]
    hsDF = hsDF.groupby(["id"], as_index=False)["hs"].sum()
    hsDF = hsDF.rename(columns={'hs': 'NumYesHs'})
    hsDF["NumNoHs"] = NUMANNOTATORS-hsDF["NumYesHs"]
    categories = ['NumYesHs', 'NumNoHs']
    k = calculateFleissKappa(hsDF, categories, NENTRIES, NUMANNOTATORS, 0)
    print("Fleiss' Fixed-Kappa for hate speech:")
    print(k)
    k = calculateFleissKappa(hsDF, categories, NENTRIES, NUMANNOTATORS, 1)
    print("Fleiss' Free-marginal for hate speech:")
    print(k)


def analysisKappaIr(dfAn):
    irDF = dfAn[["id", "irony"]]
    irDF = irDF.groupby(["id"], as_index=False)["irony"].sum()
    irDF = irDF.rename(columns={'irony': 'NumYesIr'})
    irDF["NumNoIr"] = NUMANNOTATORS-irDF["NumYesIr"]
    categories = ['NumYesIr', 'NumNoIr']
    k = calculateFleissKappa(irDF, categories, NENTRIES, NUMANNOTATORS, 0)
    print("Fleiss' Fixed-Kappa for irony:")
    print(k)
    k = calculateFleissKappa(irDF, categories, NENTRIES, NUMANNOTATORS, 1)
    print("Fleiss' Free-marginal for irony:")
    print(k)


def analysisKappaStance(df):

    replacement_couting_fav = {
        "no": 0,
        "yes": 1,
        "none": 0,
        "against": 0,
        "favor": 1,
    }

    replacement_couting_against = {
        "no": 0,
        "yes": 1,
        "none": 0,
        "against": 1,
        "favor": 0,
    }

    replacement_couting_none = {
        "no": 0,
        "yes": 1,
        "none": 1,
        "against": 0,
        "favor": 0,
    }

    dfCountFav = df.replace(replacement_couting_fav)
    dfCountAg = df.replace(replacement_couting_against)
    dfCountNone = df.replace(replacement_couting_none)

    dfCountFav = dfCountFav.groupby(["id"], as_index=False)[
        "stance"].sum()
    dfCountFav = dfCountFav.rename(columns={'stance': 'NumFavStance'})
    dfCountAg = dfCountAg.groupby(["id"], as_index=False)["stance"].sum()
    dfCountAg = dfCountAg.rename(columns={'stance': 'NumAgStance'})
    dfCountNone = dfCountNone.groupby(
        ["id"], as_index=False)["stance"].sum()
    dfCountNone = dfCountNone.rename(columns={'stance': 'NumNoneStance'})

    dfStance = pd.merge(dfCountFav, dfCountAg, on="id")
    dfStance = pd.merge(dfStance, dfCountNone, on="id")

    categories = ['NumFavStance', 'NumAgStance', 'NumNoneStance']
    k = calculateFleissKappa(dfStance, categories, NENTRIES, NUMANNOTATORS, 0)
    print("Fleiss' Fixed-marginal for stance:")
    print(k)
    k = calculateFleissKappa(dfStance, categories, NENTRIES, NUMANNOTATORS, 1)
    print("Fleiss' Free-marginal for stance:")
    print(k)

    return


def testCaseKappaFleiss():
    d = {"1": [0, 0, 0, 0, 2, 7, 3, 2, 6, 0], "2": [0, 2, 0, 3, 2, 7, 2, 5, 5, 2], "3": [
        0, 6, 3, 9, 8, 0, 6, 3, 2, 2], "4": [0, 4, 5, 2, 1, 0, 3, 2, 1, 3], "5": [14, 2, 6, 0, 1, 0, 0, 2, 0, 7]}
    df = pd.DataFrame(data=d)
    k = calculateFleissKappa(df, ["1", "2", "3", "4", "5"], 10, 14, 0)
    print("Test case from Wikipedia(results on Wikipedia: 0.210):")
    print(k)


testCaseKappaFleiss()
printFleissKappaValues(df)
