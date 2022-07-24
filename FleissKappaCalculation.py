import pandas as pd
import numpy
NUMANNOTATORS = 6
NENTRIES = 100

headers = ['id', 'hs', 'stance', 'irony', 'profile']

df = pd.read_csv('data/data.csv', names=headers, index_col=0, skiprows=1)


class AnalysisFleissKappa(object):
    def __init__(self, categories, dictReplacements) -> None:
        self._categories = categories
        self._dictReplacements = dictReplacements

    def analysisKappa(self, dfAn, label):
        listDf = []

        for cat in self._categories:
            df = dfAn.replace(self._dictReplacements[cat])
            df = df.groupby(["id"], as_index=False)[
                label].sum()
            df = df.rename(columns={label: cat})
            listDf.append(df)

        df = listDf[0]
        for i in range(1, len(listDf)):
            df = pd.merge(df, listDf[i], on="id")

        self.__printResultsKappa(df, self._categories, label)

    def __printResultsKappa(self, hsDF, categories, label):
        k = calculateFleissKappa(hsDF, categories, NENTRIES, NUMANNOTATORS, 0)
        print("Fleiss' fixed-Kappa for {}:".format(label))
        print(k)
        k = calculateFleissKappa(hsDF, categories, NENTRIES, NUMANNOTATORS, 1)
        print("Fleiss' free-marginal for {}:".format(label))
        print(k)


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
    analysisKappaIrHS(df)
    analysisKappaStance(df)

def analysisKappaIrHS(df):
    cat_hs_irony = ['NumYes', 'NumNo']

    dict_replacement_mapping = {
        "NumYes": {
            "no": 0,
            "yes": 1,
        },
        "NumNo": {
            "no": 1,
            "yes": 0,
        }
    }
    
    Analysis = AnalysisFleissKappa(categories=cat_hs_irony, dictReplacements=dict_replacement_mapping)
    Analysis.analysisKappa(df, 'hs')
    Analysis.analysisKappa(df, 'irony')

def analysisKappaStance(df):
    categories = ['NumFavStance', 'NumAgStance', 'NumNoneStance']
    dict_replacement_mapping = {
        'NumFavStance':
        {
            "none": 0,
            "against": 0,
            "favor": 1,
        },
        'NumAgStance': 
        {
            "none": 0,
            "against": 1,
            "favor": 0,
        },
        'NumNoneStance':
        {
            "none": 1,
            "against": 0,
            "favor": 0,
        }
    } 

    Analysis = AnalysisFleissKappa(categories=categories, dictReplacements=dict_replacement_mapping)
    Analysis.analysisKappa(df, 'stance')


printFleissKappaValues(df)
