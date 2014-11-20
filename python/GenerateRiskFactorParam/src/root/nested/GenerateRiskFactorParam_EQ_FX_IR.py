'''
Created on 3 Nov 2014

@author: zipeng.huang
'''

import csv

#function writing to CSV
def fnWriteList2CSV(csvfile, inList):
    #csvwriter = csv.writer(csvfile, delimiter=',', quoting = csv.QUOTE_NONE)
    csvwriter = csv.writer(csvfile, delimiter=',', quotechar = ' ')
    for item in inList:
        csvwriter.writerow(item)
    csvwriter.writerow('')
    return


#filepath = 'C:\\MA\\UOB\\auxilliary\\GeneratedRiskFactorParameter_EQ_FX_IR_1F.csv'
filepath = './GeneratedRiskFactorParameter_EQ_FX_IR_1F.csv'
csvfile = open(filepath, 'wb')


#configurable params

CCYList = ['USD','AUD','CAD','CHF','CNH','CNY','DKK','EUR','GBP','HKD','IDR','INR','JPY','KRW','MYR','NOK','NZD','PHP','SEK','SGD','THB','TWD','ZAR']
EQList = ['AIJPM5UE_USD_EQ','DJUBS_USD_EQ','RDXUSD_USD_EQ','SPXD8UE_USD_EQ','SPXT10UE_USD_EQ','SX5E_EUR_EQ']
EQCCYList = ['USD','USD','USD','USD','USD','EUR']



strModel = 'CR_EQ_FX_IF_IR_HW_NF_BK_1F'
strParamSetID = 'CR_EQ_FX_IF_IR_HW_NF_BK_1F'
strMethod = 'mt'
rMaxTimeStep = 100000
CalibrationSettings = 'CR_EQ_FX_IF_IR_HW_NF_BK_1F_PWC_CalibrationSettings'
CurveNames = 'CR_EQ_FX_IF_IR_HW_NF_BK_1F_PWC_CurveNames'
DataExtractionSettings = 'CR_EQ_FX_IF_IR_HW_NF_BK_1F_PWC_DataExtractionSettings'
mpTradeInfo = 'PORTFOLIO_INFO'
dtLastCashflow = '2101/07/28'
dtLastRequiredDF = '2101/07/28'


print 'Generating CSV File...'

mpMarketDataProviderModules = [['mpMarketDataProviderModules','{']]

#Generate Risk Factor Parameters CSV file
staticLines = [['strModel','S',strModel],
               ['strParamSetID','S',strParamSetID],
               ['strMarketDataGeneratorReference','S','QMarketDataGeneratorMFWKUsingInterpolator@quic_market_data_generator_mfwk:12'],
               ['strProviderDependencyMappingReference','S','QProviderDependencyMapping_FX_IF_IR_HW_NF_PWC@quic_provider_dependency_mapping_fx_if_ir_hw_nf:12'],
               ['strSimulatorMFWKReference','S','QMCSimulator_FX_IR_HW_1F_PWC_FM@quic_model_fx_ir_hw_1f:2']]
fnWriteList2CSV(csvfile, staticLines)



randomNumberGenerationLines = [['strRandomDrawContext','S','QRandomPseudoAntithetic'],
                               ['bUseEigenSym','B','true'],
                               ['bSimulateDefaults','B','FALSE'],
                               ['strMethod','S',strMethod],
                               ['rMaxTimeStep','D',rMaxTimeStep]]
fnWriteList2CSV(csvfile, randomNumberGenerationLines)




strModelLines = [[strModel,'{'],
                 ['CalibrationMethod','S','QModelCalibration_CR_EQ_FX_IF_IR_HW_1F_BK_1F@quic_model_cr_eq_fx_if_ir_hw_nf_bk_1f:2.0'],
                 ['DataExtractionMethod','S','QDataExtraction_CR_EQ_FX_IF_IR_HW_1F_BK_1F@quic_model_cr_eq_fx_if_ir_hw_nf_bk_1f:2.0'],
                 ['Model','S','QModel_CR_EQ_FX_IF_IR_HW_NF_BK_1F@quic_model_cr_eq_fx_if_ir_hw_nf_bk_1f:2.0'],
                 ['CalibrationSettings','L',CalibrationSettings],
                 ['CurveNames','L',CurveNames],
                 ['DataExtractionSettings','L',DataExtractionSettings],
                 ['mpTradeInfo','L',mpTradeInfo],
                 ['}']]
fnWriteList2CSV(csvfile, strModelLines)



ampCalibrationSettingsIR = []
for CCY in CCYList:
    ampCalibrationSettingsIR.append(CalibrationSettings+'_'+CCY)
ampCalibrationSettingsFX = []
for CCY in CCYList:
    if CCY==CCYList[0]:
        ampCalibrationSettingsFX.append('void')
    else:
        ampCalibrationSettingsFX.append('FX_CalibrationSettings')
CalibrationSettingsLines = [[CalibrationSettings,'{'],
                            ['astrCCY','AS',','.join(CCYList)],
                            ['astrII','AS'],
                            ['astrIICCY','AS'],
                            ['astrEquity','AS',','.join(EQList)],
                            ['astrEquityCCY','AS',','.join(EQCCYList)],
                            ['astrName','AS'],
                            ['arKappa','AR'],
                            ['aoSigma','AR'],
                            ['bCorrOrdinatesInMonths','B','false'],
                            ['ampCalibrationSettingsIR','AL',','.join(ampCalibrationSettingsIR)],
                            ['ampCalibrationSettingsFX','AL',','.join(ampCalibrationSettingsFX)],
                            ['}']]
fnWriteList2CSV(csvfile, CalibrationSettingsLines)



for CCY in CCYList:
    InList = [[CalibrationSettings+'_'+CCY,'{'],
              ['}']]
    fnWriteList2CSV(csvfile, InList)


    
FXCalibrationSettingsLines = [['FX_CalibrationSettings','{'],
                              ['bUseNegativeSpotVols','B','false'],
                              ['}']]
fnWriteList2CSV(csvfile, FXCalibrationSettingsLines)



aIRCurve = []
aMeanReversionCurve = []
aSwaptionVolCurve = []
for CCY in CCYList:
    aIRCurve.append(CCY+'.Yield.'+CCY)
    aMeanReversionCurve.append('MR_'+CCY+'.MeanReversion.'+CCY)
    aSwaptionVolCurve.append(CCY+'.SwaptionVolMtx.'+CCY)
aSpotFXCurve = []
aFXImpliedVolCurve = []
for CCY in CCYList:
    if CCY==CCYList[0]:
        aSpotFXCurve.append('void')
        aFXImpliedVolCurve.append('void')
    else:
        aSpotFXCurve.append(CCY+'.Exchange.'+CCYList[0])
        aFXImpliedVolCurve.append('IMPVOL_'+CCY+'.ImpliedVol.'+CCY)
aSpotEQCurve = []
aEQImpliedVolCurve = []
aEQDividendCurve =[]
for i in range(0,len(EQList)-1):
    aSpotEQCurve.append(EQList[i]+'.Equity.'+EQCCYList[i])
    aEQImpliedVolCurve.append(EQList[i]+'.ImpliedVol.'+EQCCYList[i])
    aEQDividendCurve.append('void')
CurveNamesLines = [[CurveNames,'{'],
                   ['aIRCurve','AS',','.join(aIRCurve)],
                   ['aMeanReversionCurve','AS',','.join(aMeanReversionCurve)],
                   ['aSwaptionVolCurve','AS',','.join(aSwaptionVolCurve)],
                   ['aSpotFXCurve','AS',','.join(aSpotFXCurve)],
                   ['aFXImpliedVolCurve','AS',','.join(aFXImpliedVolCurve)],
                   ['aINFCurve','AS'],
                   ['aSpotIICurve','AS'],
                   ['aSpotIIVolCurve','AS'],
                   ['aSpotIIImpliedVolCurve','AS'],
                   ['aINFVolCurve','AS'],
                   ['aINFMeanReversionCurve','AS'],
                   ['aSpotEQCurve','AS',','.join(aSpotEQCurve)],
                   ['aEQImpliedVolCurve','AS',','.join(aEQImpliedVolCurve)],
                   ['aEQDividendCurve','AS',','.join(aEQDividendCurve)],
                   ['aCreditCurve','AS'],
                   ['aCorrelationCurve','AS']]

for i in range(0,len(CCYList)-1):
    for j in range(i+1,len(CCYList)-1):
        CurveNamesLines.append(['...','SR_'+CCYList[j]+'.CorrelBlock.SR_'+CCYList[i]])
    for j in range(i+1,len(CCYList)-1):
        CurveNamesLines.append(['...','FX_'+CCYList[j]+'.CorrelBlock.SR_'+CCYList[i]])
    for j in range(0,len(EQList)-1):
        CurveNamesLines.append(['...','EQ_'+EQList[j]+'.CorrelBlock.SR_'+CCYList[i]])
for i in range(0,len(CCYList)-2):
    for j in range(i+1,len(CCYList)-1):
        CurveNamesLines.append(['...','FX_'+CCYList[j]+'.CorrelBlock.FX_'+CCYList[i]])
    for j in range(0,len(EQList)-1):
        CurveNamesLines.append(['...','EQ_'+EQList[j]+'.CorrelBlock.FX_'+CCYList[i]])
for i in range(0,len(EQList)-1):
    for j in range(i+1,len(EQList)-1):
        CurveNamesLines.append(['...','EQ_'+EQList[j]+'.CorrelBlock.EQ_'+EQList[i]])

CurveNamesLines.append(['}'])
fnWriteList2CSV(csvfile, CurveNamesLines)


ampDataExtractionSettingsIR = []
ampDataExtractionSettingsFX = []
ampDataExtractionSettingsEQ = []
for CCY in CCYList:
    ampDataExtractionSettingsIR.append('Swaption_DataExtractionSetting')
    ampDataExtractionSettingsFX.append('EMPTY_MAP')
for EQ in EQList:
    ampDataExtractionSettingsEQ.append('EMPTY_MAP')

DataExtractionSettingsLines = [[DataExtractionSettings,'{'],
                               ['ampDataExtractionSettingsIR','AL',','.join(ampDataExtractionSettingsIR)],
                               ['ampDataExtractionSettingsFX','AL',','.join(ampDataExtractionSettingsFX)],
                               ['ampDataExtractionSettingsEQ','AL',','.join(ampDataExtractionSettingsEQ)],
                               ['ampDataExtractionSettingsII','AL'],
                               ['aidxCDSCCY','AN'],
                               ['}']]
fnWriteList2CSV(csvfile, DataExtractionSettingsLines)




pDataExtractionSettingsIRLines = [['Swaption_DataExtractionSetting','{'],
                               ['strMethod','S','Swaption'],
                               ['}']]
fnWriteList2CSV(csvfile, pDataExtractionSettingsIRLines)



emptyMapLines = [['EMPTY_MAP','{'],
                 ['}']]
fnWriteList2CSV(csvfile, emptyMapLines)



mpTradeInfoLines = [[mpTradeInfo,'{'],
                    ['dtLastCashflow','D',dtLastCashflow],
                    ['dtLastRequiredDF','D', dtLastRequiredDF],
                    ['}']]
fnWriteList2CSV(csvfile, mpTradeInfoLines)


mpMarketDataProviderModules.append(['}'])
fnWriteList2CSV(csvfile, mpMarketDataProviderModules)


csvfile.close();
print 'Completed!'
