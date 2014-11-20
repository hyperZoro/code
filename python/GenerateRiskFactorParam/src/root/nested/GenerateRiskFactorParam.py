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


filepath = 'C:\\MA\\UOB\\auxilliary\\GeneratedRiskFactorParameter_IRFX1F.csv'
csvfile = open(filepath, 'wb')


#configurable params

CCYList = ['USD','AUD','CAD','CHF','CNH','CNY','DKK','EUR','GBP','HKD','IDR','INR','JPY','KRW','MYR','NOK','NZD','PHP','SEK','SGD','THB','TWD','ZAR']




strModel = 'FX_IR_HW_1F_PWC'
strMethod = 'mt'
rMaxTimeStep = 100000
CalibrationSettings = 'FX_IR_HW_1F_PWC_CalibrationSettings'
CurveNames = 'FX_IR_HW_1F_PWC_CurveNames'
DataExtractionSettings = 'FX_IR_HW_1F_PWC_DataExtractionSettings'
mpTradeInfo = 'PORTFOLIO_INFO'
dtLastCashflow = '2101/07/28'
dtLastRequiredDF = '2101/07/28'


print 'Generating CSV File...'
#Generate Risk Factor Parameters CSV file
staticLines = [['strModel','S',strModel],
               ['strParamSetID','S','FX_IR_HW_1F'],
               ['strMarketDataGeneratorReference','S','QMarketDataGeneratorMFWKUsingInterpolator@quic_market_data_generator_mfwk:12'],
               ['strSimulatorMFWKReference','S','QMCSimulator_FX_IR_HW_1F_PWC_FM@quic_model_fx_ir_hw_1f:2']]
fnWriteList2CSV(csvfile, staticLines)



randomNumberGenerationLines = [['strRandomDrawContext','S','QRandomPseudoAntithetic'],
                               ['strMethod','S',strMethod],
                               ['rMaxTimeStep','D',rMaxTimeStep]]
fnWriteList2CSV(csvfile, randomNumberGenerationLines)



mdpModulesLines = [['mpMarketDataProviderModules','{'],
                   ['}']]
fnWriteList2CSV(csvfile, mdpModulesLines)



strModelLines = [[strModel,'{'],
                 ['CalibrationMethod','S','QModelCalibration_FX_IR_HW_1F_PWC@quic_model_fx_ir_hw_1f:2.0'],
                 ['DataExtractionMethod','S','QDataExtraction_FX_IR_HW_1F_PWC@quic_model_fx_ir_hw_1f:2.0'],
                 ['Model','S','QModel_FX_IR_HW_1F_PWC@quic_model_fx_ir_hw_1f:2.0'],
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
CurveNamesLines = [[CurveNames,'{'],
                   ['aIRCurve','AS',','.join(aIRCurve)],
                   ['aMeanReversionCurve','AS',','.join(aMeanReversionCurve)],
                   ['aSwaptionVolCurve','AS',','.join(aSwaptionVolCurve)],
                   ['aSpotFXCurve','AS',','.join(aSpotFXCurve)],
                   ['aFXImpliedVolCurve','AS',','.join(aFXImpliedVolCurve)],
                   ['aCorrelationCurve','AS']]

for i in range(0,len(CCYList)-1):
    for j in range(i+1,len(CCYList)-1):
            CurveNamesLines.append(['...','SR_'+CCYList[j]+'.CorrelBlock.SR_'+CCYList[i]])
    for j in range(i+1,len(CCYList)-1):
            CurveNamesLines.append(['...','FX_'+CCYList[j]+'.CorrelBlock.SR_'+CCYList[i]])
for i in range(1,len(CCYList)-2):
    for j in range(i+1,len(CCYList)-1):
        CurveNamesLines.append(['...','FX_'+CCYList[j]+'.CorrelBlock.FX_'+CCYList[i]])

CurveNamesLines.append(['}'])
fnWriteList2CSV(csvfile, CurveNamesLines)


ampDataExtractionSettingsIR = []
ampDataExtractionSettingsFX = []
for CCY in CCYList:
    ampDataExtractionSettingsIR.append('Swaption_DataExtractionSetting')
    ampDataExtractionSettingsFX.append('EMPTY_MAP')
DataExtractionSettingsLines = [[DataExtractionSettings,'{'],
                               ['ampDataExtractionSettingsIR','AL',','.join(ampDataExtractionSettingsIR)],
                               ['ampDataExtractionSettingsFX','AL',','.join(ampDataExtractionSettingsFX)],
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


csvfile.close();
print 'Completed!'
