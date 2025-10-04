# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 18:07:19 2025

@author: Ozgr
"""
import datetime
import pandas as pd
import numpy as np
import yfinance as yf
import tech_analysis as ta
import streamlit as st

def get_stock(ticker,start,end,interval,indicator):
    if not isinstance(ticker, str):
        raise TypeError("ticker bir string olmalı.")
        
    try:
        df = yf.download(ticker,start=start,end=end,interval=interval)
        df.columns = df.columns.droplevel(1)
        
        #T3 signal
        if indicator=='T3':
            t3_value =st.sidebar.slider('T3 Period',0,20,value=3)
            t3_factor_value = st.sidebar.slider('T3 Factor Period',0.01,1.0,value=0.7,step=0.05)
            df['T3'] = ta.t3(df['Close'],t3_value ,t3_factor_value)
            cond_t3 = [df['T3']>df['T3'].shift(1),
                   df['T3']<df['T3'].shift(1)]
            choice_t3 = [1,-1]
            df['T3_Signal'] = np.select(cond_t3,choice_t3,default=0)
            df.loc[df['T3_Signal'].diff() != 0, 'Signal_Change'] = df['T3_Signal']
            
            #ema
        if indicator=='EMA':
            fast_indicator = st.sidebar.slider('FAST_INDICATOR_VALUE',3,250,value=8)
            slow_indicator = st.sidebar.slider('SLOW_INDICATOR_VALUE',3,250,value=21)      
            df['FAST_EMA'] = ta.ema(df['Close'], fast_indicator)
            df['SLOW_EMA'] = ta.ema(df['Close'], slow_indicator)
            cond_ema =[((df['FAST_EMA']>df['SLOW_EMA']) & (df['FAST_EMA'].shift(1)<df['SLOW_EMA'].shift(1))),
                       ((df['FAST_EMA']<df['SLOW_EMA']) & (df['FAST_EMA'].shift(1)>df['SLOW_EMA'].shift(1)))]
            
            choices_ema = [1,-1]
            df['EMA_Signal'] = np.select(cond_ema,choices_ema,default=0)
            df.loc[df['EMA_Signal'].diff() != 0, 'Signal_Change'] = df['EMA_Signal']
            
        if indicator=='Close_f_EMA':
            fast_indicator = st.sidebar.slider('FAST_INDICATOR_VALUE',3,250,value=8)
            df['FAST_EMA'] = ta.ema(df['Close'], fast_indicator)
            
            cond_ema =[((df['FAST_EMA']>df['Close']) & (df['Close'].shift(1)<df['FAST_EMA'].shift(1))),
                       ((df['FAST_EMA']<df['Close']) & (df['FAST_EMA'].shift(1)>df['Close'].shift(1)))]
            
            choices_ema = [1,-1]
            df['o_EMA_Signal'] = np.select(cond_ema,choices_ema,default=0)
            df.loc[df['o_EMA_Signal'].diff() != 0, 'Signal_Change'] = df['o_EMA_Signal']
            
        # sma
        if indicator=='SMA':
            fast_indicator = st.sidebar.slider('FAST_INDICATOR_VALUE',3,250,value=8)
            slow_indicator = st.sidebar.slider('SLOW_INDICATOR_VALUE',3,250,value=21)      
            df['FAST_SMA'] = ta.sma(df['Close'], fast_indicator)
            df['SLOW_SMA'] = ta.sma(df['Close'], slow_indicator)
            cond_sma =[((df['FAST_SMA']>df['SLOW_SMA']) & (df['FAST_SMA'].shift(1)<df['SLOW_SMA'].shift(1))),
                       ((df['FAST_SMA']<df['SLOW_SMA']) & (df['FAST_SMA'].shift(1)>df['SLOW_SMA'].shift(1)))]
            
            choices_sma = [1,-1]
            df['SMA_Signal'] = np.select(cond_sma,choices_sma,default=0)
            df.loc[df['SMA_Signal'].diff() != 0, 'Signal_Change'] = df['SMA_Signal']
            
        # rsı
        if indicator=='RSI':
            rsi_period= st.sidebar.slider('RSI Period',3,200,value=14)
            df['RSI'] = ta.rsi(df['Close'] ,rsi_period )
            cond_rsi=[df['RSI']<30,
                      df['RSI']>70]
            
            choices_rsi= [1,-1]
            df['RSI_Signal'] = np.select(cond_rsi,choices_rsi,default=0)
            df.loc[df['RSI_Signal'].diff() != 0, 'Signal_Change'] = df['RSI_Signal']
            
        # kama
        if indicator=='KAMA':
            df['KAMA'] = ta.kama(df['Close'])
            cond_kama=[((df['Close'] > df['KAMA']) & (df['Close'].shift(1) < df['KAMA'].shift(1))),
                      ((df['Close'] < df['KAMA']) & (df['Close'].shift(1) > df['KAMA'].shift(1)))]
            
            choices_kama= [1,-1]
            df['KAMA_Signal'] = np.select(cond_kama,choices_kama,default=0)
            df.loc[df['KAMA_Signal'].diff() != 0, 'Signal_Change'] = df['KAMA_Signal']
            
        # bband
        if indicator=='Bollinger Band':
            basis,upper,lower = ta.bband(20,'SMA',df['Close'])
            df['basis'],df['upper'],df['lower'] = basis,upper,lower
            cond_bband=[df['lower'] >= df['Close'],
                      df['Close']>=df['upper']]
            
            choices_bband= [1,-1]
            df['bband_Signal'] = np.select(cond_bband,choices_bband,default=0)
            df.loc[df['bband_Signal'].diff() != 0, 'Signal_Change'] = df['bband_Signal']
            
        # supertrend
        if indicator=='Supertrend':
            supertrend_value =st.sidebar.slider('Supertrend Period',0,20,value=10)
            supertrend_mult = st.sidebar.slider('Supertrend Multiplier',2,30,value=3)
            df[['Trend','_dir','Long','Short']] = ta.supertrend(df,supertrend_value,supertrend_mult)
            
            cond_supertrend=[df['_dir']>0 ,
                      df['_dir']<0]
            
            choices_supertrend= [1,-1]
            df['supertrend_Signal'] = np.select(cond_supertrend,choices_supertrend,default=0)
            df.loc[df['supertrend_Signal'].diff() != 0, 'Signal_Change'] = df['supertrend_Signal']
            
        if indicator == 'ATR':
            
            df['ATR'] =ta.calculate_atr(df,14)
        

        if indicator == 'ATR & T3':
            atr_value = st.sidebar.slider('ATR Period',0,30,value=14)
            t3_value =st.sidebar.slider('T3 Period',0,20,value=3)
            t3_factor_value = st.sidebar.slider('T3 Factor Period',0.01,1.0,value=0.7,step=0.05)
            
            df['ATR'] =ta.calculate_atr(df,atr_value)
            df['t3'] = ta.t3(df['Close'],t3_value,0.7)
            
            df['t3_buy'] = (df['t3'] > df['t3'].shift(1)) & (df['t3'].shift(1) <= df['t3'].shift(2))
            df['t3_sell'] = (df['t3'] < df['t3'].shift(1)) & (df['t3'].shift(1) >= df['t3'].shift(2))
            
            cond_t3_Atr=[(df['t3_buy']) & (df['ATR'] > 2),
                         (df['t3_sell'])]
            
            choices_t3_Atr= [1,-1]
            df['T3_ATR_Signal'] = np.select(cond_t3_Atr,choices_t3_Atr,default=0)
            df.loc[df['T3_ATR_Signal'].diff() != 0, 'Signal_Change'] = df['T3_ATR_Signal']


        return df
    except:
        raise
    

def backtesting(ticker, signal, initial_price, commissions,start,end,intervals,indicator):
        """
        Bir alım-satım stratejisini backtest yapar.
        df: Pandas DataFrame, 'close' ve 'signal' sütunları içerir (1=al, -1=sat, 0=tut).
        signal: Sinyal sütununun adı (string).
        initial_price: Başlangıç sermayesi (int).
        commissions: İşlem başına komisyon oranı (float, örneğin 0.001 = %0.1).
        """
        # Giriş türü kontrolleri
        
        if not isinstance(signal, str):
            raise TypeError("signal bir string olmalı.")
        if not isinstance(initial_price, int):
            raise TypeError("initial_price bir tamsayı olmalı.")
        if not isinstance(commissions, float):
            raise TypeError("commissions bir float olmalı.")
        l=[]
        try:
            for interval in intervals:   
               
                df=get_stock(ticker,start=start,end=end,interval=interval,indicator=indicator)
                
        
                # Portföy durumunu takip et
                cash = initial_price  # Nakit miktarı
                position = 0  # Sahip olunan varlık miktarı
                portfolio_value = []  # Her adımda portföy değerini sakla
                alım_sayısı = 0
                satım_sayısı = 0
                for i, row in df.iterrows():
                    price = row['Close']
                    sig = row[signal]
                    
                    # Portföy değerini hesapla (nakit + varlıklar)
                    current_value = cash + position * price
                    portfolio_value.append(current_value)
                    
                    # Sinyal işleme
                    if sig == 1:  # Al sinyali
                        if position == 0:  # Henüz pozisyon yoksa
                            # Al: Tüm nakit ile varlık al
                            position = cash / price
                            commission_cost =  commissions
                            cash -= commission_cost
                            cash = 0  # Tüm nakit kullanıldı
                            alım_sayısı+=1
                    elif sig == -1:  # Sat sinyali
                        if position > 0:  # Pozisyon varsa
                            # Sat: Tüm varlığı sat
                            cash = position * price
                            commission_cost =  commissions
                            cash -= commission_cost
                            position = 0
                            satım_sayısı+=1

                # Son portföy değerini hesapla
                final_value = cash + position * df['Close'].iloc[-1]
                l.append({
                    'ticker':ticker,'interval':interval,
                    'final_value': final_value,
                    'profit': final_value - initial_price,
                    'profit_percentage': str((final_value - initial_price) / initial_price * 100)+"%",
                    'Alım_Sayısı' : alım_sayısı,
                    'Satım_Sayısı' : satım_sayısı
                    
                })
            return pd.DataFrame.from_dict(l)
        except:
            raise
st.header('Backtesting Dashboard')

intervals = st.sidebar.multiselect('Interval', ['1h','4h','1d','5d','1wk','1mo','3mo'])
intervals =list(intervals)
indicator = st.sidebar.selectbox('Indicator',['T3','EMA','SMA','RSI','KAMA','Close_f_EMA','Bollinger Band','Supertrend','ATR & T3'])

                            

stock = st.text_input('Stock Code','IREN')
start_Date=st.sidebar.date_input('Start Date',value=datetime.datetime(2025,1,1))
end_Date=st.sidebar.date_input('End Date')

tickers_nasdaq = ['AAOI',
 'AAPL',
 'ABTC',
 'ABTS',
 'ACIW',
 'ACLS',
 'ACMR',
 'ADAM',
 'ADBE',
 'ADEA',
 'ADI',
 'ADSK',
 'AEIS',
 'AEYE',
 'AGMH',
 'AGYS',
 'AI',
 'AIFF',
 'AIP',
 'AIRG',
 'AISP',
 'AISPW',
 'AIXI',
 'ALAB',
 'ALAR',
 'ALGM',
 'ALKT',
 'ALMU',
 'ALOT',
 'ALRM',
 'AMAT',
 'AMBA',
 'AMBQ',
 'AMD',
 'AMKR',
 'AMPL',
 'AMST',
 'AOSL',
 'APH',
 'API',
 'APP',
 'APPF',
 'APPN',
 'ARBB',
 'ARBE',
 'ARBEW',
 'AREN',
 'ARM',
 'ARQQ',
 'ARQQW',
 'ARW',
 'ASAN',
 'ASML',
 'ASST',
 'ASTI',
 'ASUR',
 'ASX',
 'ASYS',
 'ATGL',
 'ATHM',
 'ATHR',
 'ATOM',
 'AUID',
 'AUR',
 'AUROW',
 'AUUD',
 'AUUDW',
 'AVDX',
 'AVGO',
 'AVNW',
 'AVPT',
 'AVT',
 'AWRE',
 'AXTI',
 'AZTA',
 'BAND',
 'BASE',
 'BB',
 'BBAI',
 'BEEM',
 'BHE',
 'BIDU',
 'BILI',
 'BILL',
 'BINI',
 'BKSY',
 'BKTI',
 'BKYI',
 'BL',
 'BLBX',
 'BLIN',
 'BLIV',
 'BLKB',
 'BLND',
 'BLZE',
 'BMBL',
 'BMR',
 'BNAI',
 'BNAIW',
 'BNZI',
 'BNZIW',
 'BOX',
 'BRAG',
 'BRZE',
 'BSY',
 'BTCM',
 'BZ',
 'CACI',
 'CAMT',
 'CAN',
 'CANG',
 'CARG',
 'CARS',
 'CCCS',
 'CCLD',
 'CCLDO',
 'CCRD',
 'CCSI',
 'CDLX',
 'CDNS',
 'CDW',
 'CERS',
 'CERT',
 'CETX',
 'CEVA',
 'CFLT',
 'CGNT',
 'CGTL',
 'CHKP',
 'CHR',
 'CINT',
 'CLBT',
 'CLMB',
 'CLPS',
 'CLS',
 'CLVT',
 'CMBM',
 'CMCM',
 'CMRC',
 'CMTL',
 'CNET',
 'CNXC',
 'COHR',
 'COMM',
 'COMP',
 'COUR',
 'CRCT',
 'CRDO',
 'CREX',
 'CRM',
 'CRNC',
 'CRNT',
 'CRSR',
 'CRUS',
 'CRWD',
 'CRWV',
 'CSAI',
 'CSGS',
 'CSIQ',
 'CSPI',
 'CTS',
 'CTSH',
 'CTW',
 'CVLT',
 'CVV',
 'CW',
 'CWAN',
 'CXAI',
 'CXAIW',
 'CXM',
 'CYBR',
 'CYCU',
 'CYCUW',
 'CYN',
 'DAIC',
 'DAICW',
 'DASH',
 'DAVA',
 'DAY',
 'DBX',
 'DCBO',
 'DDD',
 'DDI',
 'DDOG',
 'DELL',
 'DFSC',
 'DFSCW',
 'DGLY',
 'DGNX',
 'DH',
 'DIOD',
 'DJT',
 'DJTWW',
 'DKI',
 'DMRC',
 'DOCN',
 'DOCS',
 'DOCU',
 'DOMO',
 'DOX',
 'DOYU',
 'DQ',
 'DSGX',
 'DSP',
 'DT',
 'DTSS',
 'DTST',
 'DTSTW',
 'DUOL',
 'DUOT',
 'DV',
 'DVLT',
 'DXC',
 'EB',
 'EBON',
 'ECX',
 'ECXWW',
 'EGAN',
 'EGHT',
 'ELTK',
 'ELWS',
 'EMR',
 'ENPH',
 'ENS',
 'EPAC',
 'EPAM',
 'ERIC',
 'ERII',
 'ESP',
 'ESTC',
 'ETN',
 'EVCM',
 'EVER',
 'EVGO',
 'EVGOW',
 'EVLV',
 'EVLVW',
 'EVTC',
 'EXFY',
 'FA',
 'FAAS',
 'FAASW',
 'FATN',
 'FDS',
 'FIG',
 'FIVN',
 'FLEX',
 'FLUT',
 'FNGR',
 'FORA',
 'FORM',
 'FORTY',
 'FROG',
 'FRSH',
 'FRSX',
 'FSLR',
 'FSLY',
 'FTCI',
 'FTNT',
 'GCL',
 'GCLWW',
 'GCTS',
 'GDDY',
 'GDEV',
 'GDEVW',
 'GDRX',
 'GDS',
 'GDYN',
 'GE',
 'GEG',
 'GEGGL',
 'GEN',
 'GENVR',
 'GFS',
 'GIBO',
 'GIBOW',
 'GIGM',
 'GILT',
 'GITS',
 'GLBE',
 'GLE',
 'GLOB',
 'GMGI',
 'GMHS',
 'GMM',
 'GOOG',
 'GOOGL',
 'GPUS',
 'GRND',
 'GRNQ',
 'GRRR',
 'GRRRW',
 'GSIT',
 'GTLB',
 'GTM',
 'GWRE',
 'GXAI',
 'HCAT',
 'HCTI',
 'HIMX',
 'HKIT',
 'HLIT',
 'HNGE',
 'HOLO',
 'HOLOW',
 'HPAI',
 'HPAIW',
 'HPE',
 'HPQ',
 'HSTM',
 'HTCR',
 'HUBB',
 'HUBS',
 'HUYA',
 'IAC',
 'IAS',
 'IBEX',
 'IBM',
 'ICG',
 'ICHR',
 'IDAI',
 'IDN',
 'IFBD',
 'IMMR',
 'IMOS',
 'INDI',
 'INFA',
 'INFY',
 'INGM',
 'INLX',
 'INOD',
 'INSE',
 'INTA',
 'INTC',
 'INTU',
 'INVE',
 'IONQ',
 'IOT',
 'IPDN',
 'IPGP',
 'IPWR',
 'IREN',
 'ISSC',
 'ITRN',
 'IVDA',
 'IVDAW',
 'JAMF',
 'JBL',
 'JG',
 'JKHY',
 'JKS',
 'JOYY',
 'KARO',
 'KC',
 'KD',
 'KE',
 'KLAC',
 'KLIC',
 'KLTR',
 'KOPN',
 'KTCC',
 'KULR',
 'KVHI',
 'KVYO',
 'LAES',
 'LASR',
 'LAW',
 'LAWR',
 'LCFY',
 'LCFYW',
 'LDOS',
 'LEDS',
 'LGCL',
 'LGL',
 'LHSW',
 'LINK',
 'LIQT',
 'LNKS',
 'LNW',
 'LOGI',
 'LPL',
 'LPSN',
 'LPTH',
 'LRCX',
 'LSCC',
 'LSPD',
 'LTRYW',
 'LZ',
 'LZMH',
 'MANH',
 'MAPS',
 'MAPSW',
 'MARA',
 'MAXN',
 'MBLY',
 'MCHP',
 'MCHPP',
 'MDB',
 'MEI',
 'META',
 'MFI',
 'MGIC',
 'MGNI',
 'MGRT',
 'MITK',
 'MKDW',
 'MKDWW',
 'MKTW',
 'MLGO',
 'MLNK',
 'MNDO',
 'MNDR',
 'MNDY',
 'MOBX',
 'MOBXW',
 'MOMO',
 'MPTI',
 'MPWR',
 'MQ',
 'MRAM',
 'MRCY',
 'MRVL',
 'MSAI',
 'MSAIW',
 'MSFT',
 'MSGM',
 'MSI',
 'MSPR',
 'MSPRW',
 'MSPRZ',
 'MSTR',
 'MTC',
 'MTCH',
 'MTLS',
 'MTSI',
 'MU',
 'MVIS',
 'MX',
 'MXL',
 'MYPS',
 'MYPSW',
 'MYSZ',
 'NABL',
 'NBIS',
 'NCNO',
 'NEE',
 'NEON',
 'NET',
 'NEXN',
 'NICE',
 'NIQ',
 'NIXX',
 'NIXXW',
 'NNDM',
 'NOK',
 'NOW',
 'NRDS',
 'NSYS',
 'NTAP',
 'NTCL',
 'NTCT',
 'NTES',
 'NTNX',
 'NTWK',
 'NVDA',
 'NVEC',
 'NVMI',
 'NVNI',
 'NVTS',
 'NXDR',
 'NXPI',
 'NXTT',
 'NYAX',
 'OBLG',
 'OCFT',
 'ODYS',
 'OKTA',
 'OLED',
 'OMCL',
 'ON',
 'ONDS',
 'ONFO',
 'ONFOW',
 'ONTF',
 'OOMA',
 'OPRA',
 'OPTX',
 'OPTXW',
 'OPXS',
 'ORCL',
 'ORKT',
 'OS',
 'OSIS',
 'OSPN',
 'OSS',
 'OST',
 'OTEX',
 'OTIS',
 'PAGS',
 'PANW',
 'PATH',
 'PAYC',
 'PCOR',
 'PCTY',
 'PD',
 'PDD',
 'PDFS',
 'PDYN',
 'PDYNW',
 'PEGA',
 'PENG',
 'PERF',
 'PERI',
 'PHUN',
 'PI',
 'PINS',
 'PL',
 'PLAB',
 'PLTK',
 'PLTR',
 'PLUS',
 'PLXS',
 'PN',
 'PODC',
 'POET',
 'PONY',
 'POWI',
 'PRCH',
 'PRGS',
 'PRO',
 'PRSO',
 'PSN',
 'PSTG',
 'PT',
 'PTC',
 'PUBM',
 'PXLW',
 'QBTS',
 'QCOM',
 'QLYS',
 'QMCO',
 'QRVO',
 'QTWO',
 'QUBT',
 'QUIK',
 'RAIN',
 'RAINW',
 'RAMP',
 'RBBN',
 'RBLX',
 'RBRK',
 'RCAT',
 'RDCM',
 'RDDT',
 'RDNW',
 'RDVT',
 'RELL',
 'RFAIR',
 'RFIL',
 'RGTI',
 'RGTIW',
 'RMBS',
 'RMSG',
 'RMSGW',
 'RNG',
 'ROP',
 'RPD',
 'RUM',
 'RUMBW',
 'RXT',
 'RYET',
 'RZLV',
 'RZLVW',
 'S',
 'SABR',
 'SAGT',
 'SAIC',
 'SAIH',
 'SAIHW',
 'SAIL',
 'SANG',
 'SANM',
 'SAP',
 'SATL',
 'SATLW',
 'SBET',
 'SCKT',
 'SCSC',
 'SEDG',
 'SEGG',
 'SELX',
 'SEMR',
 'SGN',
 'SHLS',
 'SHMD',
 'SHMDW',
 'SHOP',
 'SIFY',
 'SIMO',
 'SITM',
 'SJ',
 'SKIL',
 'SKLZ',
 'SKYT',
 'SLAB',
 'SLNH',
 'SLNHP',
 'SLP',
 'SMCI',
 'SMRT',
 'SMSI',
 'SMTC',
 'SMTK',
 'SMWB',
 'SMX',
 'SMXWW',
 'SNAL',
 'SNAP',
 'SNCR',
 'SNDK',
 'SNOW',
 'SNPS',
 'SNX',
 'SOGP',
 'SOHU',
 'SOL',
 'SOTK',
 'SOUN',
 'SOUNW',
 'SPCB',
 'SPNS',
 'SPSC',
 'SPT',
 'SPWR',
 'SPWRW',
 'SQNS',
 'SRAD',
 'SSNC',
 'SST',
 'SSTI',
 'SSTK',
 'SSYS',
 'STM',
 'STNE',
 'STRC',
 'STRD',
 'STRF',
 'STRK',
 'STX',
 'SVCO',
 'SVRE',
 'SWKS',
 'SY',
 'SYNA',
 'TACT',
 'TAIT',
 'TAOP',
 'TASK',
 'TBLA',
 'TBLAW',
 'TBRG',
 'TCX',
 'TDC',
 'TEAD',
 'TEAM',
 'TEL',
 'TEM',
 'TENB',
 'TGHL',
 'TIXT',
 'TLS',
 'TOST',
 'TRAK',
 'TRIP',
 'TRNR',
 'TRT',
 'TRUE',
 'TRVG',
 'TSEM',
 'TSM',
 'TTAN',
 'TTD',
 'TTMI',
 'TUYA',
 'TWLO',
 'TXN',
 'TYGO',
 'TYL',
 'TZUP',
 'U',
 'UBXG',
 'UCTT',
 'UI',
 'UIS',
 'ULY',
 'UMAC',
 'UMC',
 'UPLD',
 'UPWK',
 'UUU',
 'VBIX',
 'VECO',
 'VEEV',
 'VELO',
 'VERI',
 'VERX',
 'VIAV',
 'VICR',
 'VLN',
 'VMEO',
 'VNET',
 'VPG',
 'VRAR',
 'VREX',
 'VRME',
 'VRNS',
 'VRNT',
 'VRSN',
 'VRT',
 'VS',
 'VSAT',
 'VSH',
 'VSSYW',
 'VTEX',
 'VUZI',
 'VWAV',
 'VWAVW',
 'WATT',
 'WAY',
 'WB',
 'WCT',
 'WDAY',
 'WDC',
 'WEAV',
 'WETH',
 'WFCF',
 'WGS',
 'WGSWW',
 'WIMI',
 'WIT',
 'WIX',
 'WK',
 'WKEY',
 'WOLF',
 'WRD',
 'WTO',
 'WULF',
 'WYY',
 'XNET',
 'XPER',
 'XRX',
 'XTIA',
 'XTKG',
 'XYZ',
 'YAAS',
 'YALA',
 'YEXT',
 'YMM',
 'YMT',
 'YOU',
 'YXT',
 'ZDGE',
 'ZENA',
 'ZENV',
 'ZEPP',
 'ZETA',
 'ZIP',
 'ZM',
 'ZOOZ',
 'ZOOZW',
 'ZS',
 'ZSPC']

initial_price=int(st.sidebar.text_input('Initial Price', value='10000'))

st.dataframe(backtesting(ticker=stock, signal='Signal_Change', initial_price=initial_price, commissions=1.5,start=start_Date.strftime('%Y-%m-%d'),end=end_Date.strftime('%Y-%m-%d'),intervals=intervals,indicator=indicator))
























