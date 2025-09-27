# -*- coding: utf-8 -*-
"""
Created on Sat Sep 27 18:07:19 2025

@author: Ozgr
"""

import pandas as pd
import numpy as np
import yfinance as yf
import tech_analysis as ta
import streamlit as st

def get_stock(ticker,start,end,interval):
    if not isinstance(ticker, str):
        raise TypeError("ticker bir string olmalı.")
        
    try:
        df = yf.download(ticker,start=start,end=end,interval=interval)
        df.columns = df.columns.droplevel(1)
        df['T3'] = ta.t3(df['Close'],3,0.7)
        cond = [df['T3']>df['T3'].shift(1),
               df['T3']<df['T3'].shift(1)]
        choice = [1,-1]
        df['T3_Signal'] = np.select(cond,choice,default=0)
        df.loc[df['T3_Signal'].diff() != 0, 'T3_Signal_Change'] = df['T3_Signal']
        return df
    except:
        raise
    

def backtesting(ticker, signal, initial_price, commissions,start,end,intervals):
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
               
                df=get_stock(ticker,start=start,end=end,interval=interval)
           
                
        
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
stock = st.text_input('Stock Code','IREN')
start_Date=st.sidebar.date_input('Start Date')
end_Date=st.sidebar.date_input('End Date')


st.dataframe(backtesting(ticker=stock, signal='T3_Signal_Change', initial_price=10000, commissions=1.5,start_Date=start_Date.strftime('%Y-%m-%d'),end_Date=end_Date.strftime('%Y-%m-%d'),intervals=(intervals)))
























