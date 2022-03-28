import streamlit as st
import os
import io
import os.path

import shutil
import zipfile

import datetime
import pandas as pd
import numpy as np
from scipy import *
import matplotlib.pyplot as plt
from matplotlib.ticker import *
import matplotlib.ticker
import matplotlib.dates as mdates


import loop_functions



plt.rcParams['font.family']= "DejaVu Serif"
plt.rcParams['font.sans-serif'] = ['Helvetica']
plt.rcParams['mathtext.fontset'] = 'stix' # math fontの設定
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.linewidth'] = 1.5 #軸の太さを設定。目盛りは変わらない
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['xtick.direction'] = "in"
plt.rcParams['ytick.direction'] = "in"
plt.rcParams['xtick.major.size'] = 6.0
plt.rcParams['xtick.minor.size'] = 4.0
plt.rcParams['ytick.major.size'] = 6.0
plt.rcParams['ytick.minor.size'] = 4.0
plt.rcParams['xtick.major.pad']='8'
plt.rcParams['xtick.top'] = True
plt.rcParams['ytick.right'] = True
plt.rcParams['axes.edgecolor'] = '#08192D' # 枠の色
plt.rcParams['axes.labelcolor'] = '#08192D' # labelの色
plt.rcParams['xtick.color'] = '#08192D' # xticksの色
plt.rcParams['ytick.color'] = '#08192D' # yticksの色
plt.rcParams['text.color'] = '#08192D' # annotate, labelの色
plt.rcParams['legend.framealpha'] = 1.0 # legendの枠の透明度
plt.rcParams['pdf.fonttype'] = 42

    


def main():
    
    st.title('Local Outlier Probabilities (LoOPs) 計算アプリ')
    
    st.markdown('### CSVファイルをアップロードしてLoOPを計算')
    
    filetype = 'nontime'
    if st.checkbox('時系列データの場合はチェック（CSVファイルの1列目は日時のデータとして扱われます）'):
        filetype = 'time'

    

    uploaded_file = st.file_uploader(label='', type=['csv'])
    st.write('input: ', uploaded_file)

    if uploaded_file is not None:
        
        df = pd.read_csv(uploaded_file)
        st.write(df)

        if filetype=='time':
            import dateutil.parser
            date = df.values[:,0]
            Date = []
            for _ in range(len(date)):

                tdatetime = dateutil.parser.parse(date[_]) 
                Date.append(datetime.datetime(tdatetime.year, tdatetime.month, tdatetime.day))

            X = np.array(df.values[:,1:]).astype(np.float32)
            Nparams = X.shape[1]
        else:
            X = np.array(df.values[:,:]).astype(np.float32)
            Date = list( range(X.shape[0]) )
            Nparams = X.shape[1]

        if X.shape[1]==1:
            X = np.stack([X[:,0], np.zeros(X.shape[0])], 1) 
            
        else:
            for col in range(X.shape[1]):
                Q1 = np.nanpercentile(X[:,col], 25)
                Q2 = np.nanpercentile(X[:,col], 50)
                Q3 = np.nanpercentile(X[:,col], 75)
                X[:,col] = (X[:,col]-Q2)/(Q3-Q1)
               
        
        st.markdown('#### LoOP計算のパラメータを設定')
        col1, col2, col3 = st.columns(3)

        col1.markdown('##### neighborhood size')
        n_neighbors = col1.slider("", 3, X.shape[0]-1, 10, 1)

        col2.markdown('##### ')

        col3.markdown('##### extent')
        extent = col3.radio('',[3,2,1])
        
        scores = loop_functions.LocalOutlierProbability(X, extent=extent, n_neighbors=n_neighbors, use_numba=True).fit().local_outlier_probabilities
        scores *= 100
        
        st.markdown('#### LoOPの計算結果のプロット')
        Npanels = Nparams+1
        fig = plt.figure(figsize=(10, 2.5*Npanels))
        ax1 = plt.subplot(Npanels,1,1)

        ax1.plot(Date, scores, 'o-')
        if filetype=='time':
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%Y'))
        else:
            ax1.set_xlabel('Data index', fontsize=16)

        ax1.set_xlim(Date[0], Date[-1])
        ax1.set_ylim(0,100)
        ax1.set_ylabel('LoOP [%]', fontsize=16)

        for cc, row in enumerate(range(2,Nparams+2)):
            ax2 = plt.subplot(Npanels,1,row)
            ax2.plot(Date, X[:,cc], color='C'+str(cc+1), marker='o')
            if filetype=='time':
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%Y'))
            else:
                ax2.set_xlabel('Data index', fontsize=16)

            ax2.set_xlim(Date[0], Date[-1])
            ax2.set_ylabel('Obs. '+str(cc+1), fontsize=16)
    

        plt.tight_layout()
        st.pyplot(fig)


        
        st.markdown('### LoOPの計算結果をCSVファイルでダウンロード')
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')

        csv = convert_df(pd.DataFrame({'date': Date, 'LoOP': scores}))
        st.download_button(
            label="Download calculated LoOPs as CSV",
            data=csv,
            file_name='loop_out.csv',
            mime='text/csv',
        )

        st.markdown('### サンプルコードをダウンロード')
        # Create an in-memory buffer to store the zip file
        with io.BytesIO() as buffer:
            # Write the zip file to the buffer
            with zipfile.ZipFile(buffer, "w") as zip:
                zip.writestr("LoOP_results.csv", csv)
                zip.writestr(uploaded_file.name, convert_df(df))
                
                Codes = ''
                with open("loop_functions.py", encoding='utf8') as f:
                    for s_line in f:
                        Codes += s_line
                
                zip.writestr("loop_functions.py", Codes)

                Codes = ''
                path = 'template.py'
                csv_name = uploaded_file.name
                with open(path, encoding='utf8') as f:
                    for s_line in f:
                        
                        if 'csv_name =' in s_line:
                            Codes += '    csv_name = "'+csv_name+'"\n'
                        elif 'extent =' in s_line:
                            Codes += '    extent = '+str(int(extent))+'\n'
                        elif 'n_neighbors =' in s_line:
                            Codes += '    n_neighbors = '+str(n_neighbors)+'\n'
                        else:
                            Codes += s_line
                
                zip.writestr("sample_main.py", Codes)


            buffer.seek(0)

            st.download_button(
                label="Download sample code as ZIP",
                data=buffer,  # Download buffer
                file_name="sample.zip" 
            )

        

                



if __name__ == '__main__':
 
    main()
    