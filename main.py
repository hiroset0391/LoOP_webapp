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
from PIL import Image
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

def normalize(X):
    if X.shape[1]==1:
        X = np.stack([X[:,0], np.zeros(X.shape[0])], 1) 
        
    else:
        for col in range(X.shape[1]):
            ### robust z score
            # Q1 = np.nanpercentile(X[:,col], 25)
            # Q2 = np.nanpercentile(X[:,col], 50)
            # Q3 = np.nanpercentile(X[:,col], 75)
            # X[:,col] = (X[:,col]-Q2)/(Q3-Q1)

            X[:,col] = (X[:,col]-np.nanmean(X[:,col])) / np.nanstd(X[:,col])
    
    return X

def remove_nan(X, Date):
    nnan_idx = ~np.isnan(X).any(axis=1)
    X = X[nnan_idx, :]
    Date = Date[nnan_idx]
    return X, Date

def score_fillnan(scores, Date, Date_scores):
    
    scores_new = np.zeros(len(Date))*np.nan
    for dd, fday in enumerate(Date):
        idx = np.where(Date_scores==fday)[0]
        if len(idx)>0:
            scores_new[dd] = scores[idx]  
        else:
            pass
    return scores_new


def compute_loop(refperid, date_arr, currdate, X_in, extent=3, n_neighbors=30):

    loop_val = np.nan

    
    if (currdate-datetime.datetime.now()).days< -1:

        ref_ini_idx = np.where(refperid[0]==date_arr)[0][0]
        ref_fini_idx = np.where(refperid[1]==date_arr)[0][0]+1
        curr_idx = np.where(currdate==date_arr)[0][0]

        X_ref = X_in[ref_ini_idx:ref_fini_idx,:]
        date_ref = date_arr[ref_ini_idx:ref_fini_idx]

        X_curr = X_in[curr_idx,:].reshape(1,X_in.shape[1])
        
        if np.mean(X_curr)==np.mean(X_curr):
        
            if ref_ini_idx<=curr_idx<=ref_fini_idx-1:
                del_idx = np.where(currdate==date_ref)[0][0]
                X_ref = np.delete(X_ref, del_idx, axis=0)
                
            X = np.append(X_ref, X_curr, axis=0) 
            index = ~np.isnan(X).any(axis=1)
            X = X[index, :]

            X = normalize(X)

    
            scores = loop_functions.LocalOutlierProbability(X, extent=extent, n_neighbors=n_neighbors, use_numba=True).fit().local_outlier_probabilities
            loop_val = scores[-1]
        
            
    

    return loop_val


def make_plot(Date, scores, X, filetype, Nparams):
    Npanels = Nparams+1
    Ndays = len(Date)
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

    if filetype == 'time':
        if Ndays>=90:
            ax1.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
        else:
            ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=1))

    for cc, row in enumerate(range(2,Nparams+2)):
        ax2 = plt.subplot(Npanels,1,row)
        ax2.plot(Date, X[:,cc], color='C'+str(cc+1), marker='o')
        if filetype=='time':
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d\n%Y'))

            if filetype == 'time':
                if Ndays>=90:
                    ax2.xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
                else:
                    ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=1))


        else:
            ax2.set_xlabel('Data index', fontsize=16)

        ax2.set_xlim(Date[0], Date[-1])
        ax2.set_ylabel('Obs. '+str(cc+1), fontsize=16)


    plt.tight_layout()
    st.pyplot(fig)
    
    
@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def main():

    st.title('Local Outlier Probabilities (LoOPs) 計算アプリ')
    
    st.markdown('### CSVファイルをアップロードしてLoOPを計算')
    

    uploaded_file = st.file_uploader(label='', type=['csv'])
    st.write('input: ', uploaded_file)

    if uploaded_file is not None:
        
        df_in = pd.read_csv(uploaded_file)
        st.write(df_in)

        try:
            filetype = 'time'
            import dateutil.parser
            date = df_in.values[:,0]
            Date = []
            for _ in range(len(date)):

                tdatetime = dateutil.parser.parse(date[_]) 
                Date.append(datetime.datetime(tdatetime.year, tdatetime.month, tdatetime.day))
            
            Date = np.array(Date)
            X = np.array(df_in.values[:,1:]).astype(np.float32)
            Nparams = X.shape[1]

        except:
            filetype = 'nontime'
            X = np.array(df_in.values[:,:]).astype(np.float32)
            Date = np.array( range(X.shape[0]) )
            Nparams = X.shape[1]
            

        X_scores, Date_scores = remove_nan(X, Date)
        X_scores = normalize(X_scores) 

        
        st.markdown('#### LoOP計算のパラメータを設定')
        col1, col2, col3 = st.columns(3)

        col1.markdown('##### **neighborhood size**')
        col1.markdown('LoOP計算に用いる近傍点の数。小さすぎるとLoOPの計算が不安定になることがある。')
        n_neighbors = col1.number_input(label='',
                        value=10) #col1.slider("", 3, X_scores.shape[0]-1, 10, 1)

        col2.markdown('##### ')

        col3.markdown('##### **extent**')
        col3.markdown('値が小さいほどLoOPが高く算出される。extent=3がよく使われる。')
        extent = col3.radio('',[3,2,1])

        st.markdown('#### LoOP計算のQuiet Period（Reference data）を設定')
        st.markdown('Quiet Periodのデータとの比較からCurrent day (Current data-index)のデータのLoOPを計算する。')
        
        with st.expander('See details'):
            image = Image.open('schematic.png')
            st.image(image, caption='',use_column_width=True)


        col1, col2, col3 = st.columns(3)
        
        if filetype == 'time':
            col1.markdown('##### **Start (date)**')
            col3.markdown('##### **End (date)**')
        
            startday_quiet = col1.date_input('',
                                min_value=Date[0],
                                max_value=Date[-1],
                                value=Date[0],
                                )
            
            col2.markdown('##### ')

            endday_quiet = col3.date_input('',
                                min_value=Date[0],
                                max_value=Date[-1],
                                value=Date[-1],
                                )
        else:
            col1.markdown('##### **Start (index)**')
            col3.markdown('##### **End (index)**')

            startday_quiet = col1.number_input(label='', value=0)
            
            col2.markdown('##### ')

            endday_quiet = col3.number_input(label='', value=len(Date)-1)

    
        if 'initial_load'  not in st.session_state:
            # 初期化処理を示す状態変数
            st.session_state.initial_load = True

        is_click = st.button('Calculate LoOP')
        
        ### 初回。LoOPは必ず新規計算。
            
        if filetype == 'time':
            startday_quiet = datetime.datetime(startday_quiet.year, startday_quiet.month, startday_quiet.day)
            endday_quiet = datetime.datetime(endday_quiet.year, endday_quiet.month, endday_quiet.day)
        else:
            pass

        if startday_quiet==Date[0] and endday_quiet==Date[-1]:
            scores = loop_functions.LocalOutlierProbability(X_scores, extent=extent, n_neighbors=n_neighbors, use_numba=True).fit().local_outlier_probabilities
            scores *= 100
        else:
            scores = []
            for currdate in Date:
                daily_loop = compute_loop([startday_quiet, endday_quiet], Date, currdate, X, extent=extent, n_neighbors=n_neighbors)
                scores.append(daily_loop*100)
            scores = np.array(scores)
    
    

        scores = score_fillnan(scores, Date, Date_scores)
        

        st.markdown('#### LoOPの計算結果のプロット')
        st.markdown('1段目:LoOP,  2段目以降:LoOPの計算に使用したデータ')
        
        make_plot(Date, scores, X, filetype, Nparams)


        if filetype=='time':
            df_outcsv = pd.DataFrame({'date': Date, 'LoOP': scores})
        else:
            df_outcsv = pd.DataFrame({'data_index': Date, 'LoOP': scores})
        
        st.write(df_outcsv)

        
        st.markdown('### LoOPの計算結果をCSVファイルでダウンロード')
        
        csv = convert_df(df_outcsv)

        st.download_button(
            label="Download calculated LoOPs as CSV",
            data=csv,
            file_name='LoOP_result.csv',
            mime='text/csv'
        )
        
        st.markdown('### サンプルコードをダウンロード')
        # Create an in-memory buffer to store the zip file
        with io.BytesIO() as buffer:
            # Write the zip file to the buffer
            with zipfile.ZipFile(buffer, "w") as zip:
                zip.writestr("LoOP_result.csv", csv)
                zip.writestr(uploaded_file.name, convert_df(df_in))
                
                Codes = ''
                with open("loop_functions.py", encoding='utf8') as f:
                    for s_line in f:
                        Codes += s_line
                
                zip.writestr("loop_functions.py", Codes)
                f.close()

                Codes = ''
                path = 'template.py'
                csv_name = uploaded_file.name
                filetype_bool = False
                if filetype=='time':
                    filetype_bool = True

                with open(path, encoding='utf8') as f:
                    for s_line in f:
                        
                        if 'csv_name =' in s_line:
                            Codes += '    csv_name = "'+csv_name+'"\n'
                        elif 'extent =' in s_line:
                            Codes += '    extent = '+str(int(extent))+'\n'
                        elif 'n_neighbors =' in s_line:
                            Codes += '    n_neighbors = '+str(n_neighbors)+'\n'
                        elif 'time_series =' in s_line:
                            Codes += '    time_series = '+str(filetype_bool)+'\n'
                        else:
                            Codes += s_line
                
                zip.writestr("sample_main.py", Codes)
                f.close()

            buffer.seek(0)

            st.download_button(
                label="Download sample code as ZIP",
                data=buffer,  # Download buffer
                file_name="sample.zip" 
            )

        

        ### 初期状態を示す状態変数をFalse
        st.session_state.initial_load = False

        st.session_state.date = Date
        st.session_state.score = scores
            
            
        # ### 初回。
        # if st.session_state.initial_load:
        #     ### 初回。LoOPは必ず新規計算。
        #     if is_click and st.session_state.initial_load==True:
        #         if filetype == 'time':
        #             startday_quiet = datetime.datetime(startday_quiet.year, startday_quiet.month, startday_quiet.day)
        #             endday_quiet = datetime.datetime(endday_quiet.year, endday_quiet.month, endday_quiet.day)
        #         else:
        #             pass

        #         if startday_quiet==Date[0] and endday_quiet==Date[-1]:
        #             scores = loop_functions.LocalOutlierProbability(X_scores, extent=extent, n_neighbors=n_neighbors, use_numba=True).fit().local_outlier_probabilities
        #             scores *= 100
        #         else:
        #             scores = []
        #             for currdate in Date:
        #                 daily_loop = compute_loop([startday_quiet, endday_quiet], Date, currdate, X, extent=extent, n_neighbors=n_neighbors)
        #                 scores.append(daily_loop*100)
        #             scores = np.array(scores)
            
            

        #         scores = score_fillnan(scores, Date, Date_scores)
                

        #         st.markdown('#### LoOPの計算結果のプロット')
        #         st.markdown('1段目:LoOP,  2段目以降:LoOPの計算に使用したデータ')
                
        #         make_plot(Date, scores, X, filetype, Nparams)


        #         if filetype=='time':
        #             df_outcsv = pd.DataFrame({'date': Date, 'LoOP': scores})
        #         else:
        #             df_outcsv = pd.DataFrame({'data_index': Date, 'LoOP': scores})
                
        #         st.write(df_outcsv)

                
        #         st.markdown('### LoOPの計算結果をCSVファイルでダウンロード')
                
        #         csv = convert_df(df_outcsv)

        #         st.download_button(
        #             label="Download calculated LoOPs as CSV",
        #             data=csv,
        #             file_name='LoOP_result.csv',
        #             mime='text/csv'
        #         )
                
        #         st.markdown('### サンプルコードをダウンロード')
        #         # Create an in-memory buffer to store the zip file
        #         with io.BytesIO() as buffer:
        #             # Write the zip file to the buffer
        #             with zipfile.ZipFile(buffer, "w") as zip:
        #                 zip.writestr("LoOP_result.csv", csv)
        #                 zip.writestr(uploaded_file.name, convert_df(df_in))
                        
        #                 Codes = ''
        #                 with open("loop_functions.py", encoding='utf8') as f:
        #                     for s_line in f:
        #                         Codes += s_line
                        
        #                 zip.writestr("loop_functions.py", Codes)
        #                 f.close()

        #                 Codes = ''
        #                 path = 'template.py'
        #                 csv_name = uploaded_file.name
        #                 filetype_bool = False
        #                 if filetype=='time':
        #                     filetype_bool = True

        #                 with open(path, encoding='utf8') as f:
        #                     for s_line in f:
                                
        #                         if 'csv_name =' in s_line:
        #                             Codes += '    csv_name = "'+csv_name+'"\n'
        #                         elif 'extent =' in s_line:
        #                             Codes += '    extent = '+str(int(extent))+'\n'
        #                         elif 'n_neighbors =' in s_line:
        #                             Codes += '    n_neighbors = '+str(n_neighbors)+'\n'
        #                         elif 'time_series =' in s_line:
        #                             Codes += '    time_series = '+str(filetype_bool)+'\n'
        #                         else:
        #                             Codes += s_line
                        
        #                 zip.writestr("sample_main.py", Codes)
        #                 f.close()

        #             buffer.seek(0)

        #             st.download_button(
        #                 label="Download sample code as ZIP",
        #                 data=buffer,  # Download buffer
        #                 file_name="sample.zip" 
        #             )

                

        #         ### 初期状態を示す状態変数をFalse
        #         st.session_state.initial_load = False

        #         st.session_state.date = Date
        #         st.session_state.score = scores

        # ### ２回目以降。
        # elif is_click and st.session_state.initial_load==False:
        #     ### ２回目以降。LoOPの新規計算あり。
        #     Date = st.session_state.date
        #     scores = st.session_state.score

        #     make_plot(Date, scores, X, filetype, Nparams)


        #     if filetype=='time':
        #         df_outcsv = pd.DataFrame({'date': Date, 'LoOP': scores})
        #     else:
        #         df_outcsv = pd.DataFrame({'data_index': Date, 'LoOP': scores})
            

        #     st.markdown('### LoOPの計算結果をCSVファイルでダウンロード')
            
        #     csv = convert_df(df_outcsv)

        #     st.download_button(
        #         label="Download calculated LoOPs as CSV",
        #         data=csv,
        #         file_name='LoOP_result.csv',
        #         mime='text/csv'
        #     )
            
        #     st.markdown('### サンプルコードをダウンロード')
        #     # Create an in-memory buffer to store the zip file
        #     with io.BytesIO() as buffer:
        #         # Write the zip file to the buffer
        #         with zipfile.ZipFile(buffer, "w") as zip:
        #             zip.writestr("LoOP_result.csv", csv)
        #             zip.writestr(uploaded_file.name, convert_df(df_in))
                    
        #             Codes = ''
        #             with open("loop_functions.py", encoding='utf8') as f:
        #                 for s_line in f:
        #                     Codes += s_line
                    
        #             zip.writestr("loop_functions.py", Codes)
        #             f.close()

        #             Codes = ''
        #             path = 'template.py'
        #             csv_name = uploaded_file.name
        #             filetype_bool = False
        #             if filetype=='time':
        #                 filetype_bool = True

        #             with open(path, encoding='utf8') as f:
        #                 for s_line in f:
                            
        #                     if 'csv_name =' in s_line:
        #                         Codes += '    csv_name = "'+csv_name+'"\n'
        #                     elif 'extent =' in s_line:
        #                         Codes += '    extent = '+str(int(extent))+'\n'
        #                     elif 'n_neighbors =' in s_line:
        #                         Codes += '    n_neighbors = '+str(n_neighbors)+'\n'
        #                     elif 'time_series =' in s_line:
        #                         Codes += '    time_series = '+str(filetype_bool)+'\n'
        #                     else:
        #                         Codes += s_line
                    
        #             zip.writestr("sample_main.py", Codes)
        #             f.close()

        #         buffer.seek(0)

        #         st.download_button(
        #             label="Download sample code as ZIP",
        #             data=buffer,  # Download buffer
        #             file_name="sample.zip" 
        #         )

        # elif st.session_state.initial_load==False:
        #     ### ２回目以降。LoOPの新規計算なし。    
        #     Date = st.session_state.date
        #     scores = st.session_state.score

        #     make_plot(Date, scores, X, filetype, Nparams)

            
            
        #     if filetype=='time':
        #         df_outcsv = pd.DataFrame({'date': Date, 'LoOP': scores})
        #     else:
        #         df_outcsv = pd.DataFrame({'data_index': Date, 'LoOP': scores})
            
            
        #     st.markdown('### LoOPの計算結果をCSVファイルでダウンロード')
            
            
        #     csv = convert_df(df_outcsv)

        #     st.download_button(
        #         label="Download calculated LoOPs as CSV",
        #         data=csv,
        #         file_name='LoOP_result.csv',
        #         mime='text/csv'
        #     )
            
        #     st.markdown('### サンプルコードをダウンロード')
        #     # Create an in-memory buffer to store the zip file
        #     with io.BytesIO() as buffer:
        #         # Write the zip file to the buffer
        #         with zipfile.ZipFile(buffer, "w") as zip:
        #             zip.writestr("LoOP_result.csv", csv)
        #             zip.writestr(uploaded_file.name, convert_df(df_in))
                    
        #             Codes = ''
        #             with open("loop_functions.py", encoding='utf8') as f:
        #                 for s_line in f:
        #                     Codes += s_line
                    
        #             zip.writestr("loop_functions.py", Codes)
        #             f.close()

        #             Codes = ''
        #             path = 'template.py'
        #             csv_name = uploaded_file.name
        #             filetype_bool = False
        #             if filetype=='time':
        #                 filetype_bool = True

        #             with open(path, encoding='utf8') as f:
        #                 for s_line in f:
                            
        #                     if 'csv_name =' in s_line:
        #                         Codes += '    csv_name = "'+csv_name+'"\n'
        #                     elif 'extent =' in s_line:
        #                         Codes += '    extent = '+str(int(extent))+'\n'
        #                     elif 'n_neighbors =' in s_line:
        #                         Codes += '    n_neighbors = '+str(n_neighbors)+'\n'
        #                     elif 'time_series =' in s_line:
        #                         Codes += '    time_series = '+str(filetype_bool)+'\n'
        #                     else:
        #                         Codes += s_line
                    
        #             zip.writestr("sample_main.py", Codes)
        #             f.close()

        #         buffer.seek(0)

        #         st.download_button(
        #             label="Download sample code as ZIP",
        #             data=buffer,  # Download buffer
        #             file_name="sample.zip" 
        #         )


    

if __name__ == '__main__':

    main()
    