import pandas as pd
import numpy as np
import pickle
from scipy.sparse import coo_matrix
from joblib import Parallel, delayed
import os
def Neighborhood_size_determination(df,max_lat_grid,max_lon_grid,folder_w,n_jobs_set):
    try:
        os.makedirs(folder_w)
    except:
        pass
    #1-1 search seed grid in a specific grid range
    def seed_grid_extraction(df,max_lat_grid,max_lon_grid):
        #Data Prepration
        data=df['count'].values
        row=df['lat_p_grid'].values
        col=df['lon_p_grid'].values
        count_matrix=coo_matrix((data,(row,col)),shape=(max_lat_grid+1,max_lon_grid+1)).toarray()
        range_grid_list=list(range(1,15))
        def df_seed_paralle(range_grid):
            print(range_grid)
            seed_row_col_index=[]
            for row_grid in range(max_lat_grid+1):
                for col_grid in range(max_lon_grid+1):
                    grid_count=count_matrix[row_grid,col_grid]
                    if grid_count>0:
                        #Calculate 1 grid range index
                        min_row_grid=max(row_grid-range_grid,0)
                        max_row_grid=min(row_grid+range_grid,max_lat_grid)
                        min_col_grid=max(col_grid-range_grid,0)
                        max_col_grid=min(col_grid+range_grid,max_lon_grid)
                        grid_range_count_max=np.max(count_matrix[min_row_grid:max_row_grid+1,min_col_grid:max_col_grid+1])
                        if grid_count>=grid_range_count_max:
                            seed_row_col_index.append([row_grid,col_grid])
            seed_row_col_index_arr=np.array(seed_row_col_index)
            df_seed=pd.DataFrame(seed_row_col_index_arr,columns=['lat_p_grid','lon_p_grid'])
            df_seed['count']=0
            df_seed=df_seed.set_index(['lat_p_grid','lon_p_grid'])
            df2=df.set_index(['lat_p_grid','lon_p_grid'])
            df_seed['count']=df2['count']
            df_seed=df_seed.reset_index(drop=False)
            lat_p_grid,lon_p_grid=df_seed[['lat_p_grid','lon_p_grid']].values.T
            return df_seed
        df_seed_list=Parallel(n_jobs=n_jobs_set)(delayed(df_seed_paralle)(range_grid) for range_grid in range_grid_list)
        return df_seed_list
    df_seed_list=seed_grid_extraction(df,max_lat_grid,max_lon_grid)
    pickle.dump(df_seed_list,open(folder_w+'1-1-df_seed_list.pickle','wb'))
    #1-2 compute cover 
    def parallel_CR(k,df,df_seed_list,max_lat_grid,max_lon_grid):
        data=df['count'].values
        row=df['lat_p_grid'].values
        col=df['lon_p_grid'].values
        count_matrix=coo_matrix((data,(row,col)),shape=(max_lat_grid+1,max_lon_grid+1)).toarray()
        print(k)
        d=k+1
        df_seed=df_seed_list[k]
        range_grid_base=np.array([[i_d,j_d] for i_d in np.arange(-d,d+1,1) for j_d in np.arange(-d,d+1)])
        range_grid_all=[]
        for i in range(len(range_grid_base)):
            range_grid_all.append((df_seed[['lat_p_grid','lon_p_grid']].values+range_grid_base[i]).tolist())
        a=np.array(range_grid_all).flatten()
        range_grid_all=a.reshape(int(len(a)/2),2)
        range_grid_all=np.unique(range_grid_all,axis=0)
        eff_index=np.where((range_grid_all[:,0]>=0) & (range_grid_all[:,0]<=max_lat_grid) & (range_grid_all[:,1]>=0) & (range_grid_all[:,1]<=max_lon_grid))[0]
        range_grid_all=range_grid_all[eff_index]
        result=sum(count_matrix[range_grid_all[:,0],range_grid_all[:,1]])
        return result
    CR_list=Parallel(n_jobs=n_jobs_set)(delayed(parallel_CR)(k,df,df_seed_list,max_lat_grid,max_lon_grid) for k in range(len(df_seed_list)))
    df_CR=pd.DataFrame(columns=['grid_range','CR'])
    df_CR['CR']=CR_list
    df_CR['grid_range']=np.arange(1,len(df_CR)+1)
    df_CR['CR_dif']=[0]+np.diff(CR_list).tolist()
    df_CR['CR_dif2']=[0,0]+np.diff(np.diff(CR_list)).tolist()
    #Calculate 
    df_CR_allow=df_CR.query('CR_dif<0')
    if len(df_CR_allow)!=0:
        r_allow=df_CR_allow.iloc[0]['grid_range']-1
    if r_allow <3:
        result_grid_range=r_allow
    else:
        CR_dif2_list=df_CR['CR_dif2'].tolist()[2:]
        for i in range(len(CR_dif2_list)):
            CR_potential=CR_dif2_list[i]
            if i==0:
                CR_compare1=CR_potential-1
                CR_compare2=CR_dif2_list[i+1]
            if i==len(CR_dif2_list)-1:
                CR_compare1=CR_dif2_list[i-1]
                CR_compare2=CR_potential-1
            if (i!=0) & (i!=len(CR_dif2_list)-1):
                CR_compare1=CR_dif2_list[i-1]
                CR_compare2=CR_dif2_list[i+1]
            if (CR_potential<CR_compare1) & (CR_potential<CR_compare2):
                result_grid_range=i+3
                break
    pickle.dump(df_CR,open(folder_w+'1-2-df_CR.pickle','wb'))
    pickle.dump(result_grid_range,open(folder_w+'1-2-result_grid_range.pickle','wb'))