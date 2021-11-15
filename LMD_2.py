import pandas as pd
import numpy as np
import pickle
import os

#1
from scipy.sparse import coo_matrix
#2
from scipy.spatial import distance
#3
#3-2
from joblib import Parallel, delayed
#3-3
from scipy.stats.mstats import gmean
from scipy import optimize
#3-4
import itertools


def extraction_LMD_all(df,max_lat_grid,max_lon_grid,NS,folder_w,n_jobs_set):
    df=df.query('count>0').reset_index(drop=True)
    try:
        os.makedirs(folder_w)
    except:
        pass
    #Step1 Local maximum determination
    #Calculate the seed_grid in the NS grid range and the index of each grid contained in its neighborhood. 
    #Input: 
    #1. df: information of number of stops in each grid
    #2. NS: grid range of a neighborhood
    #输出:
    #1. df_1: Added whether_seed and seed_id fields. Empty field, assign value -100 
    #2. grid_belong_seed_id_matrix_list: Seed_id to which each grid belongs
    #3. grid_belong_seed_num_matrix: Record how many seeds compete for each grid

    def extraction_local_hotspots(df,NS,max_lat_grid,max_lon_grid):
        #df trnasforms to matrix
        data=df['count'].values
        row=df['lat_p_grid'].values
        col=df['lon_p_grid'].values
        count_matrix=coo_matrix((data,(row,col)),shape=(max_lat_grid+1,max_lon_grid+1)).toarray()
        #1 Extract seed_grid (Local Maximum) with its neighborhood
        seed_grid_index=[]
        grid_none_zero_index=np.where(count_matrix)
        for i in range(len(grid_none_zero_index[0])):
            lat_grid=grid_none_zero_index[0][i]
            lon_grid=grid_none_zero_index[1][i]
            grid_count=count_matrix[lat_grid,lon_grid]    
            #Calculate the index of the area range, and the boundary should be cut off
            min_row_grid=max(lat_grid-NS,0)
            max_row_grid=min(lat_grid+NS,max_lat_grid)
            min_col_grid=max(lon_grid-NS,0)
            max_col_grid=min(lon_grid+NS,max_lon_grid)
            grid_range_count_max=np.max(count_matrix[min_row_grid:max_row_grid+1,min_col_grid:max_col_grid+1])
            if grid_count>=grid_range_count_max: 
                seed_grid_index.append([lat_grid,lon_grid])
        seed_grid_index=np.array(seed_grid_index)
        df['whether_seed']=-100
        df['seed_id']=-100
        #Store the seed information in df to form df_1
        #seed_grid is stored as df_seed to facilitate subsequent operations
        df_seed=pd.DataFrame(columns=['lat_p_grid','lon_p_grid','count','seed_id'])
        df_seed['lat_p_grid']=seed_grid_index[:,0]
        df_seed['lon_p_grid']=seed_grid_index[:,1]
        df_seed['count']=count_matrix[seed_grid_index[:,0],seed_grid_index[:,1]]
        df_seed=df_seed.sort_values('count',ascending=False).reset_index(drop=True)
        df_seed['seed_id']=list(range(1,len(df_seed)+1))
        #1-2 Confirm each local hotspots (allowing overlapping)
        #Record the number of seed_id to which each grid belongs, and its index
        grid_belong_seed_num_matrix=np.zeros([max_lat_grid+1,max_lon_grid+1]).astype('int')
        grid_belong_seed_id_matrix_list=[['0' for k1 in range(max_lon_grid+1)] for k2 in range(max_lat_grid+1)]
        seed_inf=df_seed.values
        range_grid_base=np.array([[i_d,j_d] for i_d in np.arange(-NS,NS+1,1) for j_d in np.arange(-NS,NS+1)])

        for i in range(len(df_seed)):
            seed_lat_lon=seed_inf[i,[0,1]]
            seed_id=seed_inf[i,3]
            range_grid_seed=range_grid_base+seed_lat_lon
            eff_index=np.where((range_grid_seed[:,0]>=0) & (range_grid_seed[:,0]<=max_lat_grid) & (range_grid_seed[:,1]>=0) & (range_grid_seed[:,1]<=max_lon_grid))[0]
            range_grid_seed=range_grid_seed[eff_index]
            #Assign a value to the quantity
            grid_belong_seed_num_matrix[range_grid_seed[:,0],range_grid_seed[:,1]]+=1
            #Number the subordinate seeds
            for range_grid in range_grid_seed:
                grid_belong_seed_id_matrix_list[range_grid[0]][range_grid[1]]=grid_belong_seed_id_matrix_list[range_grid[0]][range_grid[1]]+'-'+str(seed_id)

        #Assign fields df_seed to df
        df['whether_seed']=0
        df['seed_id']=-100
        [lat_p_grid,lon_p_grid]=df_seed[['lat_p_grid','lon_p_grid']].values.T
        df=df.set_index(['lat_p_grid','lon_p_grid'])
        lat_lon_p_index=np.vstack((lat_p_grid,lon_p_grid)).T.tolist()
        df.loc[lat_lon_p_index,'whether_seed']=1
        df.loc[lat_lon_p_index,'seed_id']=df_seed['seed_id'].values
        df=df.reset_index(drop=False)
        return df,grid_belong_seed_id_matrix_list,grid_belong_seed_num_matrix

    #
    df_1,grid_belong_seed_id_matrix_list_1,grid_belong_seed_num_matrix_1=extraction_local_hotspots(df,NS,max_lat_grid,max_lon_grid)
    #Store
    df_1.to_pickle(folder_w+'1-df.pkl')
    pickle.dump(grid_belong_seed_id_matrix_list_1,open(folder_w+'1-grid_belong_seed_id_matrix_list.pickle','wb'))
    pickle.dump(grid_belong_seed_num_matrix_1,open(folder_w+'1-grid_belong_seed_num_matrix.pickle','wb'))

    del df
    del df_1
    del grid_belong_seed_id_matrix_list_1
    del grid_belong_seed_num_matrix_1

    #Step 2 Neighborhoods reshaping
    df_1=pd.read_pickle(folder_w+'1-df.pkl')
    grid_belong_seed_id_matrix_list_1=pickle.load(open(folder_w+'1-grid_belong_seed_id_matrix_list.pickle','rb'))
    grid_belong_seed_num_matrix_1=pickle.load(open(folder_w+'1-grid_belong_seed_num_matrix.pickle','rb'))

    #Input: the ones from steps 1
    #1. df_1
    #2. grid_belong_seed_id_matrix_list_1
    #3. grid_belong_seed_num_matrix_1
    #output: 
    #1. grid_classification_matrix_2 Seed_id to which each grid belongs 
    def reshape_result(df_1,grid_belong_seed_id_matrix_list_1,grid_belong_seed_num_matrix_1,max_lat_grid,max_lon_grid):
        #gravity 
        def gravity_compute(d_gravity,N_gravity):
            return N_gravity/d_gravity

        df_seed=df_1.query('whether_seed==1').sort_values('seed_id').reset_index(drop=True)
        grid_classification_index=np.array(np.where(grid_belong_seed_num_matrix_1!=0))#Grid index waiting to be allocated
        #Whether it is a matrix of seed_id. If it is a seed, keep the original seed_id. It can be judged directly with seed_id_matrix
        seed_id_matrix=np.zeros([max_lat_grid+1,max_lon_grid+1])
        seed_id_matrix[df_seed['lat_p_grid'].values,df_seed['lon_p_grid'].values]=df_seed['seed_id'].values

        #Enter the grid to be determined, and output cluster_id
        #Grid index to be classified 
        #Determine according to grid_belong_seed_id_matrix_list
        #Need to use seed_inf information

        seed_inf=df_seed.values
        belong_seed_id_list=[]
        l=len(grid_classification_index[0])
        for i in range(l):
            grid_classification=grid_classification_index[:,i]
            if seed_id_matrix[grid_classification[0],grid_classification[1]]!=0:
                belong_seed_id=seed_id_matrix[grid_classification[0],grid_classification[1]]
            else:
                seed_id_arr=np.array(grid_belong_seed_id_matrix_list_1[grid_classification[0]][grid_classification[1]].split('-')[1:]).astype('int')

                if len(seed_id_arr)==1:
                    belong_seed_id=seed_id_arr[0]
                else:
                    #To be determined
                    d_gravity_arr=distance.cdist([grid_classification],seed_inf[seed_id_arr-1,:2],'euclidean')
                    N_gravity_arr=seed_inf[seed_id_arr-1,2]
                    gravity_compute_arr=gravity_compute(d_gravity_arr,N_gravity_arr)[0]
                    #Determine whether there are multiple maximum values
                    max_gravity=np.max(gravity_compute_arr)
                    max_gravity_index=np.where(gravity_compute_arr==max_gravity)[0]
                    if len(max_gravity_index)==1:
                        belong_seed_id=seed_id_arr[max_gravity_index][0]
                    else:
                        max_gravity_seed_id_inf=seed_inf[seed_id_arr[max_gravity_index]-1]
                        belong_seed_id=max_gravity_seed_id_inf[np.argmax(max_gravity_seed_id_inf[:,2]),-1]
            belong_seed_id_list.append(belong_seed_id)
        belong_seed_id_list=np.array(belong_seed_id_list).astype('int')
        #Construct id matrix
        grid_classification_matrix=coo_matrix((belong_seed_id_list,(grid_classification_index[0],grid_classification_index[1])),shape=(max_lat_grid+1,max_lon_grid+1)).toarray()
        return grid_classification_matrix

    grid_classification_matrix_2=reshape_result(df_1,grid_belong_seed_id_matrix_list_1,grid_belong_seed_num_matrix_1,max_lat_grid,max_lon_grid)
    pickle.dump(grid_classification_matrix_2,open(folder_w+'2-grid_classification_matrix.pickle','wb'))

    del df_1
    del grid_belong_seed_id_matrix_list_1
    del grid_belong_seed_num_matrix_1

    #Step 3 Popular local hotspots determination
    #Contain multiple steps

    #3-1 Determine the shape of each seed_id and record the shape
    df_1=pd.read_pickle(folder_w+'1-df.pkl')
    grid_classification_matrix_2=pickle.load(open(folder_w+'2-grid_classification_matrix.pickle','rb'))
    #Input：
    #1. df_1
    #2. grid_classification_matrix_2
    def compute_shape(df_1,grid_classification_matrix_2):
        shape_record_arr=np.array([[0,0,0]]).astype('object') #shape_id,shape_size,shape_record
        df_seed=df_1.query('whether_seed==1').sort_values('seed_id').reset_index(drop=True)
        df_seed_values=df_seed.values
        index_list=df_1.query('whether_seed==1').sort_values('seed_id').index.tolist()#Index of the last record
        seed_id_array=df_seed['seed_id'].values
        seed_shape_id_record_list=[]
        #seed_id
        l=len(df_seed)
        for seed_id in seed_id_array:
            neighborhood_grids=np.array(np.where(grid_classification_matrix_2==seed_id)).T
            seed_grid=df_seed_values[seed_id-1,:2]
            shape_array=neighborhood_grids-seed_grid
            shape_flatten_list=shape_array[np.lexsort((shape_array[:,1],shape_array[:,0]))].flatten().astype('int').tolist()
            shape_record=str(shape_flatten_list).replace('[','').replace(']','')
            #df_seed adds a column field to indicate its shape_id
            shape_index=np.where(shape_record_arr[:,2]==shape_record)[0]
            if len(shape_index)!=0:
                seed_shape_id_record_list.append(int(shape_record_arr[shape_index,0][0]))
            else:
                shape_id=shape_record_arr.shape[0]
                shape_size=shape_array.shape[0]
                shape_record_arr=np.vstack((shape_record_arr,[shape_id,shape_size,shape_record]))
                seed_shape_id_record_list.append(shape_id)
        df_1['shape_id']=-100
        df_1.loc[index_list,'shape_id']=np.array(seed_shape_id_record_list).astype('int')
        shape_record_arr=shape_record_arr[1:]
        shape_record_arr[:,0]=shape_record_arr[:,0].astype('int')
        shape_record_arr[:,1]=shape_record_arr[:,1].astype('int')
        df_shape_record=pd.DataFrame(shape_record_arr,columns=['shape_id','shape_size','shape_record'])
        return df_shape_record,df_1
    df_shape_record_3_1,df_3_1=compute_shape(df_1,grid_classification_matrix_2)
    df_shape_record_3_1.to_pickle(folder_w+'3-1-df_shape_record.pkl')
    df_3_1.to_pickle(folder_w+'3-1-df.pkl')

    #3-2 Record the spatial scanning results of all shapes
    df_shape_record_3_1=pd.read_pickle(folder_w+'3-1-df_shape_record.pkl')
    df_3_1=pd.read_pickle(folder_w+'3-1-df.pkl')
    #Input df_values
    #1. df_3_1
    #2. df_shape_record_3_1
    def compute_shape_random_array(df_3_1,df_shape_record_3_1,max_lat_grid,max_lon_grid):
        def compute_random_array(shape_str,df_3_1,max_lat_grid,max_lon_grid):
            data=df_3_1['count'].values
            row=df_3_1['lat_p_grid'].values
            col=df_3_1['lon_p_grid'].values
            df_values=df_3_1.values
            count_matrix=coo_matrix((data,(row,col)),shape=(max_lat_grid+1,max_lon_grid+1)).toarray()
            def shape_transform(shape_str):
                tem=shape_str.split(',')
                result=np.array(tem).astype('int').reshape(int(len(tem)/2),2)
                return result
            shape_arr=shape_transform(shape_str)
            #In order to avoid selecting a grid out of study area, the number range of the grid should be limited at the beginning
            min_lat_restricted=-min(shape_arr[:,0])
            max_lat_restricted=max_lat_grid-max(shape_arr[:,0])
            min_lon_restricted=-min(shape_arr[:,1])
            max_lon_restricted=max_lon_grid-max(shape_arr[:,1])
            #According to the above range, filter the effective df_values indexes
            eff_index_list=np.where((df_values[:,0]>=min_lat_restricted) & (df_values[:,0]<=max_lat_restricted) & (df_values[:,1]>=min_lon_restricted) & (df_values[:,1]<=max_lon_restricted))[0]

            #Each has its own corresponding shape
            random_center=df_values[np.random.choice(eff_index_list,size=10000)][:,:2]
            random_base_list=[]
            for random_center_detail in random_center:
                shape_grid=(shape_arr+random_center_detail).astype('int')
                tem_sum=np.sum(count_matrix[shape_grid[:,0],shape_grid[:,1]])
                random_base_list.append(tem_sum)
            return random_base_list
        result=Parallel(n_jobs=n_jobs_set)(delayed(compute_random_array)(shape_str,df_3_1,max_lat_grid,max_lon_grid) for shape_str in df_shape_record_3_1['shape_record'].tolist())
        result=np.array(result).astype('int32').T
        return result
    shape_random_result_3_2=compute_shape_random_array(df_3_1,df_shape_record_3_1,max_lat_grid,max_lon_grid)
    del df_shape_record_3_1
    del df_3_1
    pickle.dump(shape_random_result_3_2,open(folder_w+'3-2-shape_random_result.pickle','wb'))
    del shape_random_result_3_2

    #3-3 Record Head/tail breaks for spatial scan results
    shape_random_result_3_2=pickle.load(open(folder_w+'3-2-shape_random_result.pickle','rb'))
    df_shape_record_3_1=pd.read_pickle(folder_w+'3-1-df_shape_record.pkl')

    #Input
    #1.df_shape_record_3_1
    #2.shape_random_result_3_2
    def compute_shape_threshold(df_shape_record_3_1,shape_random_result_3_2):
        def compute_partiion(random_arr):
            times=4
            partition_time_list=list(range(1,times+1))
            remain_num_list=[]
            pro_remain_list=[]
            min_value_list=[]
            Gmean_list=[]
            data_ori=random_arr.copy()
            data=data_ori.copy()
            for i in range(times+1):
                #数据本身
                remain_num_list.append(len(data))
                min_value_list.append(min(data))
                tem=gmean(data)
                Gmean_list.append(tem)
                data=data[data>tem]
            remain_num_list=remain_num_list[1:]
            judge_threshold_list=[0]+Gmean_list
            pro_remain_list=np.array(remain_num_list)/10000
            final_list=[partition_time_list,remain_num_list,pro_remain_list.tolist(),min_value_list[1:],Gmean_list[1:]]
            return np.array(final_list).T

        result_shape=np.array([[k]*4 for k in df_shape_record_3_1[['shape_id','shape_size']].values.tolist()]).flatten()
        result_shape=result_shape.reshape((int(len(result_shape)/2),2))
        result=Parallel(n_jobs=n_jobs_set)(delayed(compute_partiion)(shape_random_result_3_2[:,j]) for j in range(shape_random_result_3_2.shape[1]))
        result=np.array(result).flatten()
        result=result.reshape(int(len(result)/5),5)
        df_result_values=np.hstack((result_shape,result))
        #The final df
        df=pd.DataFrame(df_result_values,columns=['shape_id','shape_size','partition_time','remain_num','remain_pro','min_value','Gmean'])
        df['whether_threshold']=0
        index_list=df.query('remain_pro<0.4').drop_duplicates('shape_id').index
        df.loc[index_list,'whether_threshold']=1
        return df

    df_shape_threshold_3_3=compute_shape_threshold(df_shape_record_3_1,shape_random_result_3_2)
    df_shape_threshold_3_3.to_pickle(folder_w+'3-3-df_shape_threshold.pkl')

    #3-4 Get the final filtered result
    df_3_1=pd.read_pickle(folder_w+'3-1-df.pkl') # Data of original density
    df_shape_record_3_1=pd.read_pickle(folder_w+'3-1-df_shape_record.pkl')
    grid_classification_matrix_2=pickle.load(open(folder_w+'2-grid_classification_matrix.pickle','rb')) #Affiliation data of the grid
    df_shape_threshold_3_3=pd.read_pickle(folder_w+'3-3-df_shape_threshold.pkl') #Threshold of each shape

    def final_merge(df_3_1,df_shape_record_3_1,grid_classification_matrix_2,df_shape_threshold_3_3):
        data=df_3_1['count'].values
        row=df_3_1['lat_p_grid'].values
        col=df_3_1['lon_p_grid'].values
        count_matrix=coo_matrix((data,(row,col)),shape=(max_lat_grid+1,max_lon_grid+1)).toarray()
        #
        seed_id_arr=np.unique(grid_classification_matrix_2)[1:]
        #Output'lat_p_grid','lon_p_grid'，'seed_id'
        #Further processing of other fields
        lat_p_grid_final_list=[]
        lon_p_grid_final_list=[]
        seed_id_final_list=[]
        def compute_base(seed_id,grid_classification_matrix_2,count_matrix):
            print(seed_id)
            lat_p_grid,lon_p_grid=np.where(grid_classification_matrix_2==seed_id)
            total_count=np.sum(count_matrix[lat_p_grid,lon_p_grid])
            return lat_p_grid,lon_p_grid,[seed_id]*len(lat_p_grid),[total_count]*len(lat_p_grid)
        result=Parallel(n_jobs=n_jobs_set)(delayed(compute_base)(seed_id,grid_classification_matrix_2,count_matrix) for seed_id in seed_id_arr)
        columns_name=['lat_p_grid','lon_p_grid','count','total_count','seed_id','shape_id','shape_size',
                      'count_per_grid','whether_seed','threshold_count','whether_significant','sort_seed_id']
        df=pd.DataFrame(columns=columns_name)
        lat_p_grid_list=[k[0].tolist() for k in result]
        lon_p_grid_list=[k[1].tolist() for k in result]
        seed_id_list=[k[2] for k in result]
        total_count_list=[k[3] for k in result]
        lat_p_grid_list=list(itertools.chain.from_iterable(lat_p_grid_list))
        lon_p_grid_list=list(itertools.chain.from_iterable(lon_p_grid_list))
        seed_id_list=list(itertools.chain.from_iterable(seed_id_list))
        total_count_list=list(itertools.chain.from_iterable(total_count_list))
        #Record lat_p_grid,lon_p_grid,seed_id,count,whether_seed
        df['lat_p_grid']=lat_p_grid_list
        df['lon_p_grid']=lon_p_grid_list
        df['seed_id']=seed_id_list
        df['total_count']=total_count_list
        #whether_seed
        df=df.set_index(['lat_p_grid','lon_p_grid'])
        df_3_1=df_3_1.set_index(['lat_p_grid','lon_p_grid'])
        df['count']=df_3_1['count']
        df['whether_seed']=df_3_1['whether_seed']
        df=df.reset_index(drop=False)
        df['count']=df['count'].fillna(0)
        df['whether_seed']=df['whether_seed'].fillna(0)
        df_3_1=df_3_1.reset_index(drop=False)
        #Record shape_id&shape_size
        df=df.set_index('seed_id')
        df_3_1_eff=df_3_1.reset_index(drop=False).query('seed_id>0').set_index('seed_id')
        df['shape_id']=df_3_1_eff['shape_id']
        df=df.reset_index(drop=False).set_index('shape_id')
        df_shape_record_3_1=df_shape_record_3_1.set_index('shape_id')
        df['shape_size']=df_shape_record_3_1['shape_size']
        df['shape_size']=df['shape_size'].astype('int')
        #threshold_count
        df_threshold=df_shape_threshold_3_3.query('whether_threshold==1').set_index('shape_id')
        df['threshold_count']=df_threshold['min_value']
        df=df.reset_index(drop=False)
        #count_per_grid
        df['whether_significant']=0
        index_list=df.query('total_count>=threshold_count').index.tolist()
        df.loc[index_list,'whether_significant']=1
        df['count_per_grid']=df['total_count'].values/df['shape_size'].values
        #Seed_id sorted by number of stops
        df_sort_id=df.query('whether_significant==1 & whether_seed==1').sort_values('total_count',ascending=False).reset_index(drop=True)
        df_sort_id['sort_seed_id']=np.arange(1,len(df_sort_id)+1)
        df_sort_id=df_sort_id.set_index('seed_id')
        df['sort_seed_id']=-100
        df=df.set_index('seed_id')
        df['sort_seed_id']=df_sort_id['sort_seed_id']
        df=df.reset_index(drop=False)
        df['sort_seed_id']=df['sort_seed_id'].fillna(-100)
        return df
    df=final_merge(df_3_1,df_shape_record_3_1,grid_classification_matrix_2,df_shape_threshold_3_3)
    df.to_pickle(folder_w+'3-4-df_LMD_final.pkl')