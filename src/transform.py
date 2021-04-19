import pandas as pd
import numpy as np
import math

class Transform:

    def __init__(self):
        self.__weights_disaggregation = None
        self.__statistics = None
        self.__future_dates = pd.DataFrame()

    def transform_data(self, df):
        transformed_df = self.join_duplicated_rows(df)
        transformed_df = self.derivate_dates(transformed_df, 'Date')
        self.get_statistics(transformed_df)
        df_agg = self.aggregate_data(transformed_df, ['Product_Code', 'Product_Category','year','weekofyear'], 'Order_Demand')
        #self.__weights_disaggregation = self.get_weights_warehouse_disaggregation(
        #    transformed_df, ['Product_Code','weekofyear','Warehouse'], 'Order_Demand')

        self.create_future_dates(df_agg)
        df = self.add_lags_variables(df_agg)
        df_future = self.add_lags_variables(self.__future_dates)
        df = self.add_cyclic_features(df)
        df_future = self.add_cyclic_features(df_future)
        df = df.fillna(0)
        df_future = df_future.fillna(0)
        df = self.add_moving_average(df)
        df_future = self.add_moving_average(df_future)

        return df, df_future

    def join_duplicated_rows(self, df):
        col_names = list(df.columns)
        col_names.remove('Order_Demand')

        df_no_duplicates = df.groupby(col_names)['Order_Demand'].sum().reset_index()

        return df_no_duplicates

    def derivate_dates(self, df, col):
        df['year'] = df[col].dt.year.astype(int)
        df['month'] = df[col].dt.month.astype(int)
        df['weekday'] = df[col].dt.weekday.astype(int)
        df['weekofyear'] = df[col].apply(lambda x: int(x.strftime("%V")) if pd.notnull(x) else None)

        return df

    def get_weights_warehouse_disaggregation(self, df, cols_group, col_agg):
        min_date = '2016-01-01'
        max_date = '2017-01-01'

        df = df[(df.Date>=min_date) & (df.Date<max_date)]
        sum_demands = df.groupby(cols_group).agg({col_agg: 'sum'})
        df_weights = sum_demands/sum_demands.groupby(level=[0,1]).transform('sum')
        df_weights = df_weights.reset_index()
        df_weights = df_weights.rename(columns={col_agg: 'weights'})

        return df_weights

    def aggregate_data(self, df, cluster, y):
        def reindex_by_date(df, max_date):
            dates = pd.date_range(df.index.min(), max_date, freq='W')
            return df.reindex(dates,fill_value=0)

        df = df[df.year>=2012]

        df = df.groupby(cluster)[y].sum().reset_index()
        df['date'] = pd.to_datetime((df.year*100+df.weekofyear).astype(str) + '0', format='%Y%W%w')
        df.set_index(pd.DatetimeIndex(df.date), inplace=True)

        max_date = df.index.max()

        df.drop(columns=['year','weekofyear','date'], axis=1, inplace=True)
        df2=df.groupby(['Product_Code','Product_Category']).resample('W').asfreq(fill_value=0)#.reset_index()
        df2.drop(columns=['Product_Code','Product_Category'], inplace=True)
        df2.reset_index(level=[0,1], inplace=True, col_level=1)
        df2=df2.groupby(['Product_Code','Product_Category']).apply(lambda x: reindex_by_date(x,max_date))
        df2.drop(columns=['Product_Code','Product_Category'], inplace=True)
        df2.reset_index(level=[0,1], inplace=True, col_level=1)

        df2['n_observations'] = df2.groupby('Product_Code')['Order_Demand'].transform('count')
        series_range = 52 + 52 + 52 # at least three years of history
        df2 = df2[df2['n_observations']>series_range]
        df2.drop(columns=['n_observations'], axis=1, inplace=True)

        return df2

    def create_future_dates(self,df):
        min_date = df.index.max()
        max_date = min_date + pd.Timedelta(days=30)
        self.__future_dates = df.groupby(['Product_Code','Product_Category']).apply(lambda x: x.reindex(pd.date_range(min_date,max_date, freq="W")))
        self.__future_dates.drop(columns=['Product_Code','Product_Category'], inplace=True, axis=1)
        self.__future_dates.reset_index(level=[0,1],inplace=True, col_level=1)

    def add_cyclic_features(self, df):
        cyclic_feat = pd.DataFrame()

        df['year'] = df.date.dt.year.apply(lambda x: int(str(x)[-1]))
        df['weekofyear'] = df.date.dt.weekofyear
        #self.__X
        df['sin_year'] = np.sin(2*math.pi*df['year']/df['year'].max())
        df['cos_year'] = np.cos(2*math.pi*df['year']/df['year'].max())
        #df['sin_month'] = np.sin(2*math.pi*df['month']/df['month'].max())
        #df['cos_month'] = np.cos(2*math.pi*df['month']/df['month'].max())
        df['sin_weekofyear'] = np.sin(2*math.pi*df['weekofyear']/df['weekofyear'].max())
        df['cos_weekofyear'] = np.cos(2*math.pi*df['weekofyear']/df['weekofyear'].max())

        df.drop(columns=['weekofyear','year'], inplace=True, axis=1)

        return df

    def calculate_statistics(self, df, cols_group, y):
        statistics = df.groupby(cols_group)[y].agg(['mean','median','std','max','min'])
        statistics.reset_index(inplace=True)
        statistics['min2'] = statistics['mean'] - 2*statistics['std']
        statistics['max2'] = statistics['mean'] + 2*statistics['std']
        statistics['min'] = statistics[['min','min2']].max(axis=1)
        statistics['max'] = statistics[['max','max2']].min(axis=1)
        statistics.drop(columns=['min2','max2','std','mean'], inplace=True)

        return statistics

    def get_statistics(self, df, by_group=['weekofyear']):

        for group in by_group:
            cols = ['Product_Code','year'] + [group]
            self.__statistics = self.calculate_statistics(df, cols, 'Order_Demand') #add statistics month, year?
            self.__statistics.rename(columns={'median':'median_'+group, 'min': 'min_'+group, 'max': 'max_'+group}, inplace=True)

    def add_moving_average(self, df):
        df_avg = pd.DataFrame()
        categories = df['Product_Category'].unique()

        for category in categories:
            df_temp = df.loc[df.Product_Category==category,['Product_Category','date','Order_Demand']]
            df_temp[f'y_avg_sma'] = df_temp['Order_Demand'].rolling(window=7, center=True).mean()
            #df_temp[f'y_avg_sma'] = df_temp['Order_Demand'].ewm(span=7).mean()
            df_temp.drop(columns=['Order_Demand'], inplace=True)
            df_avg = df_avg.append(df_temp, ignore_index=True)

        df_avg = self.derivate_dates(df_avg, "date")
        df_avg = df_avg.groupby(['Product_Category','date','weekofyear'])[f'y_avg_sma'].mean().reset_index()
        df_avg.drop(columns=['weekofyear'], axis=1, inplace=True)

        df = df.merge(df_avg, on=['Product_Category','date'], how='left')

        return df

    def add_lags_variables(self, df):
        #previous year same week, previous week, next week holiday

        df['y_lag_pre_year'] = df.groupby(["Product_Code","Product_Category"])['Order_Demand'].shift(52)
        df['y_lag_pre_oneweek'] = df.groupby(["Product_Code","Product_Category"])['Order_Demand'].shift(1)
        df['y_lag_pre_twoweek'] = df.groupby(["Product_Code","Product_Category"])['Order_Demand'].shift(2)

        df['year'] = df.index.year
        df['weekofyear'] = df.index.weekofyear
        df = df.merge(self.__statistics, how='left', on=['Product_Code','year','weekofyear'])
        df['lag_median_pre_weekofyear'] = df.groupby(["Product_Code","Product_Category"])['median_weekofyear'].shift(52)
        df['lag_min_pre_weekofyear'] = df.groupby(["Product_Code","Product_Category"])['min_weekofyear'].shift(52)
        df['lag_max_pre_weekofyear'] = df.groupby(["Product_Code","Product_Category"])['max_weekofyear'].shift(52)
        df['date'] = pd.to_datetime((df.year*100+df.weekofyear).astype(str) + '0', format='%Y%W%w')
        #df.set_index(pd.DatetimeIndex(df.date), inplace=True)
        df.drop(columns=['median_weekofyear','min_weekofyear','max_weekofyear','year','weekofyear'], axis=1, inplace=True)

        return df

    def get_cross_validation_indices(self):
        folds=[]
        val_end = pd.to_datetime(self.__X_train['date'].iloc[-1])
        print('Validation indices..')
        for k in range(self.__k_folds_val):
            val_ini = (val_end - pd.Timedelta(days=self.__validation_size-1))#f0r daily
            train_end = (val_ini - pd.Timedelta(days=self.__lag_size+1))

            train_idx = self.__X_train[self.__X_train['date']<=train_end].index
            val_idx = self.__X_train[(self.__X_train['date']>=val_ini)&(self.__X_train['date']<=val_end)].index
            folds.append((np.array(train_idx), np.append(val_idx)))

            print('Fold',k,val_ini,'-',val_end)
            val_end = (val_end- pd.Timedelta(days=self.__backtest_offset))

        return folds





