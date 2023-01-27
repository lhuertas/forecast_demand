import pandas as pd
import numpy as np
import statistics
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error
import pickle

class Model:

    def __init__(self):
        self.__X_train = None
        self.__X_test = None
        self.__Y_train = None
        self.__Y_test = None
        self.__models = None
        self.__parameters = None
        self.__folds= []

    def convert_categorical(self,df):
        """
        Encode categorical variables

        :param df: Dataframe
        :return: Dataframe with encoding
        """
        df = pd.get_dummies(df, columns=['Product_Category'], drop_first=True)
        #df = pd.get_dummies(df, columns=['Warehouse'], drop_first=True)

        return df

    def split_train_test(self, df):
        """
        Train and test splitting
        :param df: Training dataframe
        """
        last_date = df.date.max()
        stop_date = last_date - pd.Timedelta(days=365)
        df['product'] = df.Product_Code.apply(lambda x: int(x.split("_")[1]))
        df.drop(['Product_Code'],axis=1, inplace=True)
        train, test = df[df.date<stop_date], df[df.date>=stop_date]
        train.Order_Demand = train.Order_Demand.astype(np.int32)
        test.Order_Demand = test.Order_Demand.astype(np.int32)

        self.__X_train = train.drop(['Order_Demand'],axis=1)
        self.__X_test = test.drop(['Order_Demand','date'],axis=1)
        self.__Y_train, self.__Y_test = train['Order_Demand'], test['Order_Demand']

        val_ini = stop_date - pd.Timedelta(days=365)
        train_idx = self.__X_train[self.__X_train.date<val_ini].index
        val_idx = self.__X_train[(self.__X_train.date>=val_ini)].index
        #groups = self.__X.groupby('product').groups
        #sorted_groups = [value for (key,value) in sorted(groups.items())]

        self.__folds.append([np.array(train_idx), np.array(val_idx)])
        self.__X_train.drop(['date'],axis=1, inplace=True)


    def get_regressors(self):
        """
        Definition of model and parameters for training
        :return: dict models and dict with parameters
        """
        models = {
            'naive': GaussianNB(),
            'svm': SVR(cache_size=700),
            'extratr': ExtraTreesRegressor(),
            'xgboost': XGBRegressor()
        }
        parameters = {
            'naive': dict(var_smoothing= np.logspace(0,-9, num=3)),
            'svm': dict(C=[0.01, 0.1, 1.0], epsilon=[0.01,0.1,1], kernel=['linear','rbf']),
            'extratr': dict(n_estimators=[10,100], max_depth=[3,10])
        }

        for model_name, params in parameters.items():
            parameters[model_name] = {model_name + '__' + parameter:values for parameter, values in params.items()}

        return models, parameters

    def baseline_error(self, df):
        """
        MSE calculated with respect to demand of previous week
        :param df: dataframe
        :return: Print baseline error
        """
        df['error'] = mean_squared_log_error(df['Order_Demand'], df['y_lag_pre_oneweek'])
        print(f'Baseline error {statistics.mean(df.error)}')
        df.drop(columns=['error'], axis=1, inplace=True)

    def modeling(self):
        """
        Model competition
        :return: Randomized search results (dict) and best model (dict)
        """
        self.__models, self.__parameters = self.get_regressors()
        scaler = ('scaler', StandardScaler())
        scorer ='neg_mean_squared_log_error'

        search_results = {}
        best_model = {}
        best_score =  9e20

        for model_name in ['naive','svm','extratr']:
            model = self.__models[model_name]
            parameters = self.__parameters[model_name]
            print('Training', model_name, 'model...')
            pipe = Pipeline([scaler, (model_name, model)])

            reg_search = RandomizedSearchCV(estimator=pipe, param_distributions=parameters, cv=2, n_iter=2,
                                            scoring=scorer, n_jobs=-1, verbose=True)
            reg_search.fit(self.__X_train, self.__Y_train)
            search_results[model_name] = reg_search
            print(reg_search.best_score_)
            if -reg_search.best_score_ >= best_score:
                best_model['model_name'] = model_name
                best_model['parameters'] = parameters
                best_score = -reg_search.best_score_

        print(f'Model competition won by {best_model["model_name"]} with score {best_score:.2f}.')

        return search_results, best_model

    def linear_model(self):
        pipe = make_pipeline(StandardScaler(), LinearRegression())

        model_fit = pipe.fit(self.__X_train, self.__Y_train)

        y_pred = model_fit.predict(self.__X_test)
        results = pd.DataFrame()
        results['product'] = self.__X_test['product']
        results['y_true'] = self.__Y_test.values
        results['y_pred'] = y_pred
        results['error'] = mean_squared_log_error(results['y_true'],results['y_pred'])

        x_train = pd.concat([self.__X_train, self.__X_test], axis=0)
        y_train = pd.concat([self.__Y_train, self.__Y_test], axis=0)
        model = pipe.fit(x_train,y_train)
        filename = 'model.sav'
        pickle.dump(model, open(filename, 'wb'))

        return results

    def make_prediction(self, df, model):
        """
        Make predictions with trained model
        :param df: Dataframe
        :param model: Trained model used for predictions
        :return: Dataframe
        """
        df['product'] = df.Product_Code.apply(lambda x: int(x.split("_")[1]))
        date = df.date
        df.drop(['Product_Code','date','Order_Demand'],axis=1, inplace=True)

        loaded_model = pickle.load(open(model, 'rb'))
        df['y_pred'] = loaded_model.predict(df)
        df['y_pred'] = df.y_pred.apply(lambda x: max(0,x))
        df['date'] = date

        return df

    def get_validation_metrics(self, search_results):
        """
        Evaluation metrics
        :param search_results: Dict with results of randomized search
        :return: dict with metrics
        """
        scores={}
        mean_scores={}
        mean_times={}
        best_params={}
        for model_name, reg_search in search_results.items():
            cv_scores=[]
            idx=reg_search.best_index_
            for i in range(self.__k_fold_val): #k_fold_val = 10
                sc=reg_search.cv_results_['split'+str(i)+'_test_score'][idx]
                cv_scores.append(sc)

            scores[model_name] = list(map(abs,cv_scores))
            mean_scores[model_name] = -reg_search.best_score_
            mean_times[model_name] = round(float(reg_search.cv_results_['mean_fit_time'][idx]),3)
            best_params[model_name] = reg_search.best_params_

        mean_scores_list = sorted(mean_scores.items(), key=operator.itemgetter(1))
        mean_times = sorted(mean_times.items(), key=lambda x: mean_scores[x[0]])
        mean_scores = mean_scores_list

        return scores, mean_scores, mean_times, best_params

