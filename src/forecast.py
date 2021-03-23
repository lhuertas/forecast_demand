import os
import pandas as pd
import re
from datetime import datetime
import requests
from bs4 import BeautifulSoup

from transform import Transform
from model import Model

class Forecast:

    def __init__(self):
        self.__products_demand = None
        self.__transform = Transform()
        self.__model = Model()

    def run(self):
        print('Running forecast')
        self.read_data()
        transformed_data, df_future = self.transform_data()
        df_future = self.add_holidays(df_future)
        df_future = self.__model.convert_categorical(df_future)

        df = self.add_holidays(transformed_data)
        df = self.__model.convert_categorical(df)
        #df.to_csv("final_df.csv", index=False)
        self.__model.baseline_error(df)
        model = self.__model.split_train_test(df)
        models, parameters= self.__model.get_regressors()
        #search_results, best_model = self.__model.modeling()
        self.__model.linear_model()

        prediction = self.__model.make_prediction(df_future,'model.sav')

        print('Done')

    def read_data(self):
        print('Loading input data...')
        self.__products_demand = self.load_main_data()

    def load_main_data(self):

        dtypes = {
            "Product_Code": "str",
            "Warehous": "str",
            "Product_Category": "str",
            "Order_Demand": "str"
        }
        df = pd.read_csv("data/Historical Product Demand.csv", dtype=dtypes, parse_dates=["Date"])
        # some of the values are between parenthesis
        df['Order_Demand'] = df.Order_Demand.apply(lambda x: re.sub('[()]', '', x))
        df['Order_Demand'] = df['Order_Demand'].astype('int64')

        return df

    def read_holidays(self, ini_year, fin_year):
        country = "norway"
        url = "http://www.timeanddate.com/holidays/"
        holidays = pd.DataFrame(columns=["name","type","details","date"])
        years = range(ini_year, fin_year+1, 1)

        for year in years:
            url_link = f'{url}/{country}/{year}'
            response = requests.get(url_link)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                table = soup.findAll('table', {"class": "table"})[0]
                tr = table.findAll(['tr'])
                for cell in tr:
                    if "data-date" in str(cell):
                        tr_text = str(cell).split('"', 2)
                        tr_date = datetime.fromtimestamp(int(tr_text[1])/1000).strftime('%Y-%m-%d')
                        td = cell.find_all('td')
                        row = [i.text.replace('\n', '') for i in td][1:] + [tr_date]
                        row = {
                            "name": row[0],
                            "type": row[1],
                            "details": row[2],
                            "date": row[3]
                        }
                        holidays = holidays.append([row], ignore_index=True)
            else:
                print(f"No response from site, holidays for year {year}")

        holidays = holidays[holidays.type.str.contains('|'.join(["National holiday", "Bank Holiday"]))]
        return holidays

    def add_holidays(self, df):
        ini_year = min(df.date.dt.year)
        fin_year = max(df.date.dt.year)
        holidays = self.read_holidays(ini_year, fin_year)
        holidays['date'] = pd.to_datetime(holidays.date)
        holidays['weekofyear'] = holidays.date.dt.weekofyear
        holidays['year'] = holidays.date.dt.year
        holidays['n_holidays'] = holidays.groupby(['year','weekofyear'])['type'].transform('count')
        holidays['date'] = pd.to_datetime((holidays.year*100+holidays.weekofyear).astype(str) + '0', format='%Y%W%w')
        df = df.merge(holidays[['date','n_holidays']], how='left', on='date')
        df = df.fillna(0)

        return df


    def transform_data(self):
        df_transformed = self.__transform.transform_data(self.__products_demand)

        return df_transformed


if __name__ == '__main__':
    os.chdir('../')

    forecast = Forecast()
    forecast.run()