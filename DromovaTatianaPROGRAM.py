import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from typing import Optional
import DromovaTatianaModels


class Visualisation:
    """
    it's much easily to understand where to move if we can see the situation
    """
    def __init__(self, data: pd.DataFrame):
        self.data = data.astype(float)

    def correlate(self) -> pd.DataFrame:
        """
        just show dependence between features
        """
        sb.heatmap(self.data.corr(), annot=True)
        plt.show()
        return self.data.corr()


class Analysis:
    """
    data preparation and data mining
    """
    def __init__(self, path_tab: str, path_hash: str, path_train: str, path_test: str):
        self.x_test: Optional[pd.DataFrame] = None
        self.x_train: Optional[pd.DataFrame] = None
        self.data = pd.read_csv(path_tab)
        self.y_test = pd.read_csv(path_test)
        self.y_train = pd.read_csv(path_train)
        self.hashed = pd.read_csv(path_hash)

    def convert2des(self):
        """
        convert hexadecimal values (25th and 50th hashed features) to decimal values
        and convert dtype: object to float
        """
        self.data['feature_25'] = self.data['feature_25'].apply(lambda x: int(x, 16))
        self.hashed['feature_50'] = self.hashed['feature_50'].fillna('0')
        self.hashed['feature_50'] = self.hashed['feature_50'].apply(lambda x: int(x, 16))
        self.hashed['feature_50'] = self.hashed['feature_50'].replace(0, None)
        self.data['feature_25'] = self.data['feature_25'].astype('float64')
        self.hashed['feature_50'] = self.hashed['feature_50'].astype('float64')

    def normalize(self):
        '''
        scale and normalise all features by formula (data - min) / (max - min)
        now values should be in range [0, 1]
        '''
        self.data.loc[:, 'feature_0':] = (self.data.loc[:, 'feature_0':] - self.data.loc[:, 'feature_0':].min()) / \
                                         (self.data.loc[:, 'feature_0':].max() - self.data.loc[:, 'feature_0':].min())
        self.hashed['feature_50'] = (self.hashed['feature_50'] - self.hashed['feature_50'].min()) / \
                                    (self.hashed['feature_50'].max() - self.hashed['feature_50'].min())

    def group_by_id(self):
        """
        take average of all features by id
        """
        self.data = self.data.groupby(['id']).mean()
        self.data = self.data.drop('period', axis=1)
        self.hashed = self.hashed.groupby(['id']).mean()

    def drop(self):
        """
        knowing dependence, drop dependent variables (corr more than 70%)
        """
        list_of_high_corr = [1, 3, 4, 8, 10, 14, 18, 23, 27, 30, 32, 33, 34, 36, 41, 44, 45, 47, 48]
        self.data = self.data.drop(['feature_'+str(x) for x in list_of_high_corr], axis=1)

    def merge(self):
        """
        add 50th feature from file 'hashed_data.csv'
        """
        self.data = pd.merge(self.data, self.hashed, on='id')

    def clear(self) -> pd.DataFrame:
        """
        clear NA, exchanging it on mean of 2 neighbours values; if NA in first row -
        exchange on mean of feature values by id
        """
        self.data[:1] = self.data[:1].fillna(self.data.groupby('id').transform('mean'))
        self.data = self.data.where(self.data.values == 0,
                                    other=(self.data.fillna(method='ffill') + self.data.fillna(method='bfill')) / 2)
        return self.data

    def split(self):
        """
        separate data to train and test
        """
        self.x_train = self.data[:len(self.y_train)]
        self.x_test = self.data[len(self.y_train):]

    def prepare(self):
        self.convert2des()
        self.normalize()
        self.group_by_id()
        self.drop()
        self.merge()
        self.clear()
        self.split()


if __name__ == "__main__":
    analysis = Analysis('tabular_data.csv', 'hashed_feature.csv', 'train.csv', 'test.csv')
    analysis.prepare()

    model = DromovaTatianaModels.Ensamble(analysis.x_train, analysis.y_train, analysis.x_test)
    model.fit()  # 62.90%
    prediction = model.predict()[:, 0]

    result = pd.DataFrame({'id': analysis.y_test['id'], 'score': prediction})
    result.to_csv('DromovaTatiana_test.csv', columns=['id', 'score'], index=False)
