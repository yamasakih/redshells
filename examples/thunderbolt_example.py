from logging import getLogger

import gokart
import luigi
import numpy as np
import pandas as pd
import redshells
import xgboost
from luigi import FloatParameter, IntParameter
from luigi.util import inherits
from sklearn.datasets import load_iris

logger = getLogger(__file__)
luigi.configuration.LuigiConfigParser.add_config_path('./config/example.ini')
np.random.seed(57)


class SampleTask(gokart.TaskOnKart):
    task_namespace = 'sample'


class LoadIrisData(SampleTask):
    '''load iris.'''
    def output(self):
        return self.make_target('LoadIrisData.pkl')

    def run(self):
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['target'] = iris.target
        df['target_name'] = df.target.apply(lambda x: iris.target_names[x])
        self.dump(df)


class MakeMoldelInputData(SampleTask):
    def requires(self):
        return LoadIrisData()

    def output(self):
        return self.make_target('MakeMoldelInputData.pkl')

    def run(self):
        data = self.load()
        data = data.drop(['target_name'], axis=1)
        self.dump(data)


class OptimizeClassificationModel(
        redshells.train.train_clasification_model._ClassificationModelTask):
    task_namespace = 'sample'
    test_size = luigi.FloatParameter()  # type: float
    optuna_param_name = luigi.Parameter(
        description='The key of "redshells.factory.get_optuna_param".')
    output_file_path = luigi.Parameter(default='model/xgboost_model.pkl')

    def run(self):
        redshells.train.utils.optimize_model(self,
                                             param_name=self.optuna_param_name,
                                             test_size=self.test_size,
                                             binary=True)


class OptimizeXGBoostModel(SampleTask):
    n_estimators = IntParameter()
    test_size = FloatParameter()

    def requires(self):
        data = MakeMoldelInputData()
        redshells.factory.register_prediction_model('XGBClassifier',
                                                    xgboost.XGBClassifier)
        return OptimizeClassificationModel(
            rerun=True,
            train_data_task=data,
            target_column_name='target',
            model_name='XGBClassifier',
            model_kwargs=dict(n_estimators=self.n_estimators),
            test_size=self.test_size,
            optuna_param_name='XGBClassifier_default',
            output_file_path='model/OptimizeXGBoostModel.pkl')

    def output(self):
        return self.make_target('OptimizeXGBoostModel.pkl')

    def run(self):
        model = self.load()
        self.dump(model)


@inherits(OptimizeXGBoostModel)
class TrainXGBoostModel(SampleTask):
    def requires(self):
        data = MakeMoldelInputData()
        param = self.clone(OptimizeXGBoostModel).load()
        logger.info(f'param = {param}')
        return redshells.train.TrainClassificationModel(
            rerun=True,
            train_data_task=data,
            target_column_name='target',
            model_name='XGBClassifier',
            model_kwargs=param['best_params'],
            output_file_path='model/train_xgboost_model.pkl')

    def output(self):
        return self.make_target('TrainXGBoostModel.pkl')

    def run(self):
        model = self.load()
        self.dump(model)


if __name__ == '__main__':
    # gokart.run([
    #     'sample.OptimizeXGBoostModel', '--local-scheduler', '--no-lock',
    #     '--n-estimators=50', '--test-size=0.2'
    # ])
    gokart.run([
        'sample.TrainXGBoostModel', '--local-scheduler', '--no-lock',
        '--n-estimators=50', '--test-size=0.2'
    ])
