from logging import getLogger

import gokart
import luigi
from luigi.util import inherits
from luigi.util import requires
import pandas as pd
import sklearn.datasets
import xgboost

import redshells

logger = getLogger(__name__)


class MakeData(gokart.TaskOnKart):
    task_namespace = 'examples'

    def output(self):
        return self.make_target('binary_classification/data.pkl')

    def run(self):
        x, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        data = pd.DataFrame(dict(x=list(x), y=list(y)))
        logger.info(f'columns={data.columns}')
        logger.info(f'info=\n{data.info()}')
        self.dump(data)


class OptimizeModelExample(gokart.TaskOnKart):
    task_namespace = 'examples'

    def requires(self):
        data = MakeData()
        redshells.factory.register_prediction_model('XGBClassifier',
                                                    xgboost.XGBClassifier)
        return redshells.train.OptimizeBinaryClassificationModel(
            rerun=True,
            train_data_task=data,
            target_column_name='y',
            model_name='XGBClassifier',
            model_kwargs=dict(n_estimators=50),
            test_size=0.2,
            optuna_param_name='XGBClassifier_default')

    def output(self):
        return self.make_target('binary_classification/results.pkl')

    def run(self):
        model = self.load()
        logger.info(model)

        logger.info('dumping...')
        self.dump(model)


@requires(OptimizeModelExample)
class TrainModelExample(gokart.TaskOnKart):
    task_namespace = 'examples'

    def output(self):
        return self.make_target('binary_classification/train.pkl')

    def run(self):
        logger.info('loading best parameters...')
        param = self.load()

        logger.info('dumping...')
        self.dump(param)


if __name__ == '__main__':
    luigi.configuration.add_config_path('./config/example.ini')
    gokart.run()
