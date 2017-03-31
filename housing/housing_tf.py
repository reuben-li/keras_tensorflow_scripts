from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import pandas as pd
import tensorflow as tf

COLUMNS = ["Id", "MSSubClass", "MSZoning", "LotFrontage", "LotArea", "Street",
  "Alley", "LotShape", "LandContour", "Utilities", "LotConfig", "LandSlope",
  "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle",
  "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", "RoofStyle",
  "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "MasVnrArea",
  "ExterQual", "ExterCond", "Foundation", "BsmtQual", "BsmtCond",
  "BsmtExposure", "BsmtFinType1", "BsmtFinSF1", "BsmtFinType2", "BsmtFinSF2",
  "BsmtUnfSF", "TotalBsmtSF", "Heating", "HeatingQC", "CentralAir",
  "Electrical", "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
  "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath", "BedroomAbvGr",
  "KitchenAbvGr", "KitchenQual", "TotRmsAbvGrd", "Functional", "Fireplaces",
  "FireplaceQu", "GarageType", "GarageYrBlt", "GarageFinish", "GarageCars",
  "GarageArea", "GarageQual", "GarageCond", "PavedDrive", "WoodDeckSF",
  "OpenPorchSF", "EnclosedPorch", "3SsnPorch", "ScreenPorch", "PoolArea",
  "PoolQC", "Fence", "MiscFeature", "MiscVal", "MoSold", "YrSold", "SaleType",
  "SaleCondition", "SalePrice"]

LABEL_COLUMN = "SalePrice"

CATEGORICAL_COLUMNS = ["MSSubClass", "MSZoning", "Street", "Alley", "LotShape",
  "LandContour", "Utilities", "LotConfig", "LandSlope", "Neighborhood",
  "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle",
  "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "ExterQual",
  "ExterCond", "Foundation", "BsmtQual", "BsmtCond", "BsmtExposure",
  "BsmtFinType1", "BsmtFinType2", "Heating", "HeatingQC", "CentralAir",
  "Electrical",  "KitchenQual", "Functional", "FireplaceQu", "GarageType",
  "GarageFinish", "GarageQual", "GarageCond", "PavedDrive", "PoolQC", "Fence",
  "MiscFeature", "MoSold", "YrSold", "SaleType", "SaleCondition" ]

CONTINUOUS_COLUMNS = ["Id", "LotFrontage", "LotArea", "OverallQual",
  "OverallCond", "YearBuilt", "YearRemodAdd", "MasVnrArea","BsmtFinSF1",
  "BsmtFinSF2","BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF",
  "LowQualFinSF", "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath",
  "HalfBath", "BedroomAbvGr", "KitchenAbvGr","TotRmsAbvGrd", "Fireplaces",
  "GarageYrBlt", "GarageCars", "GarageArea",  "WoodDeckSF", "OpenPorchSF",
  "EnclosedPorch", "3SsnPorch", "ScreenPorch", "MiscVal", "PoolArea"]

train_file_name = './train.csv'
test_file_name = './test.csv'

def build_estimator(model_dir, model_type):
  """Build an estimator."""
  # Sparse base columns.

  MSSubClass = tf.contrib.layers.sparse_column_with_hash_bucket(
    "MSSubClass", hash_bucket_size=1000, combiner="sqrtn")
  MSZoning = tf.contrib.layers.sparse_column_with_hash_bucket(
    "MSZoning", hash_bucket_size=1000, combiner="sqrtn")
  Street = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Street", hash_bucket_size=1000, combiner="sqrtn")
  Alley = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Alley", hash_bucket_size=1000, combiner="sqrtn")
  LotShape = tf.contrib.layers.sparse_column_with_hash_bucket(
    "LotShape", hash_bucket_size=1000, combiner="sqrtn")
  LandContour = tf.contrib.layers.sparse_column_with_hash_bucket(
    "LandContour", hash_bucket_size=1000, combiner="sqrtn")
  Utilities = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Utilities", hash_bucket_size=1000, combiner="sqrtn")
  LotConfig = tf.contrib.layers.sparse_column_with_hash_bucket(
    "LotConfig", hash_bucket_size=1000, combiner="sqrtn")
  LandSlope = tf.contrib.layers.sparse_column_with_hash_bucket(
    "LandSlope", hash_bucket_size=1000, combiner="sqrtn")
  Neighborhood = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Neighborhood", hash_bucket_size=1000, combiner="sqrtn")
  Condition1 = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Condition1", hash_bucket_size=1000, combiner="sqrtn")
  Condition2 = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Condition2", hash_bucket_size=1000, combiner="sqrtn")
  BldgType = tf.contrib.layers.sparse_column_with_hash_bucket(
    "BldgType", hash_bucket_size=1000, combiner="sqrtn")
  HouseStyle = tf.contrib.layers.sparse_column_with_hash_bucket(
    "HouseStyle", hash_bucket_size=1000, combiner="sqrtn")
  RoofStyle = tf.contrib.layers.sparse_column_with_hash_bucket(
    "RoofStyle", hash_bucket_size=1000, combiner="sqrtn")
  RoofMatl = tf.contrib.layers.sparse_column_with_hash_bucket(
    "RoofMatl", hash_bucket_size=1000, combiner="sqrtn")
  Exterior1st = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Exterior1st", hash_bucket_size=1000, combiner="sqrtn")
  Exterior2nd = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Exterior2nd", hash_bucket_size=1000, combiner="sqrtn")
  MasVnrType = tf.contrib.layers.sparse_column_with_hash_bucket(
    "MasVnrType", hash_bucket_size=1000, combiner="sqrtn")
  ExterQual = tf.contrib.layers.sparse_column_with_hash_bucket(
    "ExterQual", hash_bucket_size=1000, combiner="sqrtn")
  ExterCond = tf.contrib.layers.sparse_column_with_hash_bucket(
    "ExterCond", hash_bucket_size=1000, combiner="sqrtn")
  Foundation = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Foundation", hash_bucket_size=1000, combiner="sqrtn")
  BsmtQual = tf.contrib.layers.sparse_column_with_hash_bucket(
    "BsmtQual", hash_bucket_size=1000, combiner="sqrtn")
  BsmtCond = tf.contrib.layers.sparse_column_with_hash_bucket(
    "BsmtCond", hash_bucket_size=1000, combiner="sqrtn")
  BsmtExposure = tf.contrib.layers.sparse_column_with_hash_bucket(
    "BsmtExposure", hash_bucket_size=1000, combiner="sqrtn")
  BsmtFinType1 = tf.contrib.layers.sparse_column_with_hash_bucket(
    "BsmtFinType1", hash_bucket_size=1000, combiner="sqrtn")
  BsmtFinType2 = tf.contrib.layers.sparse_column_with_hash_bucket(
    "BsmtFinType2", hash_bucket_size=1000, combiner="sqrtn")
  Heating = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Heating", hash_bucket_size=1000, combiner="sqrtn")
  HeatingQC = tf.contrib.layers.sparse_column_with_hash_bucket(
    "HeatingQC", hash_bucket_size=1000, combiner="sqrtn")
  CentralAir = tf.contrib.layers.sparse_column_with_hash_bucket(
    "CentralAir", hash_bucket_size=1000, combiner="sqrtn")
  Electrical = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Electrical", hash_bucket_size=1000, combiner="sqrtn")
  KitchenQual = tf.contrib.layers.sparse_column_with_hash_bucket(
    "KitchenQual", hash_bucket_size=1000, combiner="sqrtn")
  Functional = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Functional", hash_bucket_size=1000, combiner="sqrtn")
  FireplaceQu = tf.contrib.layers.sparse_column_with_hash_bucket(
    "FireplaceQu", hash_bucket_size=1000, combiner="sqrtn")
  GarageType = tf.contrib.layers.sparse_column_with_hash_bucket(
    "GarageType", hash_bucket_size=1000, combiner="sqrtn")
  GarageFinish = tf.contrib.layers.sparse_column_with_hash_bucket(
    "GarageFinish", hash_bucket_size=1000, combiner="sqrtn")
  GarageQual = tf.contrib.layers.sparse_column_with_hash_bucket(
    "GarageQual", hash_bucket_size=1000, combiner="sqrtn")
  GarageCond = tf.contrib.layers.sparse_column_with_hash_bucket(
    "GarageCond", hash_bucket_size=1000, combiner="sqrtn")
  PavedDrive = tf.contrib.layers.sparse_column_with_hash_bucket(
    "PavedDrive", hash_bucket_size=1000, combiner="sqrtn")
  PoolQC = tf.contrib.layers.sparse_column_with_hash_bucket(
    "PoolQC", hash_bucket_size=1000, combiner="sqrtn")
  Fence = tf.contrib.layers.sparse_column_with_hash_bucket(
    "Fence", hash_bucket_size=1000, combiner="sqrtn")
  MiscFeature = tf.contrib.layers.sparse_column_with_hash_bucket(
    "MiscFeature", hash_bucket_size=1000, combiner="sqrtn")
  MiscVal = tf.contrib.layers.sparse_column_with_hash_bucket(
    "MiscVal", hash_bucket_size=1000, combiner="sqrtn")
  MoSold = tf.contrib.layers.sparse_column_with_hash_bucket(
    "MoSold", hash_bucket_size=1000, combiner="sqrtn")
  YrSold = tf.contrib.layers.sparse_column_with_hash_bucket(
    "YrSold", hash_bucket_size=1000, combiner="sqrtn")
  SaleType = tf.contrib.layers.sparse_column_with_hash_bucket(
    "SaleType", hash_bucket_size=1000, combiner="sqrtn")
  SaleCondition = tf.contrib.layers.sparse_column_with_hash_bucket(
    "SaleCondition", hash_bucket_size=1000, combiner="sqrtn")

  # Continuous base columns.
  Id = tf.contrib.layers.real_valued_column("Id")
  LotFrontage = tf.contrib.layers.real_valued_column("LotFrontage")
  LotArea = tf.contrib.layers.real_valued_column("LotArea")
  OverallQual = tf.contrib.layers.real_valued_column("OverallQual")
  OverallCond = tf.contrib.layers.real_valued_column("OverallCond")
  YearBuilt = tf.contrib.layers.real_valued_column("YearBuilt")
  YearRemodAdd = tf.contrib.layers.real_valued_column("YearRemodAdd")
  MasVnrArea = tf.contrib.layers.real_valued_column("MasVnrArea")
  BsmtFinSF1 = tf.contrib.layers.real_valued_column("BsmtFinSF1")
  BsmtFinSF2 = tf.contrib.layers.real_valued_column("BsmtFinSF2")
  BsmtUnfSF = tf.contrib.layers.real_valued_column("BsmtUnfSF")
  TotalBsmtSF = tf.contrib.layers.real_valued_column("TotalBsmtSF")
  x1stFlrSF = tf.contrib.layers.real_valued_column("1stFlrSF")
  x2ndFlrSF = tf.contrib.layers.real_valued_column("2ndFlrSF")
  LowQualFinSF = tf.contrib.layers.real_valued_column("LowQualFinSF")
  GrLivArea = tf.contrib.layers.real_valued_column("GrLivArea")
  BsmtFullBath = tf.contrib.layers.real_valued_column("BsmtFullBath")
  BsmtHalfBath = tf.contrib.layers.real_valued_column("BsmtHalfBath")
  FullBath = tf.contrib.layers.real_valued_column("FullBath")
  HalfBath = tf.contrib.layers.real_valued_column("HalfBath")
  BedroomAbvGr = tf.contrib.layers.real_valued_column("BedroomAbvGr")
  KitchenAbvGr = tf.contrib.layers.real_valued_column("KitchenAbvGr")
  TotRmsAbvGrd = tf.contrib.layers.real_valued_column("TotRmsAbvGrd")
  Fireplaces = tf.contrib.layers.real_valued_column("Fireplaces")
  GarageYrBlt = tf.contrib.layers.real_valued_column("GarageYrBlt")
  GarageCars = tf.contrib.layers.real_valued_column("GarageCars")
  GarageArea = tf.contrib.layers.real_valued_column("GarageArea")
  WoodDeckSF = tf.contrib.layers.real_valued_column("WoodDeckSF")
  OpenPorchSF = tf.contrib.layers.real_valued_column("OpenPorchSF")
  EnclosedPorch = tf.contrib.layers.real_valued_column("EnclosedPorch")
  x3SsnPorch = tf.contrib.layers.real_valued_column("3SsnPorch")
  ScreenPorch = tf.contrib.layers.real_valued_column("ScreenPorch")
  PoolArea = tf.contrib.layers.real_valued_column("PoolArea")

  # Transformations.
  #age_buckets = tf.contrib.layers.bucketized_column(age,
  #                                                  boundaries=[
  #                                                      18, 25, 30, 35, 40, 45,
  #                                                      50, 55, 60, 65
  #                                                  ])

  # Wide columns and deep columns.
  wide_columns = [ MSSubClass, MSZoning, Street, Alley, LotShape, LandContour,
    Utilities, LotConfig, LandSlope, Neighborhood, Condition1, Condition2,
    BldgType, HouseStyle, RoofStyle, RoofMatl, Exterior1st, Exterior2nd,
    MasVnrType, ExterQual, ExterCond, Foundation, BsmtQual, BsmtCond,
    BsmtExposure, BsmtFinType1, BsmtFinType2, Heating, HeatingQC, CentralAir,
    Electrical, KitchenQual, Functional, FireplaceQu, GarageType, GarageFinish,
    GarageQual, GarageCond, PavedDrive, PoolQC, Fence, MiscFeature, MiscVal,
    MoSold, YrSold, SaleType, SaleCondition ]
    #  tf.contrib.layers.crossed_column([education, occupation],
    #                                               hash_bucket_size=int(1e4)),
    #              tf.contrib.layers.crossed_column(
    #                  [age_buckets, education, occupation],
    #                  hash_bucket_size=int(1e6)),
    #              tf.contrib.layers.crossed_column([native_country, occupation],
    #                                               hash_bucket_size=int(1e4))]
  deep_columns = [ Id, LotFrontage, LotArea, OverallQual, OverallCond,
    YearBuilt, YearRemodAdd, MasVnrArea, BsmtFinSF1, BsmtFinSF2, BsmtUnfSF,
    TotalBsmtSF, x1stFlrSF, x2ndFlrSF, LowQualFinSF, GrLivArea, BsmtFullBath,
    BsmtHalfBath, FullBath, HalfBath, BedroomAbvGr, KitchenAbvGr, TotRmsAbvGrd,
    Fireplaces, GarageYrBlt, GarageCars, GarageArea, WoodDeckSF, OpenPorchSF,
    EnclosedPorch, x3SsnPorch, ScreenPorch, PoolArea ]
#      tf.contrib.layers.embedding_column(workclass, dimension=8),
#      tf.contrib.layers.embedding_column(education, dimension=8),
#      tf.contrib.layers.embedding_column(gender, dimension=8),
#      tf.contrib.layers.embedding_column(relationship, dimension=8),
#      tf.contrib.layers.embedding_column(native_country,
#                                         dimension=8),
#      tf.contrib.layers.embedding_column(occupation, dimension=8),
#      age,
#      education_num,
#      capital_gain,
#      capital_loss,
#      hours_per_week,
#  ]

  if model_type == "wide":
    m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                          feature_columns=wide_columns)
  elif model_type == "deep":
    m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                       feature_columns=deep_columns,
                                       hidden_units=[100, 50])
  else:
    m = tf.contrib.learn.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=[100, 50])
  return m


def input_fn(df):
  """Input builder function."""
  # Creates a dictionary mapping from each continuous feature column name (k) to
  # the values of that column stored in a constant Tensor.
  continuous_cols = {
      k: tf.constant(
          df[k].values,
          shape=[df[k].size,1])
      for k in CONTINUOUS_COLUMNS}
  # Creates a dictionary mapping from each categorical feature column name (k)
  # to the values of that column stored in a tf.SparseTensor.
  categorical_cols = {
      k: tf.SparseTensor(
          indices=[[i, 0] for i in range(df[k].size)],
          values=df[k].values,
          dense_shape=[df[k].size, 1])
      for k in CATEGORICAL_COLUMNS}
  # Merges the two dictionaries into one.
  feature_cols = dict(continuous_cols)
  feature_cols.update(categorical_cols)
  # Converts the label column into a constant Tensor.
  label = tf.constant(df[LABEL_COLUMN].values)
  # Returns the feature columns and the label.
  return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
  df_train = pd.read_csv(
      tf.gfile.Open(train_file_name),
      names=COLUMNS,
      skiprows=1,
      skipinitialspace=True,
      engine="python")
  df_test = pd.read_csv(
      tf.gfile.Open(test_file_name),
      names=COLUMNS,
      skiprows=1,
      skipinitialspace=True,
      engine="python")

  # remove NaN elements
  #df_train = df_train.dropna(how='all', axis=1)
  #df_test = df_test.dropna(how='all', axis=1)
  for i in CONTINUOUS_COLUMNS:
    df_train[i] = df_train[i].fillna(0)
    df_test[i] = df_test[i].fillna(0)
  for j in CATEGORICAL_COLUMNS:
    df_train[j] = df_train[j].fillna('')
    df_test[j] = df_test[j].fillna('')

#  df_train[LABEL_COLUMN] = (
#      df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
#  df_test[LABEL_COLUMN] = (
#      df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

  model_dir = tempfile.mkdtemp() if not model_dir else model_dir
  print("model directory = %s" % model_dir)

  m = build_estimator(model_dir, model_type)
  m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
  results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
  for key in sorted(results):
    print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
  train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                 FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.register("type", "bool", lambda v: v.lower() == "true")
  parser.add_argument(
      "--model_dir",
      type=str,
      default="./model",
      help="Base directory for output models."
  )
  parser.add_argument(
      "--model_type",
      type=str,
      default="wide_n_deep",
      help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
  )
  parser.add_argument(
      "--train_steps",
      type=int,
      default=20,
      help="Number of training steps."
  )
  parser.add_argument(
      "--train_data",
      type=str,
      default="",
      help="Path to the training data."
  )
  parser.add_argument(
      "--test_data",
      type=str,
      default="",
      help="Path to the test data."
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
