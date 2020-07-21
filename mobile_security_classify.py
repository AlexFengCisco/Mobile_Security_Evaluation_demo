from __future__ import absolute_import
from __future__ import division
from __future__ import print_function



import numpy as np
import tensorflow as tf



def main():


  # Specify that all features have real-value data
  feature_columns = [tf.contrib.layers.real_valued_column("", dimension=4)]

  # 3 layer DNN with 10, 20, 10 units respectively.
  classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                              hidden_units=[10, 20, 10],
                                              n_classes=4,
                                              model_dir="./Mobile_security_model")



  # Classify mobiles .  sample sets are 4 mobiles
  def mobile_to_predict():
    return np.array(
      [[6.4, 2.8, 2.8, 5.6, 5.6, 2.2],
       [4.4, 3.2, 3.2, 5.0, 5.0, 1.7],
       [4.8, 3.1, 3.1, 1.5, 1.5, 0.1],
       [6.4, 3.2, 3.2, 4.5,	4.5, 1.5]], dtype=np.float32)


  predictions = list(classifier.predict(input_fn=mobile_to_predict))

  print(
      "for Mobiles data input, Security Class Predictions:    {}\n"
      .format(predictions))
  print("sample case , 1 stands for 10% ,2 stands for 20% ... etc.")

if __name__ == "__main__":
    main()