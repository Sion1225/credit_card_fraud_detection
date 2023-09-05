import tensorflow as tf

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')
        self.false_positives = self.add_weight(name='fp', initializer='zeros')
        self.true_negatives = self.add_weight(name='tn', initializer='zeros')
        self.false_negatives = self.add_weight(name='fn', initializer='zeros')
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        threshold = 0.5
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.cast(y_pred >= threshold, tf.float32)
        
        self.true_positives.assign_add(tf.reduce_sum(y_true * y_pred))
        self.false_positives.assign_add(tf.reduce_sum((1 - y_true) * y_pred))
        self.true_negatives.assign_add(tf.reduce_sum((1 - y_true) * (1 - y_pred)))
        self.false_negatives.assign_add(tf.reduce_sum(y_true * (1 - y_pred)))
        
    def result(self):
        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-7)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-7)
        
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-7)
        return f1_score
    
    def reset_states(self):
        self.true_positives.assign(0.0)
        self.false_positives.assign(0.0)
        self.true_negatives.assign(0.0)
        self.false_negatives.assign(0.0)
