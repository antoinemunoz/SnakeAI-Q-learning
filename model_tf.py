import tensorflow as tf
from tensorflow.keras import layers, Model

class QNetwork(Model):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(QNetwork, self).__init__()
        # Pas besoin d'InputLayer : Keras infère la forme d'entrée
        self.dense1 = layers.Dense(hidden_dim, activation='relu', input_shape=(input_dim,))
        self.dense2 = layers.Dense(hidden_dim // 2, activation='relu')
        self.out = layers.Dense(output_dim)

    def call(self, x):
        x = tf.cast(x, tf.float32)
        x = self.dense1(x)
        x = self.dense2(x)
        return self.out(x)

class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    @tf.function
    def train_step(self, states, actions, rewards, next_states, dones):
        # Calcul du Q-cible
        next_q = self.model(next_states)
        max_next_q = tf.reduce_max(next_q, axis=1)
        target_q = rewards + (1 - dones) * self.gamma * max_next_q

        with tf.GradientTape() as tape:
            q_values = self.model(states)
            # Indices d'action, on assure même dtype
            batch_size = tf.shape(actions)[0]
            indices = tf.stack([
                tf.range(batch_size, dtype=tf.int64),
                tf.cast(tf.argmax(actions, axis=1), tf.int64)
            ], axis=1)
            q_pred = tf.gather_nd(q_values, indices)
            loss = self.loss_fn(target_q, q_pred)

        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss