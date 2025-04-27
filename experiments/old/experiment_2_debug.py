import tensorflow as tf

input_dim = 2
output_dim = 2

sample_size = 2


w1: tf.Variable = tf.Variable(tf.random.normal([input_dim, output_dim]))
b1: tf.Variable = tf.Variable(tf.random.normal([1, output_dim]))

w1 = tf.Variable([[-0.6075731, -1.5482442], [-0.91663176, 0.9762598]])
b1 = tf.Variable([[-1.3270648, -0.5962623]])

x_data = tf.random.uniform(shape=(sample_size, 2), minval=0, maxval=1)
y_data = tf.one_hot(tf.random.uniform(shape=(sample_size,), minval=0, maxval=output_dim, dtype=tf.int32), depth=output_dim)


x_data = tf.constant([[0.36067557, 0.8552792 ],[0.20301294, 0.6795099]])
y_data = tf.constant([[0., 1.], [1., 0.]])

#linear = x_data @ w1 + b1

#activation = tf.nn.softmax(linear)

loss = tf.constant(100000)
activation = None

for i in range(0, 100):
    with tf.GradientTape(persistent=False) as tape:
        tape.watch(w1)
        tape.watch(b1)
        linear = tf.matmul(x_data, w1) + b1
        #print("linear: " + str(linear))
        activation = tf.nn.softmax(linear)
        #print("activation: " + str(activation))
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)(activation, y_data)

    gradients = tape.gradient(loss, [w1, b1])
    dw_gradient = gradients[0: int(len(gradients) / 2)]
    db_gradient = gradients[int(len(gradients) / 2):]


    w1.assign_sub(dw_gradient[0])
    b1.assign_sub(db_gradient[0])
   # print("Gradients")
   # print(dw_gradient[0])
   # print(db_gradient[0])
   # print("Weights")
   # print(w1)
   # print(b1)
   # print("\n\n")

    print(loss)

