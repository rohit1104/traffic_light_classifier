import numpy as np
import cv2
from glob import glob
import os.path
import tensorflow as tf
import time

EPOCHS = 50
IMAGES_PER_BATCH = 10
NUM_CLASSES = 3


red_image_paths = glob(os.path.join('../../dataset/red/', '*.jpg'))
red_image_paths = np.array(red_image_paths)
train_red_paths, test_red_paths = np.split(red_image_paths[np.random.permutation(red_image_paths.shape[0])], [int(red_image_paths.shape[0]*0.8)])

not_red_image_paths = glob(os.path.join('../../dataset/not_red/', '*.jpg'))
not_red_image_paths = np.array(not_red_image_paths)
train_not_red_paths, test_not_red_paths = np.split(not_red_image_paths[np.random.permutation(not_red_image_paths.shape[0])], [int(not_red_image_paths.shape[0]*0.8)])

not_light_image_paths = glob(os.path.join('../../dataset/not_light/', '*.jpg'))
not_light_image_paths = np.array(not_light_image_paths)
train_not_light_paths, test_not_light_paths = np.split(not_light_image_paths[np.random.permutation(not_light_image_paths.shape[0])], [int(not_light_image_paths.shape[0]*0.8)])

train_paths = np.concatenate([train_red_paths, train_not_red_paths, train_not_light_paths])
test_paths = np.concatenate([test_red_paths, test_not_red_paths, test_not_light_paths])

train_labels = np.concatenate([0*np.ones(len(train_red_paths)), 1*np.ones(len(train_not_red_paths)), 2*np.ones(len(train_not_light_paths))])
test_labels = np.concatenate([0*np.ones(len(test_red_paths)), 1*np.ones(len(test_not_red_paths)), 2*np.ones(len(test_not_light_paths))])

def generate_final_image(image):
	image = cv2.resize(image, (800, 600))
	image = image[0:300,:,:]

	hls = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2HLS), (125,50))
	gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (125,50))

	s_ch = hls[:,:,2]

	final_image = np.dstack([s_ch, gray])
	return final_image

def get_train_batches(batch_size):
	global train_paths
	global train_labels

	s = np.arange(train_labels.shape[0])
	np.random.shuffle(s)
	train_paths = train_paths[s]
	train_labels = train_labels[s]

	for batch_i in range(0, len(train_labels), batch_size):
		images = []
		labels = []

		for image_path,label in zip(train_paths[batch_i:batch_i+batch_size],train_labels[batch_i:batch_i+batch_size]):
			final_image = generate_final_image(cv2.imread(image_path))

			images.append(final_image)
			labels.append(label)

			images.append(np.fliplr(final_image))
			labels.append(label)

		yield np.array(images), np.array(labels)

def get_test_batches(batch_size):
	global test_paths
	global test_labels

	for batch_i in range(0, len(test_labels), batch_size):
		images = []
		labels = []

		for image_path,label in zip(test_paths[batch_i:batch_i+batch_size],test_labels[batch_i:batch_i+batch_size]):
			final_image = generate_final_image(cv2.imread(image_path))

			images.append(final_image)
			labels.append(label)

		yield np.array(images), np.array(labels)

input_layer = tf.placeholder(tf.float32, shape=[None,50,125,2], name='input_layer')
y = tf.placeholder(tf.int32, shape=[None])

normalized = tf.divide(input_layer, tf.constant(255.0))

conv1_1 = tf.layers.conv2d(normalized, 40, (3,3), padding='same', activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(conv1_1, (2,2), (2,2))

conv2_1 = tf.layers.conv2d(pool1, 50, (3,3), padding='same', activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(conv2_1, (2,2), (2,2))

conv3 = tf.layers.conv2d(pool2, 60, (3,3), padding='same', activation=tf.nn.relu)
pool3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2))

#conv4 = tf.layers.conv2d(pool3, 60, (3,3), padding='same', activation=tf.nn.relu)
#pool4 = tf.layers.max_pooling2d(conv4, (2,2), (2,2))

final_pool_layer = tf.layers.max_pooling2d(pool3, (6,15), (1,1))

conv_flat = tf.reshape(final_pool_layer, [-1, 60])
#dense = tf.layers.dense(conv_flat, units=50, activation=tf.nn.relu)
logits = tf.layers.dense(conv_flat, units=NUM_CLASSES, name='logits')

one_hot_y = tf.one_hot(y, NUM_CLASSES)

# training functions
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=one_hot_y))
train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy_loss)

# accuracy functions
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
conf_mat = tf.confusion_matrix(tf.argmax(one_hot_y, 1), tf.argmax(logits, 1), NUM_CLASSES)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	#saver = tf.train.Saver()

	print("Training starting for {} samples".format(len(train_labels)*2))

	for epoch in range(EPOCHS):
		t1 = time.time()
		loss = 0.0
		train_accuracy = 0.0
		samples = 0.0
		conf = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.int32)
		for image_batch, label_batch in get_train_batches(IMAGES_PER_BATCH):
			_, curr_accuracy, curr_loss, curr_conf = sess.run([train_op, accuracy_op, cross_entropy_loss, conf_mat], feed_dict = {input_layer: image_batch, y: label_batch})
			loss += (curr_loss*len(label_batch))
			samples += len(label_batch)
			train_accuracy += (curr_accuracy*len(label_batch))
			conf = np.add(conf, curr_conf)
		loss /= samples
		train_accuracy /= samples
		t2 = time.time()
		print("Epoch {}: Training loss = {}, Training Accuracy = {}, time taken = {} seconds, Confusion Matrix =".format(epoch+1, loss, train_accuracy*100, (t2-t1)))
		print(conf)

		t1 = time.time()
		test_accuracy = 0.0
		samples = 0.0
		conf = np.zeros((NUM_CLASSES,NUM_CLASSES), dtype=np.int32)
		for image_batch, label_batch in get_test_batches(IMAGES_PER_BATCH):
			curr_accuracy, curr_conf = sess.run([accuracy_op, conf_mat], feed_dict = {input_layer: image_batch, y: label_batch})
			samples += len(label_batch)
			test_accuracy += (curr_accuracy*len(label_batch))
			conf = np.add(conf, curr_conf)
		test_accuracy /= samples
		t2 = time.time()
		print("Test Accuracy = {}, time taken = {} seconds, Confusion Matrix =".format(test_accuracy*100, (t2-t1)))
		print(conf)

		saver = tf.train.Saver()
		saver.save(sess, 'my_traffic_light_model', global_step=epoch+1)

		if test_accuracy >= 0.995 and train_accuracy >= 0.995:
			break


