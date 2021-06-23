import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

def train():
	print('training model')
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten())
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dense(10, activation='softmax'))

	model.compile(optimizer='adam',
	              loss='sparse_categorical_crossentropy',
	              metrics=['accuracy'])

	model.fit(x_train, y_train, epochs=10,)
	model.save('mnist.h5')
	print("Model saved ")

	score = model.evaluate(x_test, y_test, verbose=0) 
	print('loss=', score[0]) 
	print('accuracy=', score[1])



def accuracy():
	model = tf.keras.models.load_model('mnist.h5')
	predictions = model.predict(x_test[:])
	count = 0
	for x in range(len(predictions)):
	    guess = (np.argmax(predictions[x]))
	    actual = y_test[x]
	    # print("I predict this number is a:", guess)
	    # print("Number Actually Is a:", actual)
	    if guess != actual:
	        count+=1
	        # plt.imshow(x_test[x], cmap=plt.cm.binary)
	        # plt.show()
	        # print("I predict this number is a:", guess)
	        # print("Number Actually Is a:", actual)

	print("The program got", count, 'wrong, out of', len(x_test))
	print(str(100 - ((count/len(x_test))*100)) + '% correct')


def test():
	model = load_model('mnist.h5')
	def predict_digit(img):
	    img = np.array(img)
	#     img = img/255.0
	    img = tf.keras.utils.normalize(img, axis=1)

	    plt.imshow(img, cmap=plt.cm.binary)
	    plt.show()
	    img = img.reshape(1,28,28,1)
	    res = model.predict([img])
	    return np.argmax(res)

	path=r'./test_img.jpg'
	img = image.load_img( path, color_mode='grayscale',
	                     target_size=(28,28), interpolation="nearest")
	print('Predicted digit is: ',predict_digit( img) )
times=0
while(True):

	print('\n'*5,'*'*15 ,'Enter choice', '*'*15)
	print('Enter 1 for Training.\nEnter 2 for accuracy score.\nEnter 3 for testing image.\nEnter 4 to exit' )
	x=int(input())
	print('\n'*3)	
	if(x==1):
		train()
	elif x==2:
		accuracy()
	elif x==3 :
		test()
	elif x==4:
		print('*'*15 ,'Thankyou for using', '*'*15)
		print('Developed by Rahul Yadav')
		break
	else:
		times+=1
		print('Entered wrong choice', times ,'times')
	print('\n'*5)