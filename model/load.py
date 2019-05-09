from keras.models import model_from_json
from keras.models import load_model
import tensorflow as tf 

def init():
	# json_file = open('model_architecture.json','r')
	# loaded_model_json = json_file.read()
	# json_file.close()
	# loaded_model = model_from_json(loaded_model_json)
	# # load weights into new model
	# loaded_model.load_weights("model_weights.h5")
	# print("Loaded Model from disk")


	json_file = open('./model/model_num.json', 'r')

	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into new models
	loaded_model.load_weights("./model/model_num.h5")
	print("Loaded model from disk")



	#compile and evaluate loaded model
	loaded_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'] )
	graph = tf.get_default_graph()

	return loaded_model, graph