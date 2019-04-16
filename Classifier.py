import abc
from abc import abstractmethod

class Classifier:
	__metaclass__ = abc.ABCMeta
	@classmethod
	def __init__(self): #constructor for the abstract class
		pass

	#This is the abstract method that is implemented by the subclasses.
	@abstractmethod
	def buildClassifier(self, audio_file, audio_label):
		pass

	@abstractmethod
	def predict(self,audio_feature,clf):
		pass

