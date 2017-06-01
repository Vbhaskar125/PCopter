import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

class deepnet:
    def __init__(self):
        epoch=10
        num_classes=2
        self.fullmodel=self.buildModel()


    def buildModel(self):
        model=Sequential()
        model.add(Dense(input_dim=9,activation='relu',output_dim=6))
        model.add(Dense(input_dim=6,activation='softmax',output_dim=4))
        model.add(Dense(input_dim=4,activation='tanh',output_dim=2))
        return model

    def getDataarray(self,data):
        self.y_train = keras.utils.to_categorical(y_train, num_classes)
        self.y_test = keras.utils.to_categorical(y_test, num_classes)
        return list(data)

    def train(self,dataa):
        trainX,trainY=self.getDataarray(dataa)
        self.fullmodel.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])
