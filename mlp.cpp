#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>

using namespace std;

class neuron {
public:
  neuron(){;}
  void setNeuron(int setInputSize, float setLearningRate, float setBias);
  void setWeights(int index, float weight);
  void setError(float setError);
  void runNeuron(float * setInputs);
  void updateWeights();
  float outputWeights(int index);
  float outputOutput(){return output;};
  float outputError(){return error;};
private:
  void activationFunction();
  float *inputs;
  float *weights;
  float output;
  float error;
  float learningRate;
  float bias;
  int inputSize;
};

void neuron::setNeuron(int setInputSize, float setLearningRate, float setBias){
  float randomFloat;
  inputSize = setInputSize;
  learningRate = setLearningRate;
  bias = setBias;
  weights = new float[inputSize + 1];

  srand(10);

  for(int i = 0;i <= inputSize; i++){
    randomFloat = (rand()%100);
    randomFloat = (randomFloat/500) - 0.1;
    weights[i] = randomFloat;
  }
}

void neuron::setWeights(int index, float weight){
  weights[index] = weight;
}

void neuron::runNeuron(float * setInputs){
  inputs = setInputs;
  activationFunction();
}

void neuron::setError(float setError){
  error = setError;
}

float neuron::outputWeights(int index){
  return weights[index];
}

void neuron::updateWeights(){
  for(int i = 0; i < inputSize; i++){
    float tempfloat = learningRate;
    weights[i] += learningRate * error * inputs[i];
  }
  weights[inputSize] += learningRate * error * bias;
}

void neuron::activationFunction(){
  output = 0;

  for(int i = 0; i < inputSize;i++)
    output += (weights[i] * inputs[i]);
  output += weights[inputSize] * bias;

  //output = output / 5;

  /*
  if (output >= 0){
    output = 1;
  } else {
    output = -1;
  }
  */

}

class layer {
public:
  layer(){;}
  void setLayer(int setLayerSize, int setInputSize, float setLearningRate, float setBias);
  void neuronCalculate(float * setInputs);  // use if input layer
  void neuronCalculate(layer * prevLayer, float * layerOutputs);  // use if hidden layer
  void neuronBackProp(float * setOutputs);        // use if input layer
  void neuronBackProp(layer * nextLayer);   // use if hidden layer
  void neuronUpdate();
  void printWeights();
  float addWeights();
  void setNeuronWeights(int index, int subIndex, float weight);
  int outputLayerSize(){return layerSize;};
  float outputNeuronWeights(int index, int subIndex);

private:
  neuron * layerContents;
  float neuronOutput(int index);
  float neuronError(int index);
  int layerSize;
  int inputSize;
};

void layer::neuronUpdate(){
  for(int i = 0;i < layerSize; i++){
    layerContents[i].updateWeights();
  }
}

void layer::setNeuronWeights(int index, int subIndex, float weight){
  layerContents[index].setWeights(subIndex, weight);
}

void layer::neuronCalculate(float * setInputs){
  for (int j = 0; j < layerSize; j++){
    layerContents[j].runNeuron(setInputs);
    //cout << "hidden neuron: " << j << " output: " << neuronOutput(j) << endl;
  }
}

void layer::neuronCalculate(layer * prevLayer, float * layerOutputs){
  float * setInputs;
  setInputs = new float[inputSize];

  for (int j = 0; j < layerSize; j++){

    for (int i = 0; i < inputSize; i++){
      setInputs[i] = prevLayer->neuronOutput(i);
    }

    layerContents[j].runNeuron(setInputs);
    layerOutputs[j] = neuronOutput(j);
    //cout << "output neuron: " << j << " output: " << neuronOutput(j) << endl;
  }
}

float layer::neuronOutput(int index){
  return layerContents[index].outputOutput();
}

void layer::neuronBackProp(float * output){
  float error = 0;
  for (int i = 0; i < layerSize; i++){
    error = output[i] - layerContents[i].outputOutput();
    layerContents[i].setError(error);
    //cout << "output neuron: " << i << " error: " << error << endl;
  }
}

void layer::neuronBackProp(layer * nextLayer){
  float error;

  for (int i = 0; i < layerSize; i++){
    error = 0;
    for (int j = 0; j < nextLayer->outputLayerSize(); j++){
      error += nextLayer->neuronError(j) * nextLayer->outputNeuronWeights(j,i);
    }
    layerContents[i].setError(error);
    //cout << "hidden neuron: " << i << " error: " << error << endl;

  }
}

float layer::neuronError(int index){
  return layerContents[index].outputError();
}

float layer::outputNeuronWeights(int index, int subIndex){
  return layerContents[index].outputWeights(subIndex);
}

void layer::printWeights(){
  for(int i = 0; i < layerSize;i++){
      cout << "Neuron: " << i << endl;
      for(int j = 0; j < inputSize;j++){
        cout << " Weight: " <<  layerContents[i].outputWeights(j) << endl;
        }
      cout << " Bias Weight: " << layerContents[i].outputWeights(inputSize) << endl;
  }
}

float layer::addWeights(){
  float tempVal = 0;

  for(int i = 0; i < layerSize;i++){
      for(int j = 0; j < inputSize;j++){
        tempVal += layerContents[i].outputWeights(j);
      }
      tempVal += layerContents[i].outputWeights(inputSize);
  }

  return tempVal;

}

void layer::setLayer(int setLayerSize,int setInputSize, float setLearningRate, float setBias){
  layerSize = setLayerSize;
  inputSize = setInputSize;
  layerContents = new neuron[layerSize];

  for(int i = 0; i < layerSize;i++){
    layerContents[i].setNeuron(inputSize, setLearningRate, setBias);
  }
}

class mlp{
public:
  mlp(int setILSize, int setHLSize, float setHLLearningRate, float setHLBias, int setOLSize, float setOLLearningRate, float setOLBias);
  void mlpRun(float * inputArray, float * trainArray, float * outputArray);
  void mlpRun(float * inputArray, float * outputArray);
  void printWeights();
  float addWeights();
private:
  layer * hiddenLayer;
  layer * outputLayer;
  int networkSize[2];
  void layersCalculate(float * inputArray, float * setOutputArray);
  void layersBackprop(float * trainArray);
  void layersUpdate();
};

mlp::mlp(int setILSize, int setHLSize, float setHLLearningRate, float setHLBias, int setOLSize, float setOLLearningRate, float setOLBias){
  hiddenLayer = new layer();
  outputLayer = new layer();

  hiddenLayer->setLayer(setHLSize,setILSize,setHLLearningRate,setHLBias);
  outputLayer->setLayer(setOLSize,setHLSize,setOLLearningRate,setOLBias);
}

void mlp::mlpRun(float * setInputArray, float * setTrainArray, float * setOutputArray){
  layersCalculate(setInputArray, setOutputArray);
  layersBackprop(setTrainArray);
  layersUpdate();
}

void mlp::mlpRun(float * setInputArray, float * setOutputArray){
  layersCalculate(setInputArray, setOutputArray);
}

void mlp::printWeights(){
  cout << "Hidden Layer: " << endl;
  hiddenLayer->printWeights();
  cout << "Output Layer: " << endl;
  outputLayer->printWeights();
}

float mlp::addWeights(){
  float tempVal = 0;
  tempVal += hiddenLayer->addWeights();
  tempVal += outputLayer->addWeights();

  return tempVal;
}

void mlp::layersCalculate(float * inputArray, float * setOutputArray){
  hiddenLayer->neuronCalculate(inputArray);
  outputLayer->neuronCalculate(hiddenLayer, setOutputArray);
}

void mlp::layersBackprop(float * trainArray){
  outputLayer->neuronBackProp(trainArray);
  hiddenLayer->neuronBackProp(outputLayer);
}

void mlp::layersUpdate(){
  outputLayer->neuronUpdate();
  hiddenLayer->neuronUpdate();
}

class readFile {
public:
  readFile(string fileName);
  void normaliseData();
  bool returnLine(int lineNumber, float * inputArray, float * outputArray);
  int returnDataSize(){return dataSize;}
  int returnDataLength(){return dataLength;}
private:
  vector< float* > inputData;
  vector< float > outputData;
  int dataSize;
  int dataLength;
  float findMaxInputData(int varId);
  float findMinInputData(int varId);
  void readInput(string fileName);
};

readFile::readFile(string fileName){
  readInput(fileName);
}

void readFile::readInput (string fileName){
  ifstream inStream (fileName.c_str());

  inStream >> dataSize;
  inStream >> dataLength;

  int tempOut;

  int lineId = 0;
  while (!inStream.eof ())
    {
      inputData.push_back (new float[dataLength - 1]);
      for (int j = 0; j < dataLength; j++)
    inStream >> inputData[lineId][j];
    inStream >> tempOut;
    outputData.push_back(tempOut);
      lineId++;
    }

}

void readFile::normaliseData(){
  float * maxVals = new float[dataLength];
  float * minVals = new float[dataLength];
  float tempFloat;

  for(int i = 0; i < dataLength; i++){
    maxVals[i] = findMaxInputData(i);
    minVals[i] =  findMinInputData(i);
  }

  for(int y = 0; y < dataSize; y++){
    for(int x = 0; x < dataLength; x++){
      inputData[y][x] = inputData[y][x] - minVals[x];
      tempFloat = maxVals[x] - minVals[x];
      inputData[y][x] = inputData[y][x] / tempFloat;
    }
  }
}

float readFile::findMaxInputData(int varId){
float maxVal = inputData[0][varId];
    for (int i=1; i < dataSize; i++) {
        if (inputData[i][varId] > maxVal) {
            maxVal = inputData[i][varId];
        }
    }
    return maxVal;
}

float readFile::findMinInputData(int varId){
float minVal = inputData[0][varId];
    for (int i=1; i < dataSize; i++) {
        if (inputData[i][varId] < minVal) {
            minVal = inputData[i][varId];
        }
    }
    return minVal;
}

bool readFile::returnLine(int lineNumber, float * inputArray, float * outputArray){
  if (lineNumber < dataSize){
    for (int i = 0; i < dataLength; i++){
      inputArray[i] = inputData[lineNumber][i];
    }
    outputArray[0] = outputData[lineNumber];
    return true;
  } else{
    return false;
  }
}

void main(){
  float * inputExamples;
  float * outputExamples;

  readFile learningSet("learn.txt");
  learningSet.normaliseData();

  readFile testSet("test.txt");
  testSet.normaliseData();

  inputExamples = new float[learningSet.returnDataSize() - 1];
  outputExamples = new float[1];

  int setILSize = learningSet.returnDataLength();
  int setHLSize = 2;
  float setHLLearningRate = 0.1;
  float setHLBias = 1;
  int setOLSize = 1;
  float setOLLearningRate = 0.5;
  float setOLBias = 1;
  float testStop = 0;
  bool converged = false;
  int epochCount = 0;

  float * outputArray = new float[setOLSize]; // same as output layer size
  mlp spliceSitesPerceptron(setILSize, setHLSize, setHLLearningRate, setHLBias, setOLSize, setOLLearningRate, setOLBias);

  while (!converged){
    epochCount++;

    for (int i = 0;i < learningSet.returnDataSize(); i++){
      learningSet.returnLine(i,inputExamples,outputExamples);
      spliceSitesPerceptron.mlpRun(inputExamples,outputExamples,outputArray);
    }

    if (testStop == int(spliceSitesPerceptron.addWeights()*1000)){
      converged = true;
    }

    //cout << testStop << endl;

    testStop = int(spliceSitesPerceptron.addWeights() * 1000);

  }

    cout << "Epoch: " << epochCount << endl << endl;
    spliceSitesPerceptron.printWeights();
    cout << endl;

    float testAccuracyClass1 = 0;
    float testAccuracyClass2 = 0;
    float testInAccuracyClass1 = 0;
    float testInAccuracyClass2 = 0;
    float trainAccuracyClass1 = 0;
    float trainAccuracyClass2 = 0;
    float trainInAccuracyClass1 = 0;
    float trainInAccuracyClass2 = 0;
    float tempAccuracy = 0;
    int testInt = 0;

  for (int trainIndex = 0; trainIndex < 5999; trainIndex++){
    learningSet.returnLine(trainIndex,inputExamples,outputExamples);
    spliceSitesPerceptron.mlpRun(inputExamples, outputArray);

    if (outputArray[0] > 0){
      testInt = 1;
    } else {
      testInt = -1;
    }

    if (outputExamples[0] == testInt){
      if (outputExamples[0] > 0){
        trainAccuracyClass1++;
      } else {
        trainAccuracyClass2++;
      }
    } else {
      if (outputExamples[0] > 0){
        trainInAccuracyClass1++;
      } else {
        trainInAccuracyClass2++;
      }
    }

    //cout << "target value: " << outputExamples[0] << endl;
    //cout << "output value: " << sign(outputArray[0]) << endl;
  }

  cout << "Accuracy: " << endl;
  tempAccuracy = int((trainAccuracyClass1 /(trainInAccuracyClass1 + trainAccuracyClass1))*100);
  cout << "Training set (class 1): " << tempAccuracy << endl;
  //cout << "Accuracy on training set (class 1): " << trainAccuracyClass1 <<  "/" << trainInAccuracyClass1 +  trainAccuracyClass1<< endl;
  tempAccuracy = int((trainAccuracyClass2 /(trainInAccuracyClass2 + trainAccuracyClass2))*100);
  cout << "Training set (class 2): " << tempAccuracy << endl;
  //cout << "Accuracy on training set (class 2): " << trainAccuracyClass2 << "/" << trainInAccuracyClass2 + trainAccuracyClass2<< endl;

  cout << "Total on training set: " << int((((trainAccuracyClass1 /(trainInAccuracyClass1 + trainAccuracyClass1))+(trainAccuracyClass2 /(trainInAccuracyClass2 + trainAccuracyClass2)))/2)*100) << endl;

  for (int testIndex = 0; testIndex < 6000; testIndex++){
    testSet.returnLine(testIndex,inputExamples,outputExamples);
    spliceSitesPerceptron.mlpRun(inputExamples, outputArray);

    if (outputArray[0] > 0){
      testInt = 1;
    } else {
      testInt = -1;
    }

    if (outputExamples[0] == testInt){
      if (outputExamples[0] > 0){
        testAccuracyClass1++;
      } else {
        testAccuracyClass2++;
      }
    } else {
      if (outputExamples[0] > 0){
        testInAccuracyClass1++;
      } else {
        testInAccuracyClass2++;
      }
    }

    //cout << "target value: " << outputExamples[0] << endl;
    //cout << "output value: " << sign(outputArray[0]) << endl;
  }

  tempAccuracy = int((testAccuracyClass1 /(testInAccuracyClass1 + testAccuracyClass1))*100);
  cout << "Test set (class 1): " << tempAccuracy << endl;
  //cout << "Accuracy on test set (class 1): " << testAccuracyClass1 << "/" << testInAccuracyClass1 + testAccuracyClass1<< endl;
  tempAccuracy = int((testAccuracyClass2 /(testInAccuracyClass2 + testAccuracyClass2))*100);
  cout << "Test set (class 2): " << tempAccuracy <<endl;
  //cout << "Accuracy on test set (class 2): " << testAccuracyClass2 << "/" << testInAccuracyClass2 + testAccuracyClass2<< endl;


  cout << "Total on test set: " << int((((testAccuracyClass1 /(testInAccuracyClass1 + testAccuracyClass1))+(testAccuracyClass2 /(testInAccuracyClass2 + testAccuracyClass2)))/2)*100) << endl;

  cout << endl;

  //spliceSitesPerceptron.printWeights();
}

