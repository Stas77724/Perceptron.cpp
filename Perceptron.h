#pragma once
#include <vector>
#include <string>

class NeuralClass
{
public:
    // inputSize по умолчанию 64*64, 128 скрытых нейронов, 6 классов ( Default input size is 64×64, 128 hidden neurons, 6 classes )
    NeuralClass(int inputSize = 64 * 64, int hiddenSize = 128, int outputSize = 6);

    std::vector<double> forward(const std::vector<double>& x);   // возвращает вероятности (softmax) ( returns probabilities (softmax) )
    void train(const std::vector<double>& x, const std::vector<double>& target);
    int predict(const std::vector<double>& x);

    void setLearningRate(double v) { lr = v; }
    double getLearningRate() const { return lr; }

    void saveWeights(const std::string& path);
    bool loadWeights(const std::string& path);   // false, если файла нет или размеры не совпали (false if the file does not exist or the sizes do not match)

private:
    int inputSize, hiddenSize, outputSize;

    std::vector<std::vector<double>> W1; // вход -> скрытый  [input][hidden] ( input -> hidden  [input][hidden] )
    std::vector<std::vector<double>> W2; // скрытый -> выход [hidden][output] ( hidden -> output [hidden][output] )
    std::vector<double> B1, B2, h, y;

    double lr = 0.002;
};
