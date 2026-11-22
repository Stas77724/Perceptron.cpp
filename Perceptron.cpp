#include "Perceptron.h"
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <algorithm>

static double randUniform(double a, double b)
{
    return a + (b - a) * ((double)rand() / RAND_MAX);
}

NeuralClass::NeuralClass(int in, int hid, int out)
    : inputSize(in), hiddenSize(hid), outputSize(out)
{
    W1.assign(inputSize, std::vector<double>(hiddenSize));
    W2.assign(hiddenSize, std::vector<double>(outputSize));
    B1.assign(hiddenSize, 0.0);
    B2.assign(outputSize, 0.0);
    h.assign(hiddenSize, 0.0);
    y.assign(outputSize, 0.0);

    // Инициализация Xavier: диапазон зависит от размеров слоя ( Xavier Initialization: The range depends on the layer size )
    double lim1 = std::sqrt(6.0 / (inputSize + hiddenSize));
    double lim2 = std::sqrt(6.0 / (hiddenSize + outputSize));
    for (auto& row : W1) for (auto& w : row) w = randUniform(-lim1, lim1);
    for (auto& row : W2) for (auto& w : row) w = randUniform(-lim2, lim2);
}

std::vector<double> NeuralClass::forward(const std::vector<double>& x)
{
    // скрытый слой: ReLU. Нулевые пиксели пропускаем - картинки в основном чёрные, это сильно ускоряет ( Hidden layer: ReLU. We skip zero pixels—the images are mostly black, which speeds things up significantly )
    for (int j = 0; j < hiddenSize; j++) h[j] = B1[j];
    for (int i = 0; i < inputSize; i++) {
        if (x[i] == 0.0) continue;
        const std::vector<double>& row = W1[i];
        for (int j = 0; j < hiddenSize; j++) h[j] += x[i] * row[j];
    }
    for (int j = 0; j < hiddenSize; j++) if (h[j] < 0) h[j] = 0;

    // выходной слой: softmax ( input layer: softmax )
    double mx = -1e300;
    for (int j = 0; j < outputSize; j++) {
        double s = B2[j];
        for (int i = 0; i < hiddenSize; i++) s += h[i] * W2[i][j];
        y[j] = s;
        mx = std::max(mx, s);
    }
    double sum = 0;
    for (int j = 0; j < outputSize; j++) { y[j] = std::exp(y[j] - mx); sum += y[j]; }
    for (int j = 0; j < outputSize; j++) y[j] /= sum;
    return y;
}

int NeuralClass::predict(const std::vector<double>& x)
{
    std::vector<double> out = forward(x);
    int best = 0;
    for (int i = 1; i < outputSize; i++) if (out[i] > out[best]) best = i;
    return best;
}

void NeuralClass::train(const std::vector<double>& x, const std::vector<double>& target)
{
    std::vector<double> out = forward(x);

    // softmax + cross-entropy: градиент на выходе = (target - out) ( Softmax + cross-entropy: output gradient = (target - output) )
    std::vector<double> d_out(outputSize);
    for (int j = 0; j < outputSize; j++) d_out[j] = target[j] - out[j];

    // градиент скрытого слоя (производная ReLU: 1 если h>0) ( the gradient of the hidden layer (the derivative of ReLU: 1 if h > 0) )
    std::vector<double> d_h(hiddenSize, 0.0);
    for (int j = 0; j < hiddenSize; j++) {
        if (h[j] <= 0) continue;
        double s = 0;
        for (int k = 0; k < outputSize; k++) s += d_out[k] * W2[j][k];
        d_h[j] = s;
    }

    for (int i = 0; i < hiddenSize; i++)
        for (int j = 0; j < outputSize; j++) W2[i][j] += lr * d_out[j] * h[i];
    for (int j = 0; j < outputSize; j++) B2[j] += lr * d_out[j];

    for (int i = 0; i < inputSize; i++) {
        if (x[i] == 0.0) continue;
        std::vector<double>& row = W1[i];
        for (int j = 0; j < hiddenSize; j++) row[j] += lr * d_h[j] * x[i];
    }
    for (int j = 0; j < hiddenSize; j++) B1[j] += lr * d_h[j];
}

void NeuralClass::saveWeights(const std::string& path)
{
    std::ofstream f(path, std::ios::binary);
    f.write((char*)&inputSize, sizeof(int));
    f.write((char*)&hiddenSize, sizeof(int));
    f.write((char*)&outputSize, sizeof(int));
    for (auto& r : W1) for (double w : r) f.write((char*)&w, sizeof(double));
    for (auto& r : W2) for (double w : r) f.write((char*)&w, sizeof(double));
    for (double w : B1) f.write((char*)&w, sizeof(double));
    for (double w : B2) f.write((char*)&w, sizeof(double));
    std::cout << "Weights saved: " << path << std::endl;
}

bool NeuralClass::loadWeights(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) { std::cout << "Weights file not found\n"; return false; }
    int a = 0, b = 0, c = 0;
    f.read((char*)&a, sizeof(int));
    f.read((char*)&b, sizeof(int));
    f.read((char*)&c, sizeof(int));
    if (a != inputSize || b != hiddenSize || c != outputSize) {
        std::cout << "Weights file does not match the network size - delete it and train again\n";
        return false;
    }
    for (auto& r : W1) for (double& w : r) f.read((char*)&w, sizeof(double));
    for (auto& r : W2) for (double& w : r) f.read((char*)&w, sizeof(double));
    for (double& w : B1) f.read((char*)&w, sizeof(double));
    for (double& w : B2) f.read((char*)&w, sizeof(double));
    std::cout << "Weights loaded: " << path << std::endl;
    return true;
}
