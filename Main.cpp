#define _CRT_SECURE_NO_WARNINGS
#include "Perceptron.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <string>
#include <set>
#include <opencv2/core/utils/filesystem.hpp>
#include <random>
#include <numeric>

using namespace std;

const int SIZE = 64;
const int NUM_CLASSES = 6;
const double PI = 3.14159265358979;


cv::Mat shapeToImage(const std::vector<double>& shape)
{
    cv::Mat img(SIZE, SIZE, CV_8UC1);
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            img.at<unsigned char>(i, j) = (unsigned char)(shape[i * SIZE + j] * 255);

    cv::Mat big;
    cv::resize(img, big, cv::Size(400, 400), 0, 0, cv::INTER_NEAREST);
    cv::cvtColor(big, big, cv::COLOR_GRAY2BGR);
    return big;
}

std::vector<double> prepareImage(const std::string& path, cv::Mat* preview = nullptr)
{
    cv::Mat img = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (img.empty())
    {
        std::cout << "Не удалось открыть: " << path << std::endl;
        return {};
    }

    // 1. Бинаризация (порог подбирается автоматически) ( 1. Binarization (the threshold is selected automatically) )
    cv::Mat bin;
    cv::threshold(img, bin, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

    // 2. Фон должен быть чёрным: смотрим цвет по краям картинки ( 2. The background must be black: check the color at the edges of the image )
    double border = (cv::mean(bin.row(0))[0] + cv::mean(bin.row(bin.rows - 1))[0] +
        cv::mean(bin.col(0))[0] + cv::mean(bin.col(bin.cols - 1))[0]) / 4.0;
    if (border > 127) cv::bitwise_not(bin, bin);

    // 3. Находим самый большой объект и закрашиваем его ( 3. Find the largest object and fill it in )
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bin.clone(), contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) return {};

    int best = 0;
    double bestArea = 0;
    for (int i = 0; i < (int)contours.size(); i++)
    {
        double a = cv::contourArea(contours[i]);
        if (a > bestArea) { bestArea = a; best = i; }
    }

    cv::Mat filled = cv::Mat::zeros(bin.size(), CV_8UC1);
    cv::drawContours(filled, contours, best, cv::Scalar(255), cv::FILLED);

    // 4. Вырезаем фигуру, масштабируем и ставим в центр 64x64 (  4. Cut out the shape, resize it, and place it in the center of the 64x64 area )
    cv::Rect box = cv::boundingRect(contours[best]);
    cv::Mat crop = filled(box);
    double scale = 48.0 / std::max(box.width, box.height);

    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(), scale, scale, cv::INTER_AREA);
    cv::threshold(resized, resized, 127, 255, cv::THRESH_BINARY);

    cv::Mat canvas = cv::Mat::zeros(SIZE, SIZE, CV_8UC1);
    int x = (SIZE - resized.cols) / 2;
    int y = (SIZE - resized.rows) / 2;
    resized.copyTo(canvas(cv::Rect(x, y, resized.cols, resized.rows)));

    if (preview) *preview = canvas.clone();

    // 5. В вектор из 0 и 1 ( 5. A vector consisting of 0s and 1s )
    std::vector<double> result;
    result.reserve(SIZE * SIZE);
    for (int i = 0; i < SIZE; i++)
        for (int j = 0; j < SIZE; j++)
            result.push_back(canvas.at<unsigned char>(i, j) > 127 ? 1.0 : 0.0);
    return result;
}

void loadDataset(
    const string& folder,
    vector<vector<double>>& data,
    vector<int>& labels,
    const string* classNames)
{
    for (int type = 0; type < NUM_CLASSES; type++)
    {
        string path = folder + "/" + classNames[type];

        cout << "Checking folder: " << path << endl;

        vector<cv::String> files;

        if (!cv::utils::fs::exists(path))
        {
            cout << "Folder does not exist!" << endl;
            continue;
        }

        cv::glob(path + "/*", files, false);

        int count = 0;

        for (const auto& file : files)
        {
            // та же предобработка, что и при распознавании: обрезка, масштаб, центр (Same preprocessing as for recognition: cropping, scaling, centering)
            vector<double> input = prepareImage(file);
            if (input.empty())
                continue;

            data.push_back(input);
            labels.push_back(type);
            count++;
        }

        cout << classNames[type] << ": "
            << count << " images loaded" << endl;
    }
}

void drawTrainingGraph(const std::vector<double>& loss,
    const std::vector<double>& acc,
    const std::string& savePath = "training_graph.png")
{
    const int W = 900, H = 500, L = 70, R = 30, T = 40, B = 60;
    cv::Mat img(H, W, CV_8UC3, cv::Scalar(255, 255, 255));
    int n = (int)loss.size();
    if (n < 2) return;

    double maxLoss = *std::max_element(loss.begin(), loss.end());
    if (maxLoss <= 0) maxLoss = 1;

    auto px = [&](int i) { return L + (int)((W - L - R) * (double)i / (n - 1)); };
    auto pyLoss = [&](double v) { return H - B - (int)((H - T - B) * v / maxLoss); };
    auto pyAcc = [&](double v) { return H - B - (int)((H - T - B) * v / 100.0); };

    // Оси (Axes)
    cv::line(img, cv::Point(L, T), cv::Point(L, H - B), cv::Scalar(0, 0, 0), 1);
    cv::line(img, cv::Point(L, H - B), cv::Point(W - R, H - B), cv::Scalar(0, 0, 0), 1);

    // Сетка и подписи по вертикали (точность в %) (a network and signatures on veritcal lines (accurancy in %))
    for (int p = 0; p <= 100; p += 20)
    {
        int y = pyAcc(p);
        cv::line(img, cv::Point(L, y), cv::Point(W - R, y), cv::Scalar(225, 225, 225), 1);
        cv::putText(img, std::to_string(p) + "%", cv::Point(25, y + 5),
            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 120, 0), 1);
    }

    // Подписи эпох (signatures of the epochs)
    int step = std::max(1, n / 10);
    for (int i = 0; i < n; i += step)
        cv::putText(img, std::to_string(i + 1), cv::Point(px(i) - 8, H - B + 20),
            cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    cv::putText(img, "Epoch", cv::Point(W / 2 - 20, H - 15),
        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    // Линии: loss (красная) и accuracy (зелёная) (Lines: loss (red) and accuracy (green))
    for (int i = 1; i < n; i++)
    {
        cv::line(img, cv::Point(px(i - 1), pyLoss(loss[i - 1])),
            cv::Point(px(i), pyLoss(loss[i])), cv::Scalar(0, 0, 220), 2);
        cv::line(img, cv::Point(px(i - 1), pyAcc(acc[i - 1])),
            cv::Point(px(i), pyAcc(acc[i])), cv::Scalar(0, 170, 0), 2);
    }

    // Легенда (the legend)
    cv::putText(img, "Loss (max " + std::to_string(maxLoss).substr(0, 5) + ")",
        cv::Point(L + 10, 25), cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 0, 220), 2);
    cv::putText(img, "Accuracy %", cv::Point(L + 250, 25),
        cv::FONT_HERSHEY_SIMPLEX, 0.55, cv::Scalar(0, 170, 0), 2);

    cv::imwrite(savePath, img);
    cv::imshow("Training graph", img);
}
void processFolder(NeuralClass& nn, const std::string& folder,
    const std::string* classNames,
    std::set<std::string>& done)
{
    std::vector<cv::String> files;
    cv::glob(folder + "/*.*", files, false);

    for (const auto& f : files) {
        std::string path = f;
        std::string low = path;
        for (auto& c : low) c = (char)tolower(c);
        bool ok = low.size() > 4 &&
            (low.rfind(".png") == low.size() - 4 || low.rfind(".jpg") == low.size() - 4 ||
                low.rfind(".bmp") == low.size() - 4 || low.rfind(".jpeg") == low.size() - 5);
        if (!ok || done.count(path)) continue;
        done.insert(path);

        cv::Mat preview;
        std::vector<double> in = prepareImage(path, &preview);
        if (in.empty()) continue;

        std::vector<double> out = nn.forward(in);
        int best = 0;
        for (int i = 1; i < (int)out.size(); i++) if (out[i] > out[best]) best = i;

        std::string name = path.substr(path.find_last_of("/\\") + 1);
        std::cout << name << " -> " << classNames[best]
            << " (" << (int)(out[best] * 100) << "%)" << std::endl;

        // оригинал + то, что увидела сеть, рядом (original + what the network sees near)
        cv::Mat orig = cv::imread(path);
        cv::resize(orig, orig, cv::Size(300, 300));
        cv::Mat prev3, prevBig;
        cv::resize(preview, prevBig, cv::Size(300, 300), 0, 0, cv::INTER_NEAREST);
        if (prevBig.channels() == 1) cv::cvtColor(prevBig, prev3, cv::COLOR_GRAY2BGR);
        else prev3 = prevBig;

        cv::Mat canvas;
        cv::hconcat(orig, prev3, canvas);

        std::string label = classNames[best] + " " + std::to_string((int)(out[best] * 100)) + "%";
        cv::rectangle(canvas, cv::Rect(0, 0, canvas.cols, 40), cv::Scalar(0, 0, 0), cv::FILLED);
        cv::putText(canvas, label, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX,
            1.0, cv::Scalar(0, 255, 0), 2);

        cv::imshow("Result", canvas);
        int key = cv::waitKey(0);          // любая клавиша = следующее изображение (any key = next image) 
        if (key == 27) exit(0);            // Esc = выход (ESC = exit)
    }
}

int main()
{
    NeuralClass nn;

    string weightsPath = "weights_shapes.bin";

    string classNames[NUM_CLASSES] = {
    "1_circle",
    "2_square",
    "3_triangle",
    "4_diamond",
    "5_inv_triangle",
    "6_octagon"
    };

    srand(42);

    // Данные объявляем здесь, чтобы они были доступны (we update the data here so that they are aviable in the entire main() function)
    // во всей функции main() 
    vector<vector<double>> data;
    vector<int> labels;

    string datasetPath = "D:/dataset_shapes";

    // Загружаем изображения из папок
    loadDataset(datasetPath, data, labels, classNames);

    cout << "Dataset: " << data.size() << " images" << endl;

    // ---------- Загрузка весов или обучение (loading the weights or training)---------- 
    bool loaded = false;
    {
        ifstream check(weightsPath, ios::binary);
        if (check.is_open()) { check.close(); loaded = nn.loadWeights(weightsPath); }
    }

    if (!loaded)
    {
        if (data.empty()) { cout << "Dataset is empty!" << endl; return 1; }

        cout << "Starting training..." << endl;

        // делим на обучающую (80%) и проверочную (20%) части (we divide it into a training section (80%) and an assessment section (20%))
        vector<int> all(data.size());
        iota(all.begin(), all.end(), 0);
        random_device rd;
        mt19937 g(rd());
        shuffle(all.begin(), all.end(), g);

        size_t nVal = data.size() / 5;
        vector<int> valIdx(all.begin(), all.begin() + nVal);
        vector<int> trainIdx(all.begin() + nVal, all.end());
        cout << "Train: " << trainIdx.size() << ", validation: " << valIdx.size() << endl;

        vector<double> lossHistory, accHistory;
        double bestVal = -1;
        const int EPOCHS = 80;

        for (int epoch = 0; epoch < EPOCHS; epoch++)
        {
            nn.setLearningRate(0.002 / (1.0 + 0.03 * epoch));   // скорость падает по ходу обучения (speed decreases as training progresses)
            shuffle(trainIdx.begin(), trainIdx.end(), g);

            for (int idx : trainIdx)
            {
                vector<double> target(NUM_CLASSES, 0.0);
                target[labels[idx]] = 1.0;
                nn.train(data[idx], target);
            }

            // точность на обучающих (до 600 штук) и на проверочных (accuracy on training sets (up to 600 examples) and on test sets)
            int okTrain = 0, nTr = min(600, (int)trainIdx.size());
            for (int k = 0; k < nTr; k++)
                if (nn.predict(data[trainIdx[k]]) == labels[trainIdx[k]]) okTrain++;

            double loss = 0.0; int okVal = 0;
            for (int idx : valIdx)
            {
                vector<double> out = nn.forward(data[idx]);
                loss += -log(out[labels[idx]] + 1e-12);        // cross-entropy
                int best = 0;
                for (int c = 1; c < NUM_CLASSES; c++) if (out[c] > out[best]) best = c;
                if (best == labels[idx]) okVal++;
            }
            double valAcc = okVal * 100.0 / valIdx.size();
            lossHistory.push_back(loss / valIdx.size());
            accHistory.push_back(valAcc);

            cout << "Epoch " << epoch + 1
                << " val_loss = " << lossHistory.back()
                << " train_acc = " << okTrain * 100.0 / nTr << "%"
                << " val_acc = " << valAcc << "%" << endl;

            // сохраняем веса лучшей эпохи (Saving the weights of the epoches)
            if (valAcc > bestVal) { bestVal = valAcc; nn.saveWeights(weightsPath); }
        }

        nn.loadWeights(weightsPath);   // возвращаем лучшие веса
        cout << "Best validation accuracy: " << bestVal << "%" << endl;

        drawTrainingGraph(lossHistory, accHistory);
        cv::waitKey(0);
        cv::destroyAllWindows();
    }

    // ---------- Проверка точности на всём датасете (Accuracy check across the entire dataset) ----------
    if (!data.empty())
    {
        int correct = 0;
        for (int i = 0; i < (int)data.size(); i++)
            if (nn.predict(data[i]) == labels[i]) correct++;

        cout << "Dataset accuracy: " << correct << "/" << data.size()
            << " (" << correct * 100.0 / data.size() << "%)" << endl;
    }

    // ---------- Интерактивное распознавание (Interactive Recognition) ----------
    string folder = "D:/test";

    set<string> done;

    cout << "Monitoring folder: " << folder
        << "\nPut images into this folder."
        << "\nPress ESC to exit." << endl;

    cv::Mat status = cv::Mat::zeros(100, 600, CV_8UC3);

    cv::putText(
        status,
        "Monitoring folder... Press ESC to exit.",
        cv::Point(10, 55),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6,
        cv::Scalar(255, 255, 255),
        1
    );

    cv::imshow("Status", status);

    while (true)
    {
        processFolder(nn, folder, classNames, done);

        int key = cv::waitKey(2000);

        if (key == 27)
            break;
    }

    cv::destroyAllWindows();

    return 0;
}
