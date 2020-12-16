#ifndef DETECTOR_H
#define DETECTOR_H
#include<opencv2/opencv.hpp>
#include<vector>
#include<random>
#include"Extractor.h"
using namespace std;
namespace PED_CAL
{
class PeriodicDetector
{
private:
std::vector<Posture> data;
public:
PeriodicDetector();
void push_back(Posture posture);
void clear();
const size_t getSize();
std::vector<Posture> ampd();
std::vector<Posture>&getData();

};
}
#endif