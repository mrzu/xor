
#include "stdafx.h"

#include <iostream>
#include <stdlib.h> 

static int const NumHidden = 13;
static int const NumInputs = 4;
static int const NumOutputs = 2;

typedef char err_acc;

// Input -> Hidden layer 'weights'
bool W1[NumHidden][NumInputs];

// Hidden -> Output layer 'weights'
bool W2[NumOutputs][NumHidden];

// Input activations
bool Input[NumInputs];

// Hidden layer activations
bool Hidden[NumHidden];

// Output activations
bool Output[NumOutputs];

// Error gradient accumulators
err_acc E1[NumHidden][NumInputs];
err_acc E2[NumOutputs][NumHidden];

bool nor(bool const a, bool const b) {
	return !(a || b);
}

// Predict
void forward() {

	Hidden[0] = false;
	Hidden[1] = true;
	for (int i = 2; i < NumHidden; i++) {
		bool out = true;
		for (int j = 0; j < NumInputs; j++) {
			if (W1[i][j] ^ Input[j]) {
				out = false;
				break;
			}
		}
		Hidden[i] = out;
	}

	for (int o = 0; o < NumOutputs; o++) {
		bool out = true;
		for (int i = 0; i < NumHidden; i++) {
			if (W2[o][i] ^ Hidden[i]) {
				out = false;
				break;
			}
		}
		Output[o] = out;
	}
}


void incErr(err_acc & w) {
	if (w < 3) {
		w++;
	}
}


void decErr(err_acc & w) {
	if (w > -3) {
		w--;
	}
}



// Learn
int backward(bool expected[NumOutputs]) {

	if (Output == expected) {
		bool differs = false;
		for (int i = 0; i < NumOutputs; i++) {
			if (expected[i] != Output[i]) {
				differs = true;
				break;
			}
		}
		if (!differs) {
			return 0;
		}
	}

	int numUpdates = 0;

	int grad[NumHidden];
	for (int i = 0; i < NumHidden; i++) {
		grad[i] = 0;
	}

	for (int o = 0; o < NumOutputs; o++) {
		for (int i = 0; i < NumHidden; i++) {
			if ((W2[o][i] ^ Hidden[i]) != expected[o]) {
				grad[i]++;;
				incErr(E2[o][i]);
				if ((rand() % 4) < E2[o][i]) {
					W2[o][i] = !W2[o][i];
					E2[o][i] = 0;
					numUpdates++;
				}
			}
			else {
				decErr(E2[o][i]);
			}
		}
	}

	for (int i = 0; i < NumInputs; i++) {
		for (int j = 0; j < NumHidden; j++) {
			if (grad[j] >= NumOutputs / 2) {
				if (!(Input[i] ^ Hidden[j])) {
					incErr(E1[j][i]);
					if ((rand() % 4) < E1[j][i]) {
						W1[j][i] = !W1[j][i];
						E1[j][i] = 0;
						numUpdates++;
					}
				}
				else {
					decErr(E1[j][i]);
				}
			}
		}
	}

	return numUpdates;
}


// Learn
int backward_2(bool expected[NumOutputs]) {

	if (Output == expected) {
		bool differs = false;
		for (int i = 0; i < NumOutputs; i++) {
			if (expected[i] != Output[i]) {
				differs = true;
				break;
			}
		}
		if (!differs) {
			return 0;
		}
	}

	int numUpdates = 0;

	int grad[NumHidden];
	for (int i = 0; i < NumHidden; i++) {
		grad[i] = 0;
	}

	for (int o = 0; o < NumOutputs; o++) {
		for (int i = 0; i < NumHidden; i++) {
			if ((nor(W2[o][i], Hidden[i])) != expected[o]) {
				grad[i]++;
				incErr(E2[o][i]);
				if ((rand() % 4) < E2[o][i]) {
					W2[o][i] = !W2[o][i];
					E2[o][i] = 0;
					numUpdates++;
				}
			}
			else {
				decErr(E2[o][i]);
				grad[i]--;
			}
		}
	}

	for (int i = 0; i < NumInputs; i++) {
		for (int j = 0; j < NumHidden; j++) {
			if ((rand() % NumOutputs) < grad[j]) {
				if (nor(Input[i], Hidden[j])) {
					incErr(E1[j][i]);
					if ((rand() % 4) < E1[j][i]) {
						W1[j][i] = !W1[j][i];
						E1[j][i] = 0;
						numUpdates++;
					}
				}
				else {
					decErr(E1[j][i]);
				}
			}
		}
	}

	return numUpdates;
}


// Learn
int backward_3(bool expected[NumOutputs]) {

	if (Output == expected) {
		bool differs = false;
		for (int i = 0; i < NumOutputs; i++) {
			if (expected[i] != Output[i]) {
				differs = true;
				break;
			}
		}
		if (!differs) {
			return 0;
		}
	}

	int numUpdates = 0;

	for (int o = 0; o < NumOutputs; o++) {
		for (int i = 0; i < NumHidden; i++) {
			if ((rand() % 1000) < 500) {
				W2[o][i] = !W2[o][i];
				numUpdates++;
			}
		}
	}

	for (int i = 0; i < NumInputs; i++) {
		for (int j = 0; j < NumHidden; j++) {
			if ((rand() % 1000) < 500) {
				W1[j][i] = !W1[j][i];
				numUpdates++;
			}
		}
	}

	return numUpdates;
}


// Reset data
void init() {
	for (int i = 0; i < NumHidden; i++) {
		for (int j = 0; j < NumInputs; j++) {
			W1[i][j] = (rand() & 1) != 0;
			E1[i][j] = 0;
		}
		for (int j = 0; j < NumOutputs; j++) {
			W2[j][i] = (rand() & 1) != 0;
			E2[j][i] = 0;
		}
		Hidden[i] = false;
	}

	for (int i = 0; i < NumOutputs; i++) {
		Output[i] = false;
	}
}


int train() {

	bool const noisy = false;

	init();

	float avgA = 0.0f;

	for (int i = 0; i < 20000; i++) {

		Input[0] = (i & 1) != 0;
		Input[1] = (i & 2) != 0;
		Input[2] = false; // (i & 3) == 0;
		Input[3] = true; // (i & 3) == 0;
		bool ex = Input[0] ^ Input[1];

		// Add noise
		if (noisy) {
			for (int n = 0; n < NumInputs; n++) {
				if ((rand() % 100) < 15) {
					Input[n] = !Input[n];
				}
			}
		}

		forward();
		bool exps[NumOutputs]{ex, !ex};
		int const numUpdates = backward_3(exps);

		float const kI = 0.99f;
		avgA = kI * avgA + (Output[0] == ex /*&& Output[1] == !ex*/ ? 1.0f - kI : 0.0f);

		if ((i % 89) == 0) {
			std::cout << (i + 1) << " I:" << Input[0] << " " << Input[1] << " " << Input[2] << " O:" << Output[0] << " " << Output[1];

			if (Output[0] == ex && Output[1] == !ex) {
				std::cout << " +";
			}
			else {
				std::cout << "  ";
			}

			std::cout << " a: " << int(0.5f + avgA * 100.0f) << " nu: " << numUpdates << std::endl;
		}
	}

	return 0;
}


#include <fstream>
#include <string>
#include <vector>
#include <random>

static int const NumAttribs = 48;

struct Sample {
	bool input[NumAttribs];
	bool output;
};

std::vector<Sample> trainingSet;
std::vector<Sample> testSet;

int CountSpam[NumAttribs];

void loadTrainset() {

	std::default_random_engine generator;
	
	std::ifstream f("C:\\Users\\thomas.busser\\Downloads\\spambase.data");
	while (!f.eof()) {
		std::string line;
		f >> line;
		if (line.empty()) {
			break;
		}
		size_t lastPos = 0;
		Sample smp;
		for (int i = 0; i < NumAttribs; i++) {
			size_t p = line.find(',', lastPos);
			std::string feature = line.substr(lastPos, p - lastPos);
			smp.input[i] = std::stof(feature) > 1e-6;
			lastPos = p + 1;
		}

		smp.output = line[line.length() - 1] == '1';		

		if ((generator() % 10) == 0) {
			trainingSet.push_back(smp);
		} 
		else {
			testSet.push_back(smp);
		}
	}

	for (int i = 0; i < NumAttribs; i++) {
		CountSpam[i] = 0;
	}	
}

void trainSpam() {

	for (auto const & s : trainingSet) {
		for (int i = 0; i < NumAttribs; i++) {
			if (s.output) {
				if (s.input[i]) {
					CountSpam[i]++;
				}
				else {
					CountSpam[i]--;
				}
			}
		}
	}
}


void testSpam(float thresholdRatio, std::ofstream & fOut) {
	int threshold = int(float(trainingSet.size()) * thresholdRatio + 0.5f);
	int score = 0;
	int correct = 0;

	int confusion[2][2];
	confusion[0][0] = confusion[0][1] = confusion[1][0] = confusion[1][1] = 0;

	for (auto const & s : testSet) {

		int cntSpam = 0;
		for (int i = 0; i < NumAttribs; i++) {
			if (s.input[i]) {
				cntSpam += CountSpam[i];
			}
		}

		bool const pred = cntSpam > threshold;
		if (pred == s.output) {
			correct++;
			if (s.output) {
				score += 1;
			}
			else {
				score += 1;
			}
		} 
		else {
			if (s.output) {
				score -= 1;
			}
			else {
				score -= 1;
			}
		}

		confusion[s.output ? 1 : 0][pred ? 1 : 0]++;
	}

	size_t const nt = testSet.size();
	float const pct_correct = 100.0f * float(correct) / float(nt);
	std::cout << " score: " << score << "  % correct: " << pct_correct << std::endl;
	//std::cout << score << "," << confusion[0][1] << "," << confusion[1][0] << std::endl;

	fOut << score << "," << confusion[0][1] << "," << confusion[1][0] << std::endl;

}

/*
    a b    w1a=0 w1b=0
	0 0     
	0 1
	1 0
	1 1      1

*/

int _tmain(int argc, _TCHAR* argv[])
{
	// 86	1573
	// 357	967

	loadTrainset();
	trainSpam();

	std::ofstream fOut("C:\\Users\\thomas.busser\\Documents\\is.csv");
	fOut << "i,s,fp,fn,=C2*$H$1+D2*(1-$H$1)" << std::endl;

	for (float i = -5.0f; i < 2.0f; i += 0.01f) {
		std::cout << i << ",";
		fOut << i << ",";
		testSpam(i, fOut);
	}

	//train();
	return 0;
}

