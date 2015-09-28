
#include "stdafx.h"

#include <iostream>
#include <stdlib.h> 

static int const NumHidden = 11;
static int const NumInputs = 3;
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


// Predict
void forward() {

    for (int i = 0; i < NumHidden; i++) {
        int acc = 0;
        for (int j = 0; j < NumInputs; j++) {
            if (W1[i][j] ^ Input[j]) {
                acc++;
            }
        }
        Hidden[i] = (acc > NumInputs / 2);
    }

    for (int o = 0; o < NumOutputs; o++) {
        int acc = 0;
        for (int i = 0; i < NumHidden; i++) {
            if (W2[o][i] ^ Hidden[i]) {
                acc++;
            }
        }
        Output[o] = (acc > NumHidden / 2);
    }
}


void incWeight(err_acc & w) {
    if (w < 3) {
        w++;
    }
}


void decWeight(err_acc & w) {
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
                incWeight(E2[o][i]);
                if ((rand() % 4) < E2[o][i]) {
                    W2[o][i] = !W2[o][i];
                    E2[o][i] = 0;
                    numUpdates++;
                }
            }
            else {
                decWeight(E2[o][i]);
            }
        }
    }

    for (int i = 0; i < NumInputs; i++) {
        for (int j = 0; j < NumHidden; j++) {
            if (grad[j] >= NumOutputs / 2) {
                if (!(Input[i] ^ Hidden[j])) {
                    incWeight(E1[j][i]);
                    if ((rand() % 4) < E1[j][i]) {
                        W1[j][i] = !W1[j][i];
                        E1[j][i] = 0;
                        numUpdates++;
                    }
                }
                else {
                    decWeight(E1[j][i]);
                }
            }
        }
    }

    return numUpdates;
}


// Reset data
void init() {
    for (int i = 0; i < NumHidden; i++) {
        for (int j = 0; j < NumInputs; j++) {
            W1[i][j] = false;
            E1[i][j] = 0;
        }
        for (int j = 0; j < NumOutputs; j++) {
            W2[j][i] = false;
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

    for (int i = 0; i < 2000; i++) {

        Input[0] = (i & 1) != 0;
        Input[1] = (i & 2) != 0;
        Input[2] = (i & 3) == 0;
        bool ex = Input[0] ^ (Input[1] ^ Input[2]);

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
        int const numUpdates = backward(exps);

        float const kI = 0.99f;
        avgA = kI * avgA + (Output[0] == ex /*&& Output[1] == !ex*/ ? 1.0f - kI : 0.0f);

        if ((i % 10) == 0) {
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

int _tmain(int argc, _TCHAR* argv[])
{
    train();
    return 0;
}

