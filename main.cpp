#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <cstdlib>
using namespace std;

/*
 * I0(01)   H0(0)
 *                  O
 * I1(23)   H1(1)
 */

#define speed 0.5
#define maxEpoch 100

double fRand(double fMin, double fMax)
{
    double f = (double)rand() / RAND_MAX;
    return fMin + f * (fMax - fMin);
} //double random generator
double sigmoid(double x){
    return 1.0/(1+ exp(-x));
}
double mse(double *ideal, double *actual, int n){
    double s=0;
    for(int i=0;i<n;i++){
        s+=pow((ideal[i]-actual[i]),2);
    }
    return s/n;
}
double ideal_xor(double x,double y){
    if((x!=0 and y==0) or (y!=0 and x==0)){
        return 1.0;
    }else{
        return 0.0;
    }
}
double diff_sigm(double x){
    return (1-x)/x;
}

void network(double in1, double in2,double *input, double **hidden, double *syn1, double *syn2, double *output, double *nw){
    double data[2];
    cout<<"Type a,b(0/1 each)"<<endl;
    //start iteration
    input[0] = in1;
    input[1] = in2;
    //H0 input(0) then H1output
    hidden[0][0] = input[0]*syn1[0] + input[1]*syn1[1];
    hidden[0][1] = sigmoid(hidden[0][0]);
    //also for H1
    hidden[1][0] = input[0]*syn1[2] + input[1]*syn1[3];
    hidden[1][1] = sigmoid(hidden[1][0]);
    //now O neuron
    output[0] = hidden[0][1]*syn2[0]+hidden[1][1]*syn2[1];
    output[1] = sigmoid(output[0]);

    nw[1] = pow((ideal_xor(input[0],input[1]) - output[1]),2);
    nw[0] = output[1];

}


int main() {
    srand(time(NULL));
    double input[2]; //input neurons
    double hidden[2][2]; //hidden layer
    double output[2]; //oup neuron
    double out_ideal; //ideal answer (XOR)
    double error; //error
    double ans_bp; // answer
    double nw[2]={0.0,0.0}; // answer + error (for exporting from network())

    auto *syn1 = new double[4]();
    auto *syn2 = new double[2]();
    int trainSet[4][3]= {
            {0,0,0},
            {0,1,1},
            {1,0,1},
            {1,1,0}
    };

    //initialize start weights
    for(int i=0;i<4;i++){
        syn1[i]= fRand(0.0,1.0);
        syn2[i%2] = fRand(0.0,1.0);
    }

    //custom data
    cout<<"Type a,b(0/1 each)"<<endl;
    double a_i,b_i;
    cin>>a_i,b_i;
    //start iteration
    input[0] = a_i;
    input[1] = b_i;
    //H0 input(0) then H1output
    hidden[0][0] = input[0]*syn1[0] + input[1]*syn1[1];
    hidden[0][1] = sigmoid(hidden[0][0]);
    //also for H1
    hidden[1][0] = input[0]*syn1[2] + input[1]*syn1[3];
    hidden[1][1] = sigmoid(hidden[1][0]);
    //now O neuron
    output[0] = hidden[0][1]*syn2[0]+hidden[1][1]*syn2[1];
    output[1] = sigmoid(output[0]);

    error = pow((ideal_xor(input[0],input[1]) - output[1]),2);

    cout<<"Result: "<<output[1]<<endl<<"Error: "<<error<<endl;

    //back propagation

    double delta_out;
    double delta_hidden[2];
    double delta_inp[2];

    for(int i=0;i<maxEpoch;i++){ //epoch
        for(int j=0;j<4;j++){    //run through train set
            network(trainSet[i][0], trainSet[i][1], input, reinterpret_cast<double **>(hidden), syn1, syn2, output, nw);
            ans_bp =nw[0];
            error = nw[1];
            delta_out =(trainSet[i][2]-ans_bp)* diff_sigm(ans_bp);
            delta_hidden[0] = diff_sigm(hidden[0][0]) //process
        }
    }









    cout<<"End!\n";
}
