#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <cstdlib>
using namespace std;
// IDEA: https://habr.com/ru/post/313216/
/*
 * I0(01)   H0(0)
 *                  O
 * I1(23)   H1(1)
 */

#define maxEpoch 10
#define alpha 0.3
#define trainSpeed 0.7

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

void network(double in1, double in2,double input[2], double hidden[2][2], double syn1[4], double syn2[2], double output[2], double *nw){
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
/*void network(double in1, double in2,double *input, double hidden[2][2], double *syn1, double *syn2, double *output, double *nw){
    cout<<"Type a,b(0/1 each)"<<endl;
    //start iteration
    input[0] = in1;
    input[1] = in2;
    //H0 input(0) then H1output
    hidden[0*2+0] = input[0]*syn1[0] + input[1]*syn1[1];
    hidden[0*2+1] = sigmoid(hidden[0*2+0]);
    //also for H1
    hidden[1*2+0] = input[0]*syn1[2] + input[1]*syn1[3];
    hidden[1*2+1] = sigmoid(hidden[1*2+0]);
    //now O neuron
    output[0] = hidden[0*2+1]*syn2[0]+hidden[1*2+1]*syn2[1];
    output[1] = sigmoid(output[0]);

    nw[1] = pow((ideal_xor(input[0],input[1]) - output[1]),2);
    nw[0] = output[1];

}*/


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

    cout<<"Result(random weights): "<<output[1]<<endl<<"Error: "<<error<<endl;

    cout<<"Training...\n";

    //back propagation

    double delta_out;
    double delta_hidden[2];
    double delta_inp[2];
    double deltaw[6]; //delta weight for all 6 neurons
    double deltaw_previous[6] = {0.0,0.0,0.0,0.0,0.0,0.0};

    for(int i=0;i<maxEpoch;i++){ //epoch
        cout<<"Epoch: "<<i<<endl;
        for(int j=0;j<4;j++){    //run through train set
            network(trainSet[i][0], trainSet[i][1], input, hidden, syn1, syn2, output, nw);
            ans_bp =nw[0];
            error = nw[1];
            delta_out =(trainSet[i][2]-ans_bp)* diff_sigm(ans_bp);  //delta output

            delta_hidden[0] = diff_sigm(hidden[0][0]) * (delta_out*syn2[0]); //delta for H0
            deltaw[0+4] = trainSpeed*(delta_out*hidden[0][1])+ alpha*deltaw_previous[4]; //delta weight for H0-O
            deltaw_previous[4] = deltaw[4];
            syn2[0]+=deltaw[4]; //changing weight

            delta_hidden[1] = diff_sigm(hidden[1][0]) * (delta_out*syn2[1]); //delta for H1
            deltaw[0+5] = trainSpeed*(delta_out*hidden[1][1])+ alpha*deltaw_previous[5]; //delta weight for H1-O
            deltaw_previous[5] = deltaw[5];
            syn2[1]+=deltaw[5]; //changing weight

            //now need to do the same for input layer
            deltaw[0] = trainSpeed*(delta_hidden[0]*trainSet[i][0])+ alpha*deltaw_previous[0]; //delta weight for I0-H0
            deltaw_previous[0] = deltaw[0];
            syn1[0]+=deltaw[0]; //changing weight

            deltaw[1] = trainSpeed*(delta_hidden[1]*trainSet[i][0])+ alpha*deltaw_previous[1]; //delta weight for I0-H1
            deltaw_previous[1] = deltaw[1];
            syn1[1]+=deltaw[1]; //changing weight

            deltaw[2] = trainSpeed*(delta_hidden[0]*trainSet[i][1])+ alpha*deltaw_previous[2]; //delta weight for I1-H0
            deltaw_previous[2] = deltaw[2];
            syn1[2]+=deltaw[2]; //changing weight

            deltaw[3] = trainSpeed*(delta_hidden[1]*trainSet[i][1])+ alpha*deltaw_previous[3]; //delta weight for I1-H1
            deltaw_previous[3] = deltaw[3];
            syn1[3]+=deltaw[3]; //changing weight

            cout<<error<<" ";
        }
        cout<<endl;
    }
    cout<<"Training completed!"<<endl;
    /*while (true){
        cout<<"Type a,b(2 2 to exit)"<<endl;
        double testa,testb;
        cin>>testa>>testb;
        if(testa<=1 and testb<=1){
            //do
            network(testa, testb, input, reinterpret_cast<double **>(hidden), syn1, syn2, output, nw);
            cout<<"answer: "<<nw[0]<<endl;
            cout<<"error: "<<nw[1]<<endl;
        }else{
            break;
        }
    }*/

    cout<<"End!\n";
}
