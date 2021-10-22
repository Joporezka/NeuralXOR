#include <iostream>
#include <ctime>
#include <cmath>
#include <vector>
#include <cstdlib>
using namespace std;
//committed from vscode.dev/GITHUB_NAME
// IDEA: https://habr.com/ru/post/313216/
/*
 * I0(01)   H0(0)
 *                  O
 * I1(23)   H1(1)
 */

//full restructuring

#define maxEpoch 1000
#define alpha 0.2
#define trainSpeed 0.6

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
    return (1-x)*x;
}

void network(double output_mas[], double input[],double hidden_input[], double hidden_output[], double syn1[],double o_input, double o_output){
    //H0 input(0) then H1output
    hidden_input[0] = input[0]*syn1[0] + input[1]*syn1[1];
    hidden_output[0] = sigmoid(hidden_input[0]);
    //also for H1
    hidden_input[1] = input[0]*syn1[2] + input[1]*syn1[3];
    hidden_output[1] = sigmoid(hidden_input[1]);
    //now O neuron
    o_input = hidden_output[0]*syn1[4]+hidden_output[1]*syn1[5];
    o_output = sigmoid(o_input);

    output_mas[0]=o_output;
    output_mas[1]= pow((ideal_xor(input[0],input[1]) - o_output),2);
}



int main() {
    srand(time(NULL));
    double input[2]; //input neurons
    double hidden_input[2]; //hidden layer
    double hidden_output[2];
    double o_input; //oup neuron
    double o_output;
    double out_ideal=-1.0; //ideal answer (XOR)
    double error; //error
    double ans_bp=-1.0; // answer

    double input_neural[2];
    double output_neural[2];

    double syn1[6] = {0.0,0.0,0.0,0.0,0.0,0.0}; //all synapses
    int trainSet[4][3]= {
            {0,0,0},
            {0,1,1},
            {1,0,1},
            {1,1,0}
    };

    //initialize start weights
    for(int i=0;i<6;i++){
        syn1[i]= fRand(0.0,1.0);
    }

    //custom data
    cout<<"Type a,b(0/1 each)"<<endl;
    cin>>input_neural[0]>>input_neural[1];
    //start iteration
    /*input[0] = a_i;
    input[1] = b_i;
    //H0 input(0) then H1output
    hidden_input[0] = input[0]*syn1[0] + input[1]*syn1[1];
    hidden_output[0] = sigmoid(hidden_input[0]);
    //also for H1
    hidden_input[1] = input[0]*syn1[2] + input[1]*syn1[3];
    hidden_output[1] = sigmoid(hidden_input[1]);
    //now O neuron
    o_input = hidden_output[0]*syn1[4]+hidden_output[1]*syn1[5];
    o_output = sigmoid(o_input);

    error = pow((ideal_xor(input[0],input[1]) - output[1]),2);

    cout<<"Result(random weights): "<<output[1]<<endl<<"Error: "<<error<<endl;*/

    network(output_neural,input_neural,hidden_input,hidden_output,syn1,o_input,o_output);

    cout<<"Training...\n";

    //back propagation

    double delta_out = 0 ;
    double delta_hidden[2]={0.0,0.0};
    double delta_inp[2];
    double deltaw[6] = {0.0,0.0,0.0,0.0,0.0,0.0}; //delta weight for all 6 neurons
    double deltaw_previous[6] = {0.0,0.0,0.0,0.0,0.0,0.0};

    for(int i=0;i<maxEpoch;i++){ //epoch
        cout<<"Epoch: "<<i<<endl;
        for(int j=0;j<4;j++){    //run through train set

            //run network
            input_neural[0]=trainSet[i][0];
            input_neural[1]=trainSet[i][1];
            network(output_neural,input_neural,hidden_input,hidden_output,syn1,o_input,o_output);

            ans_bp =output_neural[0];
            error = output_neural[1];
            delta_out =(trainSet[i][2]-ans_bp)* diff_sigm(ans_bp);  //delta output

            delta_hidden[0] = diff_sigm(hidden_input[0]) * (delta_out*syn1[4]); //delta for H0
            deltaw[0+4] = trainSpeed*(delta_out*hidden_output[0])+ alpha*deltaw_previous[4]; //delta weight for H0-O
            deltaw_previous[4] = deltaw[4];
            syn1[4]+=deltaw[4]; //changing weight

            delta_hidden[1] = diff_sigm(hidden_input[1]) * (delta_out*syn1[5]); //delta for H1
            deltaw[0+5] = trainSpeed*(delta_out*hidden_output[1])+ alpha*deltaw_previous[5]; //delta weight for H1-O
            deltaw_previous[5] = deltaw[5];
            syn1[5]+=deltaw[5]; //changing weight

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

            cout<<" "<<error<<" ";
        }
        cout<<endl;
    }
    cout<<"Training completed!"<<endl;
    while (true){
        cout<<"Type a,b(2 2 to exit)"<<endl;
        double testa,testb;
        cin>>testa>>testb;
        if(testa<=1 and testb<=1){
            //do
            input_neural[0]=testa;
            input_neural[1]=testb;
            network(output_neural,input_neural,hidden_input,hidden_output,syn1,o_input,o_output);
            cout<<"answer: "<<output_neural[0]<<endl;
            cout<<"error: "<<output_neural[1]<<endl;
        }else{
            break;
        }
    }

    cout<<"End!\n";
}
