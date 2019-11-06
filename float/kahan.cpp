/*
 *Kahan的思路是我先把误差存起来，误差与下一个small数先做计算
 *
 *
 */
#include<iostream>

using std::cout;
using std::endl;


float Kahan(float *a, int size){
	
	float sum = a[0];
	float err = 0.f;

	for (int i=0; i<size; ++i){
		cout<<sum<<' ';
		float tmp = a[i] - err;
		float sum1 = tmp + sum;
		float err = (sum1 - sum) - tmp;
		sum = sum1;
	}

	return sum;
}


int main(int argc, char *argv[]){

//	int size = atoi(argv[1]);

//	float *a = new float[size];
	
//	Kahan(a, size);
//
    float a = 3;           // (gdb) p/f a   = 3
	float b = 299792458;   // (gdb) p/f b   = 299792448

	float c = a + b;
	float err = (c - b) -a;
	cout.precision(10);

	cout<<a<<' '<<b<<' '<<c<<' '<<err<<endl;


//	delete []a;

	return 0;
}
