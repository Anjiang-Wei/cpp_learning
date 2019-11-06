/*
 *Kahan的思路是我先把误差存起来，误差与下一个small数先做计算
 *
 *
 */
#include<iostream>
#include<algorithm>

using std::cout;
using std::endl;


float Kahan(float *a, int size){
	
	float sum = 0.f;
	float err = 0.f;

//	cout.precision(10);

	for (int i=0; i<size; ++i){
//		cout<<err<<' ';

		float tmp = a[i] - err;
		float sum1 = tmp + sum;
		float err = (sum1 - sum) - tmp;
		sum = sum1;

//		cout<<sum1<<' '<<a[i]<<' '<<err<<' '<<sum<<endl;
	}

	return sum;
}


float raw[2] = {299792458.f, 3.f};

int main(int argc, char *argv[]){

	int size = atoi(argv[1]);

	float *a = new float[size];

	for (int i=0; i<size; ++i){
		a[i] = raw[i%2];
	}
	
	cout<<Kahan(a, size)<<endl;
	
	std::sort(a, a+size);
	cout<<Kahan(a, size)<<endl;


	delete []a;

	return 0;
}
