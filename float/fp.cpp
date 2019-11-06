/*
 *大数加小数的时候，小数一直加不上去，持续几个之后就出问题了
 *
 *
 *
 */
#include<iostream>
using std::cout;
using std::endl;


int main(){
	
	float a = 3.f;
	float b = 299792458.f;

	float c = 299792458.f;

	cout.precision(10);

	cout<<"a = "<<a<<endl;
	cout<<"b = "<<b<<endl;
	cout<<"c = "<<c<<endl;

	cout<<"a+b = "<<a+b<<endl;
	cout<<"b+c = "<<c+b<<endl;

	return 0;
}
