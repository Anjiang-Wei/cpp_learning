#include<iostream>
#include<typeinfo>
using std::cout;
using std::endl;


class A{
	public:
		A(){}
};

class B final : public A{};

int main(){
	A a;
	B b;
	A *ptr_a = &a;
	A *ptr_b = &b;
//	B *ptr_c = &a;
	B *ptr_d = &b;

	cout<<typeid(a).name()<<endl;
	cout<<typeid(b).name()<<endl;
	cout<<typeid(ptr_a).name()<<endl;
	cout<<typeid(*ptr_a).name()<<endl;
	cout<<typeid(ptr_b).name()<<endl;
	cout<<typeid(*ptr_b).name()<<endl;
//	cout<<typeid(ptr_c).name()<<endl;
//	cout<<typeid(*ptr_c).name()<<endl;
	cout<<typeid(ptr_d).name()<<endl;
	cout<<typeid(*ptr_d).name()<<endl;

	return 0;
}
