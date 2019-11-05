/*
 *shared_pointer不要使用普通指针初始化多个shared_pointer，会造成二次销毁
 *
 */

#include<iostream>
#include<memory>
using std::cout;
using std::endl;

class A{
	public:
		int *a;
		A(int n=10){
			cout<<"Constructor called"<<endl;
			a = new int[10];
		}
		~A(){
			cout<<"Destructor called"<<endl;
			if(a!=nullptr){
				delete []a;
			}
		}
};

int main(){
	
	auto *p = new A(1);
	std::shared_ptr<A> p1(p);
	std::shared_ptr<A> p2(p);

	cout<<p1.use_count()<<endl;
	cout<<p2.use_count()<<endl;

	cout<<p1.unique()<<endl;
	cout<<p2.unique()<<endl;

	return 0;
}
