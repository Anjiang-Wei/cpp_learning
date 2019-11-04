/*
 * dynamic_cast用于不确定运行时是什么类型
 *1. downcast(cast to derived class)dynamic_cast必须要求拥有多态机制才可以，不然无法正确编译
 *
 *
 *
 *
 *
 */

#include<iostream>
using namespace std;

class Base{
	public:
		Base(){
			cout<<"Base Constructor"<<endl;
		}
		virtual void func(){
			cout<<"func called from base"<<endl;
		}
};

class Derived final : public Base{
	public:
		Derived(){
			cout<<"Derived Constructor"<<endl;
		}
		void func(){
			cout<<"func called from Derived"<<endl;
		}
};

int main(){

/*  downcast
	Derived d;
	Base *b = &d;
	Derived * tmp = dynamic_cast<Derived *>(b);
	if(tmp!=nullptr){
		cout<<"Succeed in conversion"<<endl;
	}else{
		cout<<"Failed conversion"<<endl;
	}
*/

/*
	Derived d;
	Base * tmp = dynamic_cast<Base *>(&d);
	if(tmp!=nullptr){
		cout<<"Succeed in conversion"<<endl;
	}else{
		cout<<"Failed conversion"<<endl;
	}

	return 0;
}

