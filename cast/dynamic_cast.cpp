/*
 * dynamic_cast用于不确定运行时是什么类型
 *1. downcast(cast to derived class)dynamic_cast必须要求拥有多态机制才可以，不然无法正确编译
 *2. static_cast没有运行时开销，不过需要自己保证正确性
 *3. rtti 里面的typeid可以知道对象的类型，typeid既可以在编译时了解类型也可以在运行时了解
 *4. rtti 包含typeied和dynamic_cast，都需要虚函数
 *5. dynamic_cast运算符可以在执行期决定真正的类型。如果downcast是安全的（也就说，如果基类指针或者引用确实指向一个派生类对象）这个运算符会传回适当转型过的指针。如果downcast不安全，这个运算符会传回空指针（也就是说，基类指针或者引用没有指向一个派生类对象）。
 *6. 做上行转换时，和static_cast一样，做下行转换时dynamic_cast具有类型检查功能，比static_cast更安全
 *7. static_cast不需要rtti，因此不需要虚函数
 *8. dynamic_pointer_cast用于shared_pointer的情况 
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


	Derived d;
	Base * tmp = dynamic_cast<Base *>(&d);
	if(tmp!=nullptr){
		cout<<"Succeed in conversion"<<endl;
	}else{
		cout<<"Failed conversion"<<endl;
	}


	return 0;
}

