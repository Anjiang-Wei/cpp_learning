/*
 *本质上是引入了C++垃圾回收的问题，当存在循环引用的时候，其无法判断一个对象的生命周期，因此无法完成析构
 *但是普通指针的循环引用却没有问题，这是为什么？
 *
 */
#include <iostream>
#include <memory>
using namespace std;
 
class B;
class A
{
public:// 为了省去一些步骤这里 数据成员也声明为public
    //weak_ptr<B> pb;
    shared_ptr<B> pb;
    void doSomthing()
    {
//        if(pb.lock())
//        {
//
//        }
    }
 
    ~A()
    {
        cout << "kill A\n";
    }
};
 
class B
{
public:
    //weak_ptr<A> pa;
    shared_ptr<A> pa;
    ~B()
    {
        cout <<"kill B\n";
    }
};
 
int main(int argc, char** argv)
{
    shared_ptr<A> sa(new A());
    shared_ptr<B> sb(new B());
    if(sa && sb)
    {
        sa->pb=sb;
        sb->pa=sa;
    }
    cout<<"sa use count:"<<sa.use_count()<<endl;



	
	//普通指针测试一下
	class N;
	class M{
		public:
		N *a;
		~M(){
			cout<<"M destructor called"<<endl;
		}
	};
	class N{
		public:
		M *a;
		~N(){
			cout<<"N destructor called"<<endl;
		}
	};

	M tmpM;
	N tmpN;
	tmpM.a = &tmpN;
	tmpN.a = &tmpM;
    return 0;
}
