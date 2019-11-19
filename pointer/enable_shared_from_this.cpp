/*
 *1.为何不直接传递this指针

     
 使用智能指针的初衷就是为了方便资源管理，如果在某些地方使用智能指针，某些地方使用原始指针，很容易破坏智能指针的语义，从而产生各种错误。

2.可以直接传递share_ptr<this>么？

 答案是不能，因为这样会造成2个非共享的share_ptr指向同一个对象，未增加引用计数导对象被析构两次。例如：

 */
#include <iostream>
#include <memory>

class Bad {
   public:
    std::shared_ptr<Bad> getptr() { return std::shared_ptr<Bad>(this); }

    ~Bad() { std::cout << "Bad::~Bad() called\n"; }
};

class Good : std::enable_shared_from_this<Good>{
   public:
    Good() {}
    std::shared_ptr<Good> getptr() { return shared_from_this(); }
    ~Good() { std::cout << "Good::~Good() called\n"; }
};
int main() {
    /*
    std::shared_ptr<Bad> bp1(new Bad());
    std::shared_ptr<Bad> bp2 = bp1->getptr();
    std::cout << "bp1.use_count(): " << bp1.use_count() << std::endl;
    std::cout << "bp2.use_count(): " << bp2.use_count() << std::endl;
	*/

    std::shared_ptr<Good> bp3 = std::make_shared<Good>();
    // std::shared_ptr<Good> bp4 = bp3->getptr();
    std::shared_ptr<Good> bp4 = bp3->shared_from_this();
    std::cout << "bp3.use_count(): " << bp3.use_count() << std::endl;
    std::cout << "bp4.use_count(): " << bp4.use_count() << std::endl;
    return 0;
}
