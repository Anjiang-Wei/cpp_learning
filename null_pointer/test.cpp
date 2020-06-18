#include <memory>
#include <iostream>

class A : public std::enable_shared_from_this<A> {
   public:
    explicit A(int _a) : a(new int[_a]) {}
    void get_value() { std::cout << "a=" << a << std::endl; }
    std::shared_ptr<A> get_obj() { return shared_from_this(); }
    ~A() { delete[] a; }

   private:
    int *a;
};
using APtr = std::shared_ptr<A>;

int main() {
    A tmp(4);
    std::shared_ptr<A> p1 = std::make_shared<A>(10);
    APtr p2 = p1->get_obj();
    return 0;
}
