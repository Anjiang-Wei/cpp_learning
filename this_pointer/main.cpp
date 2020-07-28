#include <iostream>

class Base {
 public:
  int a{0};
  virtual void print() {
    std::cout << "Base: " << this->a << std::endl;
    this->print_();
  }
  virtual void print_() { std::cout << "Base::print_" << std::endl; }
};

class Derived : public Base {
 public:
  int b{1};
  // virtual void print() {
  //   std::cout << "Derived: " << this->a << " " << this->b << std::endl;
  //   this->print_();
  // }
  void print_() { std::cout << "Derived::print_" << std::endl; }
};

int main() {
  Base *A = new Base();
  A->print();

  Base *B = new Derived();
  B->print();
  return 0;
}