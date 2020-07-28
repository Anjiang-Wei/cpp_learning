#include <iostream>

class A {
 public:
  void print(void) = delete;
};

class B : public A {
 public:
  void print(void) { std::cout << "Print from B" << std::endl; }
};

void visit(B *);
void visit(A *) { std::cout << "Visit A" << std::endl; }

void visit(B *) = delete;

int main() {
  A tmp;
  B tmp1;

  //   visit(&tmp);
  visit(&tmp);
  return 0;
}