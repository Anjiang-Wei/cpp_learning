#include <iostream>

class Node {
 public:
  int a;

  virtual void hi() {
    std::cout << "hi() from Node\n";
    return;
  }
  virtual void two_hi() {
    std::cout << "two_hi() from Node\n";
    return;
  }
  void gogo() {
    std::cout << "gogo() from Node\n";
    return;
  }
  virtual void hi() const { std::cout << "hi() from Node\n"; };
};

class Expr : public Node {
 public:
  int b;
  void hi() {
    std::cout << "hi() from Expr\n";
    return; }
  virtual void two() {
    std::cout << "two() from Expr\n";
    return; }
  void gogo() {
    std::cout << "gogo() from Expr\n";
    hi();
    Node::hi();
    return; }
  void hi() const { std::cout << "hi() from Expr\n"; };
};

int main() {
  Expr tmp;
  tmp.gogo();
  return 0;
}
