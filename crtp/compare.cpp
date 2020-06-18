#include <iostream>
#include <memory>
#include <vector>
class Animal {
 public:
  void virtual jump() const = 0;
};

class Zoo {
 public:
  std::vector<std::shared_ptr<Animal>> ana;
  void play() {
    for (auto it : ana) {
      it->jump();
    }
  }
};

class Tiger : public Animal {
 public:
  void jump() const override final { std::cout << "Tiger jumps" << std::endl; }
};

class Mouse : public Animal {
 public:
  void jump() const override final { std::cout << "Mouse jumps" << std::endl; }
};

int main() {
  Zoo my_zoo;
  my_zoo.ana.push_back(std::make_shared<Tiger>());
  my_zoo.ana.push_back(std::make_shared<Mouse>());
  my_zoo.play();
  return 0;
}
