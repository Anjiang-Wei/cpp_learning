#include <iostream>
#include <memory>
#include <string>
#include <vector>
using namespace std;

// 前向声明，以便Observer能够使用这个类型
class Observable;

// 观察者类
class Observer {
 public:
  Observer(const char* str);
  void update();
  void observe(shared_ptr<Observable> so, shared_ptr<Observer> me);
  virtual ~Observer();

 private:
  string m_info;
  shared_ptr<Observable> m_observable;
};

// 被观察的物体类
class Observable {
 public:
  void register_(shared_ptr<Observer> ob);
  void notify();
  ~Observable();

 private:
  vector<shared_ptr<Observer> > m_observers;
  typedef vector<shared_ptr<Observer> >::iterator Iterator;
};

int main() {
  // 假设我们必须用到shared_ptr来管理对象
  shared_ptr<Observable> p(new Observable());

  // 用花括号创建一个局部作用域
  {
    /*
    这三个局部shared_ptr对象，在离开作用域之后就会被销毁。
    由于还有一份被Observable对象引用了，还是不会马上析构。
    */
    shared_ptr<Observer> o1(new Observer("hello"));
    shared_ptr<Observer> o2(new Observer("hi"));
    shared_ptr<Observer> o3(new Observer("how are you"));

    o1->observe(p, o1);
    o2->observe(p, o2);
    o3->observe(p, o3);

    p->notify();
  }

  cout << "\nget out now...\n" << endl;
  p->notify();
  cout << "Observable's use_count is: " << p.use_count() << endl;

  return 0;
}

// Observable的接口实现
void Observable::register_(shared_ptr<Observer> ob) {
  m_observers.push_back(ob);
}

void Observable::notify() {
  Iterator it = m_observers.begin();
  while (it != m_observers.end()) {
    cout << "notify, use_count = " << it->use_count() << endl;
    (*it)->update();
    ++it;
  }
}

Observable::~Observable() { cout << "~Observable()..." << endl; }

// Observer的接口实现
Observer::Observer(const char* str) : m_info(str) {}

void Observer::update() { cout << "update: " << m_info << endl; }

void Observer::observe(shared_ptr<Observable> so, shared_ptr<Observer> me) {
  so->register_(me);
  m_observable = so;
}

Observer::~Observer() { cout << "~Observer(): " << m_info << endl; }
