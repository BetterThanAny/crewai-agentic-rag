# Python 高级特性

## 装饰器（Decorator）

装饰器是 Python 中一种强大的语法特性，允许在不修改原函数代码的情况下扩展函数的功能。
装饰器本质上是一个接受函数作为参数并返回新函数的高阶函数。

### 基本用法

```python
def my_decorator(func):
    def wrapper(*args, **kwargs):
        print("函数执行前")
        result = func(*args, **kwargs)
        print("函数执行后")
        return result
    return wrapper

@my_decorator
def say_hello():
    print("Hello!")
```

使用 `@my_decorator` 语法糖等价于 `say_hello = my_decorator(say_hello)`。

## 生成器（Generator）

生成器是一种特殊的迭代器，使用 `yield` 关键字来产生值。
与普通函数不同，生成器函数在每次调用 `yield` 时会暂停执行并保存状态。

### 生成器表达式

```python
squares = (x**2 for x in range(10))
```

生成器的主要优势是惰性求值，可以高效处理大数据集而不需要一次性加载所有数据到内存。

## 上下文管理器（Context Manager）

上下文管理器通过 `with` 语句实现资源的自动管理。
最常见的用法是文件操作：

```python
with open('file.txt', 'r') as f:
    content = f.read()
```

可以通过实现 `__enter__` 和 `__exit__` 方法或使用 `contextlib.contextmanager` 装饰器来创建自定义上下文管理器。

## 元类（Metaclass）

元类是创建类的类。在 Python 中，类本身也是对象，元类定义了类的创建行为。
`type` 是 Python 中最常见的元类。

```python
class MyMeta(type):
    def __new__(mcs, name, bases, namespace):
        # 自定义类创建逻辑
        return super().__new__(mcs, name, bases, namespace)
```
