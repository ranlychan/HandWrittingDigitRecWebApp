# HandWrittingDigitRecWebApp
A Flask project that recognize handwritten digit

# # 系统概要
手写数字识别画板系统，按照MVC原则开发，主要由两部分组成：交互界面（视图View）部分是传统的HTML +CSS+JS网页（这同样也是一种遵循MVC开发方式）；手写数字识别部分（模型Model）是使用Python开发的深度学习的模型；两者间通过基于Flask框架开发的Python Web服务连接（控制Control），具体而言，两者间手写数字识别部分功能的信息传输方式为：HTTP请求收发JSON格式的数据。

> https://ranlychan.top/archives/402.html
