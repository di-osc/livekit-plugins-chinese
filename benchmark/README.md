## 简介

测试各个插件的延迟

### 使用

- 使用start模式启动agent
```
python agent.py start
```

- 复制日志到logs目录下面，必须包含speech start -》 tts end完整链路

- 运行脚本
```
python test_latency ./logs
```