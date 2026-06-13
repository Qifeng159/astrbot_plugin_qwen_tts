# astrbot_plugin_qwen_tts

基于 Qwen3-TTS 声音克隆模型，将文本以克隆音色转为语音消息的 AstrBot 插件。

## 前置条件

- 本地运行 [Qwen3-TTS WebUI](https://github.com/licyk/qwen_tts_webui)，默认端口 7860
- 已下载声音克隆模型（Base 系列，如 `Qwen/Qwen3-TTS-12Hz-1.7B-Base`）
- 准备一段参考音频（WAV 格式，10-30 秒干净人声）

## 安装

将插件目录放入 `data/plugins/` 下，重启 AstrBot 即可。

## 配置

在 WebUI 插件配置页面修改以下参数：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `tts_api_url` | 声音克隆 API 地址 | `http://127.0.0.1:7860/qwenapi/v1/voice-clone` |
| `model_name` | 模型名称 | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` |
| `ref_audio_path` | 参考音频路径 | `C:\AstrBot\voive-sample\ref_voice.wav` |
| `ref_text` | 参考音频对应文本（可选） | 空 |
| `default_language` | 默认合成语言 | `chinese` |
| `request_timeout` | API 超时（秒） | `120` |
| `whitelist` | 用户白名单，空则全员可用 | `[]` |
| `voice_probability` | 随机语音回复概率（0-1） | `0` |
| `emotion_mode` | 情绪模式开关 | `false` |
| `emotion_detect_method` | 情绪检测方式：`keyword` / `llm` | `keyword` |
| `emotion_llm_provider` | LLM 情绪检测的提供商 ID，留空跟随会话 | 空 |
| `show_tts_hint` | 显示合成进度提示和标签 | `true` |

## 指令

| 指令 | 说明 |
|------|------|
| `/tts <文本>` | 用克隆音色将文本转为语音 |
| `/ttslang` | 查看支持的语言列表 |
| `/ttsmodels` | 查看可用模型列表 |
| `/ttsconfig` | 查看当前配置和状态 |

## 情绪模式

开启 `emotion_mode` 后在插件目录下创建 `emotion_audio/` 文件夹，放入各情绪的参考音频（WAV）：

```
emotion_audio/
  neutral.wav      # 默认
  happy.wav        # 开心
  sad.wav          # 难过
  angry.wav        # 愤怒
  surprised.wav    # 惊讶
```

情绪检测支持两种方式：
- **关键词匹配**（默认）：根据消息中的关键词判断情绪
- **LLM 检测**：调用指定提供商判断，更准确但有一次 API 开销

## 随机语音回复

设置 `voice_probability` > 0 后，插件会监听 AstrBot 的聊天回复，按概率将回复文本以克隆语音发送。例如设为 `0.3` 表示 30% 的回复附带克隆语音。

语音合成前会自动清洗中文/英文括号及其内容，避免念出注释文本。

## 许可证

MIT