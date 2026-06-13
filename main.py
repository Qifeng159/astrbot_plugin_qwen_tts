"""
Qwen3-TTS 声音克隆插件

通过本地 Qwen3-TTS 的声音克隆模型，将文本以克隆音色转为语音消息。
需要在插件配置中指定参考音频文件路径（WAV 格式，10-30 秒干净人声）。

指令：
  /tts <文本>       - 用克隆音色将文本转为语音
  /ttslang          - 查看支持的语言列表
  /ttsmodels        - 查看可用模型列表
  /ttsconfig        - 查看当前配置和参考音频状态

依赖：httpx（异步 HTTP），本地 Qwen3-TTS WebUI 运行中
"""

import base64
import os
import re
import random
import tempfile
from pathlib import Path

import httpx
from astrbot.api import logger
from astrbot.api.event import filter, AstrMessageEvent, MessageChain
from astrbot.api.provider import LLMResponse
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.astr_agent_context import AstrAgentContext
from astrbot.api.star import Context, Star, register
from astrbot.api.message_components import Record, Plain

LANG_MAP = {
    "chinese": "中文",
    "english": "英语",
    "japanese": "日语",
    "korean": "韩语",
    "french": "法语",
    "german": "德语",
    "italian": "意大利语",
    "portuguese": "葡萄牙语",
    "russian": "俄语",
    "spanish": "西班牙语",
    "auto": "自动检测",
}


@register(
    "astrbot_plugin_qwen_tts",
    "Marvis",
    "基于 Qwen3-TTS 声音克隆模型，用参考音频将文本转为语音",
    "1.0.0",
    "https://github.com/user/astrbot_plugin_qwen_tts",
)
class Main(Star):
    def __init__(self, context: Context, config: dict):
        super().__init__(context)
        self.tts_api_url = config.get(
            "tts_api_url",
            "http://127.0.0.1:7860/qwenapi/v1/voice-clone",
        )
        self.model_name = config.get(
            "model_name",
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        )
        self.ref_audio_path = config.get("ref_audio_path", "").strip()
        self.ref_text = config.get("ref_text", "").strip()
        self.default_language = config.get("default_language", "chinese")
        self.request_timeout = config.get("request_timeout", 120)
        self.whitelist: list = config.get("whitelist", [])
        self.voice_probability = float(config.get("voice_probability", 0.0))
        self.emotion_mode = bool(config.get("emotion_mode", False))
        self.emotion_detect_method = config.get("emotion_detect_method", "keyword").strip()
        self.emotion_llm_provider = config.get("emotion_llm_provider", "").strip()
        self.show_tts_hint = bool(config.get("show_tts_hint", True))

        # 启动时加载参考音频
        self.ref_audio_base64: str = ""
        if self.ref_audio_path:
            self._load_ref_audio()
        else:
            logger.warning(
                "[Qwen-TTS] 未配置参考音频路径 (ref_audio_path)，"
                "请在插件配置中设置后重载插件。"
            )

        # 情绪音频
        self.emotion_audios: dict[str, str] = {}
        if self.emotion_mode:
            self._load_emotion_audios()

    def _load_ref_audio(self) -> None:
        """从配置的路径加载参考音频并编码为 Base64"""
        path = Path(self.ref_audio_path)
        if not path.exists():
            logger.error(
                f"[Qwen-TTS] 参考音频文件不存在: {self.ref_audio_path}"
            )
            return
        if not path.is_file():
            logger.error(
                f"[Qwen-TTS] 参考音频路径不是文件: {self.ref_audio_path}"
            )
            return

        try:
            file_size = path.stat().st_size
            if file_size > 5 * 1024 * 1024:  # 5MB 限制
                logger.error(
                    f"[Qwen-TTS] 参考音频文件过大 ({file_size} bytes)，请使用 5MB 以内的文件"
                )
                return

            audio_bytes = path.read_bytes()
            self.ref_audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
            logger.info(
                f"[Qwen-TTS] 参考音频加载成功: {self.ref_audio_path} "
                f"({file_size} bytes, base64 长度 {len(self.ref_audio_base64)})"
            )
        except Exception as e:
            logger.error(f"[Qwen-TTS] 读取参考音频失败: {e}")

    def _load_emotion_audios(self) -> None:
        """加载 emotion_audio 目录下的情绪参考音频"""
        plugin_dir = Path(__file__).parent
        audio_dir = plugin_dir / "emotion_audio"
        if not audio_dir.exists():
            logger.warning(f"[Qwen-TTS] 情绪音频目录不存在: {audio_dir}")
            return

        for wav_file in audio_dir.glob("*.wav"):
            emotion = wav_file.stem.lower()
            try:
                file_size = wav_file.stat().st_size
                if file_size > 5 * 1024 * 1024:
                    logger.error(f"[Qwen-TTS] 情绪音频过大，跳过: {wav_file.name} ({file_size} bytes)")
                    continue
                audio_bytes = wav_file.read_bytes()
                self.emotion_audios[emotion] = base64.b64encode(audio_bytes).decode("utf-8")
                logger.info(f"[Qwen-TTS] 情绪音频加载: {emotion} ← {wav_file.name} ({file_size} bytes)")
            except Exception as e:
                logger.error(f"[Qwen-TTS] 读取情绪音频失败 {wav_file.name}: {e}")

        if self.emotion_audios:
            logger.info(f"[Qwen-TTS] 情绪音频加载完成: {list(self.emotion_audios.keys())}")
        else:
            logger.warning("[Qwen-TTS] 未加载到任何情绪音频")

    _EMOTION_KEYWORDS: dict[str, list[str]] = {
        "angry":      ["愤怒", "生气", "可恶", "混蛋", "滚", "妈的", "气死", "火大", "恼火", "该死"],
        "happy":      ["哈哈", "开心", "高兴", "太好", "棒", "yeah", "nice", "恭喜", "快乐", "耶"],
        "sad":        ["难过", "伤心", "哭", "悲伤", "遗憾", "可惜", "唉", "难受", "呜呜", "心碎"],
        "surprised":  ["哇", "天哪", "真的吗", "不会吧", "震惊", "惊讶", "什么", "居然", "竟然", "不可思议"],
    }

    def _detect_emotion_keywords(self, text: str) -> str:
        """关键词匹配情绪检测（回退方案）"""
        for emotion, keywords in self._EMOTION_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    return emotion
        return "neutral"

    async def _detect_emotion_llm(self, text: str, event: AstrMessageEvent) -> str:
        """通过指定或当前模型提供商检测情绪，失败回退关键词"""
        try:
            provider_id = self.emotion_llm_provider
            if not provider_id:
                provider_id = await self.context.get_current_chat_provider_id(
                    umo=event.unified_msg_origin
                )
            if not provider_id:
                raise RuntimeError("未找到可用的模型提供商")
            prompt = (
                "分析以下消息的情绪，只输出一个词：happy/sad/angry/surprised/neutral。\n"
                f"消息：{text}"
            )
            llm_resp = await self.context.llm_generate(
                chat_provider_id=provider_id,
                prompt=prompt,
            )
            content = llm_resp.completion_text.strip().lower()
            if content in ("happy", "sad", "angry", "surprised", "neutral"):
                logger.info(f"[Qwen-TTS] LLM 情绪({provider_id}): {content}")
                return content
            logger.warning(f"[Qwen-TTS] LLM 返回非预期值 '{content}'，回退关键词")
        except Exception as e:
            logger.warning(f"[Qwen-TTS] LLM 情绪检测异常: {e}，回退关键词")

        return self._detect_emotion_keywords(text)

    async def _get_ref_audio_for_text(self, text: str, event: AstrMessageEvent) -> str:
        """根据文本选择合适的参考音频 base64，支持 LLM 或关键词检测"""
        if not self.emotion_mode or not self.emotion_audios:
            return self.ref_audio_base64

        if self.emotion_detect_method == "llm":
            emotion = await self._detect_emotion_llm(text, event)
        else:
            emotion = self._detect_emotion_keywords(text)

        if emotion in self.emotion_audios:
            return self.emotion_audios[emotion]
        if "neutral" in self.emotion_audios:
            return self.emotion_audios["neutral"]
        return self.ref_audio_base64

    def _check_whitelist(self, event: AstrMessageEvent) -> bool:
        """检查用户是否在白名单中，白名单为空时放行所有人"""
        if not self.whitelist:
            return True
        user_id = event.get_sender_id()
        return user_id in self.whitelist

    # ==================== 随机语音回复 ====================

    def _clean_text(self, text: str) -> str:
        """清洗文本：去掉中文/英文括号及其内容，合并空白"""
        text = re.sub(r'[（(][^）)]*[）)]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    @filter.on_agent_done()
    async def _random_voice_handler(
        self, event: AstrMessageEvent,
        run_context: ContextWrapper[AstrAgentContext],
        resp: LLMResponse,
    ):
        """监听 Agent 完成，按概率将最终回复以克隆音色转为语音"""
        if self.voice_probability <= 0:
            return
        if not self._check_whitelist(event):
            return
        if not self.ref_audio_base64 and not self.emotion_audios:
            return
        reply = self._clean_text(resp.completion_text)
        if not reply:
            return
        if random.random() > self.voice_probability:
            return
        try:
            ref_audio = await self._get_ref_audio_for_text(reply, event)
            wav_path = await self._call_clone_api(reply, ref_audio=ref_audio)
            if wav_path and os.path.exists(wav_path):
                await event.send(MessageChain([Record(file=str(wav_path))]))
        except Exception as e:
            logger.warning(f"[Qwen-TTS] 随机语音失败: {e}")

    # ==================== 指令 ====================

    @filter.command("tts")
    async def tts_generate(self, event: AstrMessageEvent, message: str):
        """用克隆音色将文本转为语音"""
        if not self._check_whitelist(event):
            yield event.plain_result("无权限使用此指令")
            return
        text = message.strip()
        if not text:
            yield event.plain_result(
                "用法: /tts <要合成的文本>\n"
                "示例: /tts 你好，这是用克隆音色生成的语音。\n"
                "使用 /ttsconfig 查看参考音频加载状态。"
            )
            return

        if not self.ref_audio_base64 and not self.emotion_audios:
            yield event.plain_result(
                "未加载参考音频，无法使用声音克隆。\n"
                "请在插件配置中设置 ref_audio_path 后重载插件。\n"
                "参考音频要求: WAV 格式，10-30 秒干净人声。"
            )
            return

        if len(text) > 500:
            yield event.plain_result("文本过长，建议控制在 500 字以内。")
            return

        if self.show_tts_hint:
            yield event.plain_result(
                f"正在用克隆音色合成语音，请稍候...\n"
                f"模型: {self.model_name}\n"
                f"语言: {LANG_MAP.get(self.default_language, self.default_language)}"
            )

        wav_path = None
        try:
            ref_audio = await self._get_ref_audio_for_text(text, event)
            wav_path = await self._call_clone_api(text, ref_audio=ref_audio)
            if wav_path and os.path.exists(wav_path):
                file_size = os.path.getsize(wav_path)
                logger.info(f"[Qwen-TTS] 克隆语音生成成功: {wav_path} ({file_size} bytes)")
                yield event.chain_result([
                    Record(file=str(wav_path)),
                ] + ([Plain(f"[TTS Clone] {text[:50]}{'...' if len(text) > 50 else ''}")] if self.show_tts_hint else []))
            else:
                yield event.plain_result("语音生成失败，未获取到音频文件。")
        except httpx.TimeoutException:
            yield event.plain_result("TTS API 请求超时，请稍后重试或缩短文本。")
        except httpx.ConnectError:
            yield event.plain_result(
                "无法连接到 Qwen3-TTS 服务。请确认 WebUI 已启动（默认端口 7860）。"
            )
        except Exception as e:
            logger.error(f"[Qwen-TTS] 生成异常: {e}")
            yield event.plain_result(f"语音生成失败: {str(e)}")

    @filter.command("ttslang")
    async def tts_languages(self, event: AstrMessageEvent):
        """列出支持的语言"""
        if not self._check_whitelist(event):
            yield event.plain_result("无权限使用此指令")
            return
        lines = ["支持的语言（可在插件配置中修改默认值）：", ""]
        for code, name in LANG_MAP.items():
            marker = " ← 当前默认" if code == self.default_language else ""
            lines.append(f"  {code:<14} {name}{marker}")
        yield event.plain_result("\n".join(lines))

    @filter.command("ttsmodels")
    async def tts_models(self, event: AstrMessageEvent):
        """从 API 获取可用模型列表"""
        if not self._check_whitelist(event):
            yield event.plain_result("无权限使用此指令")
            return
        models_url = self.tts_api_url.rsplit("/", 2)[0] + "/models"
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(models_url)
                if resp.status_code == 200:
                    data = resp.json()
                    model_list = data.get("models", [])
                    lines = ["可用模型列表：", ""]
                    for m in model_list:
                        name = m.get("name", "?")
                        mtype = m.get("type", "?")
                        current = " ← 当前使用" if name == self.model_name else ""
                        lines.append(f"  [{mtype}] {name}{current}")
                    yield event.plain_result("\n".join(lines))
                else:
                    yield event.plain_result("获取模型列表失败，请检查 TTS 服务状态。")
        except Exception as e:
            yield event.plain_result(f"无法获取模型列表: {str(e)}")

    @filter.command("ttsconfig")
    async def tts_config(self, event: AstrMessageEvent):
        """查看当前配置和参考音频状态"""
        if not self._check_whitelist(event):
            yield event.plain_result("无权限使用此指令")
            return
        if self.ref_audio_base64:
            ref_status = f"已加载 (base64 长度: {len(self.ref_audio_base64)})"
        elif self.ref_audio_path:
            ref_status = f"加载失败，请检查文件: {self.ref_audio_path}"
        else:
            ref_status = "未配置"

        ref_text_info = self.ref_text if self.ref_text else "(未填写，使用 x_vector_only 模式)"

        info = (
            f"Qwen3-TTS 声音克隆插件状态:\n"
            f"  API 地址: {self.tts_api_url}\n"
            f"  模型: {self.model_name}\n"
            f"  默认语言: {LANG_MAP.get(self.default_language, self.default_language)}\n"
            f"  参考音频: {ref_status}\n"
            f"  参考文本: {ref_text_info}\n"
            f"  请求超时: {self.request_timeout}s\n"
            f"  随机语音概率: {self.voice_probability:.0%}\n"
            f"  情绪模式: {'开启' if self.emotion_mode else '关闭'}"
        )
        if self.emotion_mode and self.emotion_audios:
            info += f"\n  已加载情绪: {', '.join(self.emotion_audios.keys())}"
        if self.emotion_mode:
            method_label = "AstrBot 模型提供商" if self.emotion_detect_method == "llm" else "关键词匹配"
            info += f"\n  情绪检测: {method_label}"
            if self.emotion_detect_method == "llm":
                if self.emotion_llm_provider:
                    info += f"\n  情绪检测提供商: {self.emotion_llm_provider}"
                else:
                    info += "\n  情绪检测提供商: 跟随当前会话"
        yield event.plain_result(info)

    # ==================== API 调用 ====================

    async def _call_clone_api(self, text: str, ref_audio: str = "") -> str:
        """调用声音克隆 API，返回生成的 WAV 文件路径。
        可传入 ref_audio 覆盖默认参考音频（情绪模式）。"""
        language = None if self.default_language == "auto" else self.default_language
        x_vector_only = not self.ref_text

        payload = {
            "model_name": self.model_name,
            "text": text,
            "language": language,
            "ref_audio_base64": ref_audio or self.ref_audio_base64,
            "ref_text": self.ref_text if self.ref_text else None,
            "segment_gen": False,
        }

        # x_vector_only_mode 传递方式：ref_text 为空时 API 内部会判断
        # 根据后端代码，当 ref_text 为 None 或空时，x_vector_only_mode=True

        async with httpx.AsyncClient(timeout=self.request_timeout) as client:
            resp = await client.post(
                self.tts_api_url,
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                detail = "未知错误"
                try:
                    detail = resp.json().get("detail", resp.text)
                except Exception:
                    detail = resp.text
                raise RuntimeError(f"API 返回错误 (HTTP {resp.status_code}): {detail}")

            data = resp.json()
            audio_list = data.get("audio_files_base64", [])
            if not audio_list:
                raise RuntimeError("API 未返回音频数据")

            audio_bytes = base64.b64decode(audio_list[0])
            wav_path = Path(tempfile.gettempdir()) / f"qwen_tts_clone_{abs(hash(text))}.wav"
            wav_path.write_bytes(audio_bytes)

            logger.info(
                f"[Qwen-TTS] 克隆 API 调用成功: info={data.get('info', '')}, "
                f"输出={wav_path}, 大小={len(audio_bytes)}bytes"
            )
            return str(wav_path.resolve())

    async def terminate(self):
        logger.info("[Qwen-TTS] 插件已卸载")
