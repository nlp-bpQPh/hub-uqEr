"""
多 LLM 故障转移演示服务（断路器版）

教学重点：
  1. 断路器模式（Circuit Breaker）：CLOSED / OPEN / HALF_OPEN 三态机
  2. 主动健康探针：后台 asyncio 任务，不依赖用户请求触发恢复
  3. 被动检测：请求失败时累计计数，达阈值后自动断路
  4. 快速失败（Fail Fast）：OPEN 状态直接跳过，不等待 API 超时
  5. 多 Provider 故障转移：同一家 DeepSeek 的两个型号 —— Pro 主、Flash 备

Provider 设计（两个 Provider 都是真实可演示的云端调用）：
  - deepseek-pro   : 主 Provider，模型默认 deepseek-v4-pro（可用 DEEPSEEK_MODEL_PRO 覆盖）
  - deepseek-flash : 备 Provider，模型默认 deepseek-flash（可用 DEEPSEEK_MODEL_FLASH 覆盖）
  两者共用 DEEPSEEK_API_KEY 与 https://api.deepseek.com，所以不需要第二家厂商的账号。

  注意：DeepSeek 的 deepseek-flash / deepseek-v4-pro 默认开启深度思考（thinking），
  演示时为了响应快、正文不空，这里统一传 thinking={"type": "disabled"}。

  演示技巧：把 Pro 打成故障（simulate-fail 或 trip-circuit），请求会自动转到 Flash；
  等冷却时间（默认 30s）过后，后台主动探针把 Pro 恢复成 CLOSED。全程真实调用 API。
  课堂演示路径（对应前端右侧三个按钮）：
    ① 点「维护」→ Pro 被人工跳过，请求走 Flash，但断路器状态一点没变（不是它坏了，是人让它下线）
    ② 点「断路器」→ Pro 立刻 OPEN，请求"快速失败"零点几秒就切到 Flash，不陪着 API 等超时
    ③ 什么都不点，等 30s → 后台探针把 Pro 从 OPEN 试成 HALF_OPEN，成功即 CLOSED，请求回到 Pro

使用方式：
  export DEEPSEEK_API_KEY="sk-xxx"      # 两个 Provider 共用
  # 可选：模型名或地址要换时再设
  export DEEPSEEK_MODEL_PRO="deepseek-v4-pro"
  export DEEPSEEK_MODEL_FLASH="deepseek-flash"
  export DEEPSEEK_BASE_URL="https://api.deepseek.com"
  uvicorn src.serve:app --host 0.0.0.0 --port 8000
  # 若 8000 被本地 vLLM 占用，换成 --port 8001

依赖：
  pip install fastapi uvicorn openai
"""

import os
import time
import asyncio
import logging
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager
from functools import partial

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from openai import OpenAI

# 打开 INFO 日志：断路器的每次状态迁移都会打到终端，课上一边点按钮一边看日志最直观。
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── 断路器状态机 ───────────────────────────────────────────────────────────────

# 三个状态的名字借自电路：CLOSED = 电路闭合 = 通电 = 正常放行；OPEN = 电路断开 = 请求被切断。
# 初学者最容易记反 —— 这里的 OPEN 是坏消息，不是"打开可用"。
# 继承 str 是为了状态值能直接进 JSON：前端拿到的就是 "CLOSED" 这样的字符串，不用再转换。
class CircuitState(str, Enum):
    CLOSED    = "CLOSED"     # 正常，所有请求放行
    OPEN      = "OPEN"       # 断路，直接快速失败，不发真实请求
    HALF_OPEN = "HALF_OPEN"  # 探针状态，放行一条请求试探恢复


@dataclass
class CircuitBreaker:
    """
    断路器核心实现。

    状态转移规则：
      CLOSED    --[连续失败 >= failure_threshold]──────> OPEN
      OPEN      --[冷却时间 >= recovery_timeout]────────> HALF_OPEN  ← 由 can_attempt() 触发
      HALF_OPEN --[探针成功 >= success_threshold]───────> CLOSED
      HALF_OPEN --[探针失败]────────────────────────────> OPEN（重置冷却计时）

    关键设计：OPEN 状态下请求直接被拒绝（快速失败），不等待 API 超时，
    保护下游不被雪崩请求拖垮，同时让调用方快速切换到备用 Provider。
    """
    name: str
    # 生产推荐值：failure_threshold=3~5, recovery_timeout=60~120s, success_threshold=2
    failure_threshold: int   = 3   # 连续失败几次后断路
    recovery_timeout:  int   = 30  # 断路后等多少秒进入 HALF_OPEN（Demo 用 30s）
    success_threshold: int   = 1   # HALF_OPEN 探针连续成功几次才真正关闭（Demo 用 1）

    # 运行期状态（全部在内存里，服务重启即清零；多实例部署时就得挪到 Redis 之类的地方共享）：
    #   failure_count     —— 连续失败次数，成功一次就归零，所以量的是"连续"而不是"累计"
    #   success_count     —— HALF_OPEN 期间探针已经成功了几次（够 success_threshold 才敢关闭）
    #   last_failure_time —— 最后一次失败的时间戳，OPEN 的冷却倒计时就是从它开始算的
    state:             CircuitState = CircuitState.CLOSED
    failure_count:     int          = 0
    success_count:     int          = 0
    last_failure_time: float        = 0.0

    def can_attempt(self) -> bool:
        """判断是否允许发出请求。OPEN 状态时检查冷却时间，到期自动迁移到 HALF_OPEN。"""
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.OPEN:
            # 这里没有专门的定时器负责"到点把状态改回来"，而是谁来问 can_attempt，
            # 就顺手看一眼表：冷却够了就当场翻成 HALF_OPEN。好处是不用起额外线程，
            # 代价是没人来问（也没有后台探针）时，状态会一直停在 OPEN。
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"[{self.name}] 断路器 OPEN → HALF_OPEN（冷却 {elapsed:.0f}s）")
                return True
            # 冷却没到就返回 False：调用方据此"快速失败"，而不是在这里 sleep 30 秒干等。
            return False
        # HALF_OPEN：只放少量探针请求过去试水 —— 成功就关门恢复，失败就重新断开。
        # 这种"试探性恢复"避免了刚恢复就被仍在涌入的流量一次性打回 OPEN。
        # NOTE: 高并发时多条请求可能同时通过，生产中需加 in-flight 锁
        return True

    def on_success(self):
        """请求成功时通知断路器。"""
        # 这里只处理 HALF_OPEN 和 CLOSED：OPEN 状态的请求在 can_attempt() 就被挡回去了，
        # 根本走不到真实调用，自然也到不了这个"成功回调"。
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            logger.info(f"[{self.name}] HALF_OPEN 探针成功 {self.success_count}/{self.success_threshold}")
            if self.success_count >= self.success_threshold:
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                logger.info(f"[{self.name}] 断路器 HALF_OPEN → CLOSED（已恢复）")
        elif self.state == CircuitState.CLOSED:
            # 成功一次就把连续失败清零：要衡量的是"连着错"，偶发失败不该一直攒着等跳闸。
            self.failure_count = 0  # 成功请求重置连续失败计数

    def on_failure(self):
        """请求失败时通知断路器。"""
        # 不管当前什么状态，失败次数和失败时间都先记一笔（冷却倒计时依赖这个时间戳）。
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.state == CircuitState.HALF_OPEN:
            # 探针只放了一条过去，它还失败说明"病没好"：立刻回到 OPEN 重新冷却。
            # 注意不需要再攒够 failure_threshold 次 —— 试探失败一次就足够说明问题。
            self.state = CircuitState.OPEN
            logger.warning(f"[{self.name}] HALF_OPEN 探针失败 → OPEN（重新冷却）")
        elif self.failure_count >= self.failure_threshold:
            # CLOSED 下要"连续"失败到阈值才跳闸，避免个别抖动就把整条线路切断。
            self.state = CircuitState.OPEN
            logger.warning(f"[{self.name}] 连续失败 {self.failure_count} 次 → OPEN（断路）")

    @property
    def recovery_in(self) -> float:
        """OPEN 状态下距离进入 HALF_OPEN 还有多少秒。"""
        # 前端拿这个值画"xx 秒后台探针"的倒计时；非 OPEN 状态一律返回 0，前端据此隐藏倒计时。
        if self.state != CircuitState.OPEN:
            return 0.0
        return max(0.0, self.recovery_timeout - (time.time() - self.last_failure_time))


# ── Provider 配置 ──────────────────────────────────────────────────────────────
#
# 顺序即优先级：先试 Pro，失败/断路则自动转移到 Flash。
# 两个 Provider 都是 DeepSeek 官方 API，共用 DEEPSEEK_API_KEY 与 base_url，
# 只是模型不同 —— 这样演示「主备切换」不需要第二家厂商的账号。

# 这两个模型共用同一个 Key 和 base_url，区别只在模型名 —— 这就是"同一家厂商的两个档位"
# 做降级：不用第二家厂商的账号，也就不用维护两套鉴权和计费。
DEFAULT_BASE_URL = "https://api.deepseek.com"
DEFAULT_MODEL_PRO = "deepseek-v4-pro"
DEFAULT_MODEL_FLASH = "deepseek-flash"


def _resolve_api_key(p: dict) -> str:
    """Provider 的 API Key 直接读环境变量（两个 Provider 共用 DEEPSEEK_API_KEY）。"""
    # Key 只从环境变量读，不写进代码、也不回显到接口里：对外的 /providers 只暴露 has_key
    # 这个布尔值，避免密钥跟着 JSON 一起被浏览器或日志看见。
    return os.environ.get(p["api_key_env"], "").strip()


# 每个 Provider 各自持有一个独立的 CircuitBreaker（cb 字段），状态互不影响 ——
# 这正是演示要的效果：Pro 连续失败不该牵连 Flash。顺序即优先级。
# 字段含义：id/name/model 直接显示在右侧卡片上；simulated_fail 是"人工维护"开关；
# cb 是该 Provider 的专属断路器，失败计数与三态都记在里面。
PROVIDERS = [
    {
        "id":             "deepseek-pro",
        "name":           "DeepSeek Pro（主）",
        "model":          os.environ.get("DEEPSEEK_MODEL_PRO", DEFAULT_MODEL_PRO),
        "api_key_env":    "DEEPSEEK_API_KEY",
        "base_url":       os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        "simulated_fail": False,
        "cb":             CircuitBreaker(name="deepseek-pro"),
    },
    {
        "id":             "deepseek-flash",
        "name":           "DeepSeek Flash（备）",
        "model":          os.environ.get("DEEPSEEK_MODEL_FLASH", DEFAULT_MODEL_FLASH),
        "api_key_env":    "DEEPSEEK_API_KEY",
        "base_url":       os.environ.get("DEEPSEEK_BASE_URL", DEFAULT_BASE_URL),
        "simulated_fail": False,
        "cb":             CircuitBreaker(name="deepseek-flash"),
    },
]


# ── API 调用封装 ───────────────────────────────────────────────────────────────

def _call_sync(p: dict, messages: list, max_tokens: int = None) -> str:
    """
    同步调用单个 Provider（供 run_in_executor 使用）。
    NOTE: 生产中应缓存 OpenAI client 实例，避免每次重建。
    """
    api_key = _resolve_api_key(p)
    if not api_key:
        # 没配 Key 直接抛错：让上层记一次失败并切到下一个 Provider，而不是编一句假答案糊弄过去。
        raise RuntimeError(f"环境变量 {p['api_key_env']} 未设置")
    # Demo 图省事每次新建 client；生产应缓存复用（省掉 TLS 握手和连接池重建的开销）。
    client = OpenAI(api_key=api_key, base_url=p["base_url"])
    kwargs = dict(model=p["model"], messages=messages)
    if max_tokens is not None:
        # 探针只为确认"这个 Provider 还活着"，答案好不好不重要 —— 给小 max_tokens
        # 既省 token 花费，也让卡住的请求更早暴露成失败。
        kwargs["max_tokens"] = max_tokens
    # 关掉深度思考：演示要的是「谁在回答」，不是慢速推理；
    # 且 thinking 默认开启时 max_tokens 会被思考链吃光，正文可能是空串。
    kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
    # 注意探针和正常对话都走这一个函数，所以"关思考"是全局统一的，
    # 主备两个 Provider 的行为一致，课上不会出现"Flash 想半天、Pro 秒回"的干扰。
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content
    if not content:
        # 模型返回空正文时视为失败，交给断路器/备 Provider 处理。
        # 不在这里偷偷重试：把"空正文"和"API 报错"统一成同一种失败，
        # 演示时无论哪种原因，表现都是干净地切到下一个 Provider。
        raise RuntimeError(f"模型 {p['model']} 返回空正文")
    return content


# ── 主动健康探针 ───────────────────────────────────────────────────────────────

PROBE_INTERVAL = 15   # 秒（Demo 用 15s，生产用 30~60s）
PROBE_MESSAGES = [{"role": "user", "content": "hi"}]


async def probe_provider(p: dict):
    """对单个 Provider 发轻量探针请求，更新断路器状态。"""
    cb: CircuitBreaker = p["cb"]
    # 探针自己也要守断路器的规矩：OPEN 且冷却没到就不发请求（否则等于绕过了断路器）。
    if not cb.can_attempt():
        return
    try:
        logger.info(f"[{p['name']}] 主动探针（断路器: {cb.state}）")
        # OpenAI SDK 是同步阻塞的，必须丢到线程池里跑；直接 await 会卡住整个 asyncio
        # 事件循环，期间连 /providers 都返回不了，前端倒计时就停住了。
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, partial(_call_sync, p, PROBE_MESSAGES, 5))
        cb.on_success()
    except Exception as e:
        # 异常必须在这里兜住：探针只是辅助功能，它失败不该把后台循环带走，
        # 否则这个进程从此再也不会自愈。
        cb.on_failure()
        logger.warning(f"[{p['name']}] 探针失败: {e}")


async def health_probe_loop():
    """
    后台常驻任务：定期对处于 OPEN 状态的 Provider 发探针。

    为什么只探 OPEN 状态，不探 CLOSED：
      - CLOSED 状态通过真实用户请求感知故障（被动检测），无需额外消耗 token
      - OPEN 状态没有用户请求到达，必须主动探针才能触发恢复
    """
    logger.info(f"健康探针后台任务已启动，间隔 {PROBE_INTERVAL}s")
    while True:
        # 先睡再探：服务刚启动时所有 Provider 都是 CLOSED，没必要立刻打一次 API。
        await asyncio.sleep(PROBE_INTERVAL)
        for p in PROVIDERS:
            if p["simulated_fail"]:
                # 维护中的 Provider 不发探针：用户请求本来就会跳过它，
                # 再花 token 去探等于白花钱；而且恢复该由人点「恢复」按钮，不该被后台悄悄改回去。
                continue
            cb: CircuitBreaker = p["cb"]
            if cb.state == CircuitState.OPEN:
                # 只探 OPEN：CLOSED 靠真实用户请求被动发现问题（不额外烧 token）；
                # HALF_OPEN 说明已经有探针在路上了，别叠第二条。
                await probe_provider(p)


# ── FastAPI 应用 ───────────────────────────────────────────────────────────────

HTML_PATH = Path(__file__).parent.parent / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时创建后台探针任务，关停时取消它 —— 不取消的话，uvicorn --reload
    # 每次重载都会多留一个死循环在后台，越跑越多。
    task = asyncio.create_task(health_probe_loop())
    yield
    task.cancel()


app = FastAPI(title="LLM Failover Demo", lifespan=lifespan)


class ChatRequest(BaseModel):
    """POST /chat 的请求体：message 是本轮新问题，history 是此前的对话（后端不存记忆，全靠前端回传）。"""
    message: str
    # 默认空列表兜底，这样前端只发 message 也能跑（Pydantic 负责校验类型）。
    history: list = []


@app.get("/")
async def index():
    """直接把项目根目录的 index.html 发给浏览器（前后端同源，前端请求才敢用相对路径）。"""
    return FileResponse(HTML_PATH)


@app.get("/providers")
async def get_providers():
    """给前端轮询用的状态快照：断路器三态、失败计数、剩余冷却秒数、当前是否可用。"""
    result = []
    for p in PROVIDERS:
        cb: CircuitBreaker = p["cb"]
        result.append({
            "id":               p["id"],
            "name":             p["name"],
            "model":            p["model"],
            "has_key":          bool(_resolve_api_key(p)),
            "simulated_fail":   p["simulated_fail"],
            "circuit_state":    cb.state,
            "failure_count":    cb.failure_count,
            "failure_threshold": cb.failure_threshold,
            "recovery_in":      round(cb.recovery_in),
            # available = 不在维护模式 且 断路器不是 OPEN
            # 注意 HALF_OPEN 仍算"可用"：它就是要放一条请求过去试探恢复。
            "available":        not p["simulated_fail"] and cb.state != CircuitState.OPEN,
        })
    return result


@app.post("/providers/{provider_id}/simulate-fail")
async def simulate_fail(provider_id: str):
    """模拟人工维护/下线（立即跳过，不影响断路器状态）。"""
    for p in PROVIDERS:
        if p["id"] == provider_id:
            # 只翻转模拟开关，一根手指都不碰 cb：维护是"人主动让它下线"，不是"它自己坏了"。
            # 所以恢复时也不用等 30s 冷却，点「恢复」立刻就能重新接流量。
            p["simulated_fail"] = True
            logger.info(f"[{p['name']}] 维护模式开启")
            return {"status": "ok"}
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.post("/providers/{provider_id}/trip-circuit")
async def trip_circuit(provider_id: str):
    """
    直接将断路器置为 OPEN（演示用途）。
    等价于真实场景中连续调用失败 failure_threshold 次后的结果。
    """
    for p in PROVIDERS:
        if p["id"] == provider_id:
            cb: CircuitBreaker = p["cb"]
            cb.state = CircuitState.OPEN
            # 顺手把计数填满、把失败时间刷成"现在"：
            #   计数填满 → 保持"计数已达阈值"的不变量，前端进度条画满，不会出现 OPEN 却只失败过 0 次
            #   时间刷新 → 30s 冷却从点击这一刻开始算，不会刚到点就被探针立刻救活
            cb.failure_count = cb.failure_threshold
            cb.last_failure_time = time.time()
            logger.info(f"[{p['name']}] 手动触发断路器 OPEN（演示），{cb.recovery_timeout}s 后探针")
            return {"status": "ok", "recovery_in": cb.recovery_timeout}
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.post("/providers/{provider_id}/restore")
async def restore_provider(provider_id: str):
    """恢复 Provider：清除维护模式并重置断路器为 CLOSED。"""
    for p in PROVIDERS:
        if p["id"] == provider_id:
            # 「恢复」一次做两件事：退出维护模式 + 把断路器强行清零回 CLOSED。
            # 之所以能直接清零，是因为这是人在表态"我确认它没问题了"，跳过冷却和探针。
            p["simulated_fail"] = False
            cb: CircuitBreaker = p["cb"]
            cb.state = CircuitState.CLOSED
            cb.failure_count = 0
            cb.success_count = 0
            logger.info(f"[{p['name']}] 已恢复，断路器重置 CLOSED")
            return {"status": "ok"}
    return JSONResponse(status_code=404, content={"error": "not found"})


@app.post("/providers/reset")
async def reset_all():
    """把两个 Provider 一起复位（对应前端「↺ 全部恢复」按钮），课上演示前清场用。"""
    for p in PROVIDERS:
        p["simulated_fail"] = False
        cb: CircuitBreaker = p["cb"]
        cb.state = CircuitState.CLOSED
        cb.failure_count = 0
        cb.success_count = 0
    logger.info("所有 Provider 已重置")
    return {"status": "ok"}


@app.post("/chat")
async def chat(req: ChatRequest):
    """
    演示的主角：按优先级依次尝试各个 Provider，谁先答出来就用谁。

    返回体里的 provider / tried 是给前端画"这次走了哪条路"的，属于前后端约定好的契约字段。
    """
    # system 提示统一在这里拼：前端不用关心，也就没法被学生改坏。
    messages = [{"role": "system", "content": "You are a helpful assistant. 请用中文回答。"}]
    # 只带最近 20 条历史：既控制上下文长度（直接等于 token 花费），也避免越聊越慢。
    for msg in req.history[-20:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": req.message})

    # 记录"每个 Provider 为什么没被用上"，原样返回给前端画降级路径（⚡ Pro › Flash）。
    tried = []

    # 按 PROVIDERS 的顺序逐个试：第一个能用的答出来就 return，后面的 Provider 根本不会被调用。
    for p in PROVIDERS:
        cb: CircuitBreaker = p["cb"]

        # 检查 1：人工维护模式（立即跳过）
        # 三道检查的顺序是有讲究的：从"最确定、最便宜"到"最昂贵"。
        # 维护和断路器都在本地内存里判，零成本；真发 API 请求才花钱花时间，放到最后。
        if p["simulated_fail"]:
            tried.append({"provider": p["name"], "reason": "维护中"})
            continue

        # 检查 2：断路器（快速失败，不发真实请求）
        # 关键就在"不发请求"：OPEN 时若还去真调 API，就得陪着等超时（几秒到几十秒），
        # 用户干等着、线程还被占着。直接判失败，把这点时间让给下一个 Provider。
        if not cb.can_attempt():
            tried.append({
                "provider": p["name"],
                "reason": f"断路器 OPEN（{round(cb.recovery_in)}s 后探针）",
            })
            continue

        # 检查 3：API Key（两个 Provider 共用 DEEPSEEK_API_KEY）
        api_key = _resolve_api_key(p)
        if not api_key:
            tried.append({"provider": p["name"], "reason": "无 API Key"})
            continue

        # 发起真实调用
        try:
            logger.info(f"[{p['name']}] 调用中（断路器: {cb.state}）")
            # 同步 SDK 丢线程池：await 期间事件循环还能继续响应别人的 /providers 轮询，
            # 否则一个慢请求会把整个页面（包括状态卡片）一起卡住。
            loop = asyncio.get_running_loop()
            content = await loop.run_in_executor(None, partial(_call_sync, p, messages))
            cb.on_success()
            logger.info(f"[{p['name']}] 调用成功")
            # 成功就立刻返回，不再往后试 —— 故障转移只关心"找到一个能用的"，不是"全部试一遍"。
            return {
                "answer":      content,
                "provider":    p["name"],
                "provider_id": p["id"],
                "tried":       tried,
            }
        except Exception as e:
            # 记一次失败，然后继续 for 循环 —— 这就是"故障转移"：同一个问题交给下一个 Provider 再答。
            cb.on_failure()
            logger.error(f"[{p['name']}] 调用失败（已失败 {cb.failure_count}/{cb.failure_threshold} 次）: {e}")
            tried.append({
                "provider": p["name"],
                "reason": f"API 错误（失败 {cb.failure_count}/{cb.failure_threshold} 次）",
            })

    # 走到这里说明所有 Provider 都被跳过或调用失败了：用 503（服务暂不可用）而不是 500，
    # 因为服务本身是好的，只是上游都不可用。带上 tried，前端才能显示"已尝试: A（原因）→ B（原因）"。
    return JSONResponse(
        status_code=503,
        content={"error": "所有 LLM 提供商均不可用", "tried": tried},
    )
