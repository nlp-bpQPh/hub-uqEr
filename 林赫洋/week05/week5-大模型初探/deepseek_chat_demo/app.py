# ============================================================================
# DeepSeek 聊天 Demo —— Flask 后端
#
# 这个文件只干三件事：
#   1. GET  /                 把 templates/index.html 发给浏览器
#   2. POST /api/chat         非流式：等模型整段生成完，一次性返回
#   3. POST /api/chat/stream  流式：边从 DeepSeek 收增量，边用 SSE 推给浏览器
#
# 为什么要在中间加一层后端、不让浏览器直连 DeepSeek？
#   - API Key 只存在服务端，不会出现在前端代码和浏览器网络面板里；
#   - model / thinking 这类参数可以在服务端统一控制，前端只说「要不要思考」；
#   - 顺手绕开浏览器跨域（CORS）限制，前端只跟同源的 127.0.0.1:5000 打交道。
#
# 运行前先设置环境变量（Windows PowerShell）：
#   $env:DEEPSEEK_API_KEY="sk-xxxx"
# ============================================================================
import json
import os

import requests
from flask import Flask, Response, jsonify, render_template, request, stream_with_context

# Flask 默认在「本文件同级的 templates/ 目录」找模板，所以 index.html 放在 templates/ 下
app = Flask(__name__)

# API Key 从环境变量读，绝不写死在代码里：这份代码会发给学生、也可能被截图或提交，
# 写死的 Key 等于公开泄露（余额可能被人刷光）。没配置时下面的路由会给出明确报错。
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY")
# 默认用 DeepSeek 官方的 OpenAI 兼容端点，所以请求格式和 OpenAI 一致；
# 换成别的兼容服务（本地 vLLM、第三方中转）只要改这个环境变量，代码不用动。
DEEPSEEK_CHAT_URL = os.environ.get(
    "DEEPSEEK_CHAT_URL", "https://api.deepseek.com/v1/chat/completions"
)
# 模型名也允许用环境变量覆盖：想对比不同模型/版本时，不用改代码重新发一份课件。
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-flash")


def build_payload(messages, stream, think):
    """把前端传来的对话组装成 DeepSeek 的请求体。

    这里最关键、也最容易踩坑的是 thinking 字段：DeepSeek 的模型**默认就开启深度思考**，
    请求里不写 thinking 并不等于关闭思考，而是照样思考。想真正关掉必须显式写
    {"type": "disabled"}。所以前端要把「是否思考」作为 think 参数显式传下来，
    由这里翻译成 enabled / disabled —— 不能指望「不传就是关」。
    """
    return {
        "model": MODEL,
        "messages": messages,
        "stream": stream,
        # 三目运算把布尔开关直接映射成协议里的两个枚举值，避免漏传时又退回「默认思考」。
        "thinking": {"type": "enabled" if think else "disabled"},
        # 注意这里有意不传 max_tokens：设得太小时，思考消耗的 token 会把预算吃光，
        # 模型可能「想完了」却挤不出一句正文，前端就会收到空的 content。
    }


def build_headers():
    # DeepSeek 兼容 OpenAI 的鉴权方式：Authorization: Bearer <API Key>。
    return {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }


@app.route("/")
def index():
    # 只渲染模板；页面上没有任何密钥，密钥始终留在服务端。
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    """非流式对话：等 DeepSeek 把整段回答生成完，再一次性返回。

    实现最简单，缺点是没有中间反馈 —— 长回答时前端只能干等（流式接口正是来解决这个的）。
    """
    # 先自查配置再发请求：直接提示「没配 Key」，比让请求打到 DeepSeek 再报 401 好排查。
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "未检测到 DEEPSEEK_API_KEY 环境变量"}), 500

    data = request.get_json(force=True)
    messages = data.get("messages", [])
    # think 由前端显式传入；取不到就按 False（非思考）处理，手写请求也不会因为缺字段而报错。
    think = bool(data.get("think", False))

    try:
        # stream=False：requests 会把整个响应体读完才返回，所以这里没有流式那些坑。
        resp = requests.post(
            DEEPSEEK_CHAT_URL,
            headers=build_headers(),
            json=build_payload(messages, stream=False, think=think),
            # 思考模式可能想很久，超时给够，否则前端会莫名报错
            timeout=300,
        )
    except Exception as e:
        # 网络层错误（TLS、DNS、超时）单独兜住，返回 502：告诉调用方「上游不通」而不是「我们写错了」。
        return jsonify({"error": f"请求 DeepSeek 失败: {e}"}), 502
    # 非 200 说明 DeepSeek 拒绝了请求（Key 错、余额不足、参数非法……），
    # 把原始状态码和响应体一起透出，定位问题最快。
    if resp.status_code != 200:
        return jsonify({"error": f"DeepSeek API 错误: {resp.status_code} {resp.text}"}), 500

    result = resp.json()
    msg = result["choices"][0]["message"]
    content = msg.get("content", "")
    # 思考链在返回里是单独的 reasoning_content 字段，和正文 content 分开存；
    # 这里取出来只给前端展示，不会写回 messages —— 思考过程不属于对话内容，
    # 回传给 API 会污染上下文，还白白多花 token。
    reasoning = msg.get("reasoning_content", "") or ""
    usage = result.get("usage", {})
    # 返回结构 {content, reasoning, usage} 是与前端约定好的契约，改字段要同步改 index.html。
    return jsonify({"content": content, "reasoning": reasoning, "usage": usage})


@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """流式对话：一边从 DeepSeek 收增量，一边用 SSE（Server-Sent Events）推给浏览器。"""
    if not DEEPSEEK_API_KEY:
        return jsonify({"error": "未检测到 DEEPSEEK_API_KEY 环境变量"}), 500

    data = request.get_json(force=True)
    messages = data.get("messages", [])
    think = bool(data.get("think", False))

    def generate():
        # 生成器每 yield 一次，Flask 就把这一段立刻写回浏览器 —— 这就是「流式」的全部秘密。
        try:
            # 关键：stream=True 让 requests 不预读整个响应，而是交给我们按行迭代。
            r = requests.post(
                DEEPSEEK_CHAT_URL,
                headers=build_headers(),
                json=build_payload(messages, stream=True, think=think),
                stream=True,
                timeout=300,
            )
        except Exception as e:
            # 注意：这时响应头已经发出去了，HTTP 状态码没法再改，
            # 所以错误也只能包成一条 data: 事件交给前端去显示。
            err = {"error": f"请求 DeepSeek 失败: {e}"}
            yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
            return

        with r:
            # 用 with 包住响应，保证生成器正常结束或抛异常时连接一定被关闭，不会泄漏连接。
            if r.status_code != 200:
                err = {"error": f"DeepSeek API 错误: {r.status_code} {r.text}"}
                yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"
                return

            # iter_lines 按行读上游的 SSE：DeepSeek 每条消息是 `data: {...}`，空行作为分隔。
            for raw_line in r.iter_lines(decode_unicode=True):
                # 空行是 SSE 的分隔符，不含内容，跳过。
                if not raw_line:
                    continue
                # 只关心 data: 开头的行（event:、id:、注释行一律忽略）。
                if not raw_line.startswith("data:"):
                    continue
                payload = raw_line[len("data:"):].strip()
                # [DONE] 是 OpenAI 兼容协议的结束标记，转发给前端后就可以收工。
                if payload == "[DONE]":
                    yield "data: [DONE]\n\n"
                    break
                try:
                    obj = json.loads(payload)
                    delta = obj["choices"][0].get("delta", {})
                    # 流式下正文和思考链都放在 delta 里：思考模式时，DeepSeek 会先连续推
                    # reasoning_content，推完才开始推 content，所以前端自然先看到思考、
                    # 后看到正文。两条通道分开读，是为了让前端分别渲染到不同面板。
                    reasoning_piece = delta.get("reasoning_content") or ""
                    content_piece = delta.get("content") or ""
                    # 增量分别包成 {reasoning: ...} / {content: ...} 再发给前端，
                    # 前端一看字段名就知道该往思考面板还是正文气泡里追加。
                    if reasoning_piece:
                        out = {"reasoning": reasoning_piece}
                        yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                    if content_piece:
                        out = {"content": content_piece}
                        yield f"data: {json.dumps(out, ensure_ascii=False)}\n\n"
                except Exception as e:
                    # 单条解析失败不中断整个流：发一条 error 事件让前端提示用户，后面的数据继续用。
                    err = {"error": f"解析流数据失败: {e}"}
                    yield f"data: {json.dumps(err, ensure_ascii=False)}\n\n"

    return Response(
        # 为什么一定要包 stream_with_context？generate() 是在视图函数返回之后才被迭代的，
        # 那时这次请求的上下文已经退出了；不包它，生成器里一旦用到 request/session 就会报
        # "working outside of request context"。包上它，Flask 会把上下文带进生成器的执行期间。
        stream_with_context(generate()),
        # Content-Type 必须是 text/event-stream，浏览器/前端才按 SSE 协议解析这个响应。
        mimetype="text/event-stream",
        headers={
            # 别缓存：流式内容必须实时生成，缓存住就变成「假流式」了。
            "Cache-Control": "no-cache",
            # 关掉 Nginx 等反向代理的缓冲：否则代理会攒够一块才转发，前端会看到「卡半天再刷一堆」。
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    # debug=True 只用于课堂本地演示：改代码自动重启，出错时页面直接显示调试器（别带到线上）。
    app.run(host="127.0.0.1", port=5000, debug=True)
