第一周作业：开发环境准备与 AI Coding 工具试用
姓名：Zhechen Tu

一、开发环境
按照课程环境教程，在 macOS Apple Silicon 上安装 Miniconda，创建
名为 py312 的 Python 3.12.15（64 位）环境，使用已有的 VS Code
1.138.0 和 Python 扩展。VS Code 已选择 py312 作为当前工作区解释器。

已安装并实际导入验证：
NumPy 1.26.4、Pandas 2.2.2、Matplotlib 3.9.2、PyTorch 2.6.0、
Gensim 4.4.0、Scikit-learn 1.5.1、PEFT 0.15.0、Transformers 4.55.0。
教程允许 Miniconda 替代 Anaconda；Mac 的 torch 版本不带 +cpu 后缀。
本次示例明确在 CPU 上运行。

环境验证不仅检查版本，还运行了数组、DataFrame、绘图、线性回归、
词典、Torch 张量以及小型 BERT + LoRA 前向计算。无需下载模型。
结果：ALL ENVIRONMENT CHECKS PASSED，pip check 无依赖冲突。

二、AI Coding 工具安装与试用
选择工具：OpenAI Codex CLI 0.162.1。
从官方 npm 包 @openai/codex@0.162.1 完成独立安装，并使用已有的
ChatGPT 登录状态实际调用 CLI。作业允许自选工具，未选择 DeepSeek Harness。

试用任务：让 Codex 编写 NumPy + PyTorch 线性回归程序，学习 y=2x+1，
固定随机种子，输出拟合参数、误差及 x=5 的预测，并加入断言。
Codex 返回的源码原样保存为 ai_coding_demo.py，完整提示词见 ai_prompt.txt。

实际执行结果：
Learned weight: 1.999999
Learned bias: 1.000002
Final MSE: 0.0000000000
Prediction for x=5: 10.999998
ALL CHECKS PASSED

试用观察：自然语言需求可以生成完整的训练和预测代码；固定随机种子、
显式指定 CPU 和加入断言，使生成代码的结果可以检查。AI 生成后仍需运行
和核对，本次以误差阈值与预测值断言验证了程序。
本作业由 AI 辅助配置、生成和验证；记录只陈述实际完成的操作与结果。

三、截图说明
screenshots/01_environment.png：VS Code 中显示真实安装命令与环境检查输出。
screenshots/02_ai_tool_trial.png：VS Code 中显示 CLI 安装信息与实际生成记录。
screenshots/03_demo_result.png：AI 生成代码、实际运行输出和 py312 解释器。
上述截图是应用窗口截图；命令输出和 CLI 记录在编辑器中展示，
不是 CLI 交互界面的直接截图。原始结果文本与源码一并附上便于复核。

四、复现方式
已安装 Conda 时，在本目录执行：
conda env create -f environment.yml
conda activate py312
python -m pip check
python environment_check.py
python ai_coding_demo.py

直接依赖见 requirements.txt；本次实际安装的完整依赖见 requirements_lock.txt。
如需完整固定本次依赖，可在 Python 3.12 环境中执行：
python -m pip install -r requirements_lock.txt

Codex CLI 安装示例：
npm install --prefix ai-tool @openai/codex@0.162.1
./ai-tool/node_modules/.bin/codex login
./ai-tool/node_modules/.bin/codex

五、提交目录
Zhechen_Tu/week01/
通过 Fork 仓库发起 Pull Request，等待老师审批。

课程环境教程：https://note.youdao.com/s/BAcOYrOB
课程提交教程：https://note.youdao.com/s/KXGBGt8D
作业仓库：https://github.com/nlp-bpQPh/hub-uqEr
Codex 官方说明：https://learn.chatgpt.com/docs/codex/cli
