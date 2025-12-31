import random
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import csv
import os
from openai import OpenAI
import time

# ==== 设置随机种子 ====
RANDOM_SEED = 12  # 你可以选择任何你喜欢的数字作为种子
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ==== 两阶段置信区间参数 ====
CONFIDENCE_LEVEL = 0.95  # 置信度
PRECISION_THRESHOLD = 0.1  # 置信区间半长阈值
Z_SCORE = 1.96  # 95%置信度对应的Z值
INITIAL_SAMPLE_SIZE = 5  # 阶段I初始样本量（建议≥30）

# ==== 新增参数：最大总游戏次数 ====
MAX_TOTAL_GAMES = 50  # 最大总游戏次数限制

# ==== 其他参数保持不变 ====
N_ROUNDS = 100  # 每局游戏的回合数
ERROR_RATE = 0.1  # 误差率
LLM_TEMPERATURE = 1
X_E = 2  # 敲诈因子
P = 1  # 双方背叛收益
R = 3  # 双方合作收益

# ==== 文件路径 ====
DETAILED_CSV = f"detailed_results_error_{ERROR_RATE:.2f}_1_m&oCT.csv"  # 详细结果文件（合作率/收益）
SUMMARY_CSV = f"summary_results_error_{ERROR_RATE:.2f}_1_m&oCT.csv"    # 汇总结果文件
#REASON_CSV = f"llm_reasons_error_{ERROR_RATE:.2f}_1_o.csv"         # 决策理由文件

# ==== DeepSeek API 配置 ====
client = OpenAI(
    api_key="sk-80d99ef7ad9e4ca5b1e1f47f97312f40",  # 替换为你的API Key
    base_url="https://api.deepseek.com"
)

# ==== LLM Prompt ====
# 为LLM提供的场景描述，告知其当前处于重复囚徒困境游戏中
# 并说明了不同决策组合下的收益规则、可能存在的误差情况
# 以及要求LLM回复的固定格式
scenario = """
You are playing a repeated Prisoner's Dilemma game against a fixed opponent.
The game has the following payoff rules:
- If both you and your opponent cooperate (C), you both receive 3 points.
- If you defect (D) and your opponent cooperates (C), you receive 5 points and your opponent gets 0.
- If you cooperate (C) and your opponent defects (D), you receive 0 points and your opponent gets 5.
- If both defect (D), you both receive 1 point.
You will be informed of your payoff in the previous round.
Please decide whether to cooperate (C) or defect (D) in this round.
Reply with only one of the following statements:
- 'I choose to cooperate ...'
- 'I choose to defect ...'
"""

# ==== 收益矩阵 ====
PAYOFF = {
    ('C', 'C'): (3, 3),
    ('C', 'D'): (0, 5),
    ('D', 'C'): (5, 0),
    ('D', 'D'): (1, 1),
}

# ==== LLM决策函数 ====
def llm_decision(history_self, history_opponent, attempt=1, max_retries=3):
    """
    调用LLM获取决策和理由。若格式错误则递归重试，不默认背叛。

    :param history_opponent: 对手之前回合的决策历史列表
    :param attempt: 当前递归尝试次数
    :param max_retries: 最大允许重试次数
    :return: 决策（'C'或'D'）和决策理由文本
    """
    if attempt > max_retries:
        raise RuntimeError("❌ 多次尝试后仍无法获取合法 LLM 决策响应。")
    prompt = f"Your opponent's last move: {history_opponent[-1] if history_opponent else 'None'}.\nYour last move: {history_self[-1] if history_self else 'None'}.\nPlease choose your move: 'I choose to cooperate ...' or 'I choose to defect ...'."
    
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": scenario},
                {"role": "user", "content": prompt}
            ],
            temperature=LLM_TEMPERATURE,
            max_tokens=10,
            stream=False
        )
        move_text = response.choices[0].message.content.strip()
        print(f"[LLM 第 {attempt} 次响应]: {move_text}")

        if move_text.lower().startswith("i choose to cooperate"):
            return 'C'
        elif move_text.lower().startswith("i choose to defect"):
            return 'D'
        else:
            print(f"⚠️ 格式错误：{move_text}，重试中...")
            time.sleep(5)
            return llm_decision(history_self, history_opponent, attempt + 1, max_retries)

    except Exception as e:
        print(f"API 请求失败（第 {attempt} 次）：{e}，重试中...")
        time.sleep(5)
        return llm_decision(history_self, history_opponent, attempt + 1, max_retries)


# ==== ZD2策略参数计算 ====
def compute_zd2_parameters(chi=2, phi=0.1):
    R, T, P, S = 3, 5, 1, 0
    p1 = 1 - phi * (chi - 1) * (R - P) / (P - S)
    p2 = 1 - phi * (1 + chi * (T - P) / (P - S))
    p3 = phi * (chi + (T - P) / (P - S))
    p4 = 0
    return p1, p2, p3, p4

p1, p2, p3, p4 = compute_zd2_parameters()

# ==== ZD2策略决策 ====
def zd2_decision(history_self, history_opponent):
    if not history_self:
        return 'C'
    last_round = (history_self[-1], history_opponent[-1])
    if last_round == ('C', 'C'):
        return 'C' if random.random() < p1 else 'D'
    elif last_round == ('C', 'D'):
        return 'C' if random.random() < p2 else 'D'
    elif last_round == ('D', 'C'):
        return 'C' if random.random() < p3 else 'D'
    else:
        return 'C' if random.random() < p4 else 'D'

# ==== CTFT策略 ====
class CTFT:
    def __init__(self):
        self.state = 1  # 1:满意, 0:悔悟, -1:激怒

    def move(self, opponent_last, my_last):
        if self.state == 1:
            if opponent_last == 'D' and my_last == 'C':
                self.state = -1
            elif my_last == 'D' and opponent_last == 'C':
                self.state = 0
        elif self.state == -1:
            if opponent_last == 'C' and my_last == 'D':
                self.state = 1
        elif self.state == 0:
            if opponent_last == 'C' and my_last == 'C':
                self.state = 1
        return 'C' if self.state != -1 else 'D'

# ==== 经典策略实现 ====
def classic_strategy(name, history_self, history_opponent):
    """
    根据不同的经典策略名称返回相应的决策

    :param name: 策略名称，如'ALLD', 'ALLC', 'TFT'等
    :param history_self: 自己之前回合的决策历史列表
    :param history_opponent: 对手之前回合的决策历史列表
    :return: 返回当前回合的决策（'C'或'D'）
    """
    if name == 'ALLD':
        # 始终选择背叛策略
        return 'D'
    elif name == 'ALLC':
        # 始终选择合作策略
        return 'C'
    elif name == 'TFT':
        # 以牙还牙策略，初始合作，后续跟随对手上一轮的决策
        return 'C' if len(history_opponent) == 0 else history_opponent[-1]
    elif name == 'GTFT':
        # 慷慨以牙还牙策略
        if len(history_opponent) == 0:
            # 初始回合选择合作
            return 'C'
        opponent_last_move = history_opponent[-1]
        if opponent_last_move == 'D':
            # 对手上一轮背叛，有33%的概率选择合作
            if random.random() < 0.3:
                return 'C'
            else:
                return 'D'
        else:
            # 对手上一轮合作，自己也选择合作
            return 'C'
    elif name == 'CTFT':
        # 悔悟以牙还牙策略
        if len(history_opponent) == 0 or len(history_self) == 0:
            return 'C'  # 初始回合选择合作
        ctft = CTFT()
        return ctft.move(history_opponent[-1], history_self[-1])
    elif name == 'WSLS':
        # 赢则继续，输则改变策略
        if len(history_self) == 0:
            # 初始回合，随机选择合作或背叛
            return 'C'

        # 获取上一轮的行动和收益
        last_self_move = history_self[-1]
        last_opponent_move = history_opponent[-1]
        last_payoff = PAYOFF[(last_self_move, last_opponent_move)][0]

        if last_payoff >= R:
            # 如果收益不低于R，则保持当前行为
            return last_self_move
        else:
            # 否则，改变行为
            return 'C' if last_self_move == 'D' else 'D'
    elif name == 'Extortion':
        # 敲诈策略，使用ZD2策略进行决策
        return zd2_decision(history_self, history_opponent)
    elif name == 'HardMajority':
        # 强硬多数策略
        if len(history_opponent) == 0:
            # 初始回合随机选择（C/D各50%）
            return 'C'
        # 如果对手背叛次数超过总回合数的一半，自己选择背叛
        return 'D' if history_opponent.count('D') >= history_opponent.count('C') else 'C'
    elif name == 'SoftMajority':
        # 温和多数策略
        # 首次博弈选择合作
        if len(history_opponent) == 0:
            # 初始回合随机选择（C/D各50%）
            return 'C'
        # 如果对手背叛次数超过总回合数的60%，自己选择背叛
        return 'C' if history_opponent.count('D') <= history_opponent.count('C') else 'D'
    elif name == 'CURE':
        # CURE策略
        if len(history_opponent) == 0 or len(history_self) == 0:
            return 'C'  # 初始回合选择合作
        # 如果对手背叛次数减去自己背叛次数小于等于1，选择合作
        return 'C' if history_opponent.count('D') - history_self.count('D') <= 1 else 'D'
    else:
        # 若传入的策略名称未知，抛出异常
        raise ValueError("Unknown strategy!")

# ==== 修改单局游戏模拟函数 ====
def run_single_game(opponent_strategy, game_number):
    # 真实选择历史（用于计算合作率）
    history_llm = []      # LLM真实选择
    history_opp = []      # 对手真实选择
    
    # 受噪声影响的历史（用于策略决策）
    history_llm_noisy = []  # 传输给对手策略的LLM历史（受噪声影响）
    history_opp_noisy = []  # 传输给LLM的对手历史（受噪声影响）
    
    coop_llm = 0  # LLM真实合作次数
    coop_opp = 0  # 对手真实合作次数
    total_llm_score = 0
    total_opp_score = 0
    
    # 详细CSV的字段名
    fieldnames = ["Strategy", "GameNumber", "Round", "LLMMove", "OpponentMove", "NoisyLLMMove", "NoisyOpponentMove"]
    
    for round_num in range(1, N_ROUNDS + 1):
        # 1. 对手基于受噪声影响的LLM历史做出真实决策
        move_opp = classic_strategy(opponent_strategy, history_opp_noisy,history_llm_noisy)
        
        # 2. LLM基于受噪声影响的对手历史做出真实决策
        move_llm = llm_decision(history_llm_noisy, history_opp_noisy)
        
        # 3. 记录真实选择（用于计算合作率）
        history_llm.append(move_llm)
        history_opp.append(move_opp)
        coop_llm += (move_llm == 'C')
        coop_opp += (move_opp == 'C')
        
       # 4. 生成受噪声影响的选择（使用固定随机种子）
        random.seed(RANDOM_SEED + game_number * N_ROUNDS + round_num)  # 确保每次运行相同游戏相同回合的噪声相同
        # 双向噪声：C可能变为D，D也可能变为C
        noisy_llm = 'D' if (move_llm == 'C' and random.random() < ERROR_RATE) else \
            'C' if (move_llm == 'D' and random.random() < ERROR_RATE) else move_llm
            
        noisy_opp = 'D' if (move_opp == 'C' and random.random() < ERROR_RATE) else \
            'C' if (move_opp == 'D' and random.random() < ERROR_RATE) else move_opp
        
        # 5. 记录受噪声影响的历史（用于下一轮决策）
        history_llm_noisy.append(noisy_llm)
        history_opp_noisy.append(noisy_opp)
        
        # 6. 计算收益（基于受噪声影响的选择）
        llm_payoff, opp_payoff = PAYOFF[(noisy_llm, noisy_opp)]
        total_llm_score += llm_payoff
        total_opp_score += opp_payoff
        
        # 7. 写入详细CSV
        round_data = {
            "Strategy": opponent_strategy,
            "GameNumber": game_number,
            "Round": round_num,
            "LLMMove": move_llm,           # 真实选择
            "OpponentMove": move_opp,       # 真实选择
            "NoisyLLMMove": noisy_llm,      # 受噪声影响的选择
            "NoisyOpponentMove": noisy_opp  # 受噪声影响的选择
        }
        
        with open(DETAILED_CSV, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if f.tell() == 0:
                writer.writeheader()
            writer.writerow(round_data)

    return (
        coop_llm / N_ROUNDS,     # LLM真实合作率
        coop_opp / N_ROUNDS,     # 对手真实合作率
        total_llm_score / N_ROUNDS,
        total_opp_score / N_ROUNDS
    )

# ==== 修改两阶段方法，添加最大次数限制 ====
def two_stage_match(opponent_strategy):
    stage1_data = []
    print(f"策略:{opponent_strategy} | 阶段I开始（样本量{INITIAL_SAMPLE_SIZE}）")
    
    # 阶段I：限制不超过MAX_TOTAL_GAMES
    stage1_samples = min(INITIAL_SAMPLE_SIZE, MAX_TOTAL_GAMES)
    for n in range(1, stage1_samples + 1):
        p_llm, p_opp, s_llm, s_opp = run_single_game(opponent_strategy, n)
        stage1_data.append((p_llm, p_opp, s_llm, s_opp))
        print(f"阶段I游戏 {n}/{stage1_samples} 完成")

    # 计算阶段I统计量
    stage1_p_llm = [d[0] for d in stage1_data]
    mean_p = np.mean(stage1_p_llm) if stage1_p_llm else 0
    sample_var = np.var(stage1_p_llm, ddof=1) if len(stage1_p_llm) > 1 else 0
    
    # 计算所需样本量并应用最大限制
    required_n = int(np.ceil((Z_SCORE ** 2 * sample_var) / (PRECISION_THRESHOLD ** 2)))
    required_n = min(max(required_n, stage1_samples), MAX_TOTAL_GAMES)  # 不超过MAX_TOTAL_GAMES
    
    # 阶段II补充抽样
    if required_n > stage1_samples:
        print(f"策略:{opponent_strategy} | 阶段II开始（补充样本量{required_n - stage1_samples}）")
        for n in range(stage1_samples + 1, required_n + 1):
            if n > MAX_TOTAL_GAMES:
                print(f"已达到最大游戏次数限制({MAX_TOTAL_GAMES})，停止抽样")
                break
            p_llm, p_opp, s_llm, s_opp = run_single_game(opponent_strategy, n)
            stage1_data.append((p_llm, p_opp, s_llm, s_opp))
            print(f"阶段II游戏 {n - stage1_samples}/{required_n - stage1_samples} 完成")

    # 计算最终统计量
    total_games = len(stage1_data)
    mean_p_llm = np.mean([d[0] for d in stage1_data]) if stage1_data else 0
    mean_p_opp = np.mean([d[1] for d in stage1_data]) if stage1_data else 0
    mean_s_llm = np.mean([d[2] for d in stage1_data]) if stage1_data else 0
    mean_s_opp = np.mean([d[3] for d in stage1_data]) if stage1_data else 0
    std_error = np.std(stage1_p_llm, ddof=1) / np.sqrt(total_games) if total_games > 1 else 0
    ci_lower = mean_p_llm - Z_SCORE * std_error
    ci_upper = mean_p_llm + Z_SCORE * std_error

    return {
        "strategy": opponent_strategy,
        "total_games": total_games,
        "llm_coop": mean_p_llm,          # LLM原始决策的合作率
        "opp_coop": mean_p_opp,          # 对手原始决策的合作率
        "llm_score": mean_s_llm,         # 基于实际决策的收益
        "opp_score": mean_s_opp,         # 基于实际决策的收益
        "ci_lower": ci_lower,
        "ci_upper": ci_upper
    }

# ==== 主程序 ====
if __name__ == "__main__":
    # 初始化随机种子
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    
    # opponent_strategies = ['ALLD', 'ALLC', 'TFT', 'GTFT', 'CTFT', 'WSLS',
    #                        'Extortion', 'HardMajority', 'SoftMajority', 'CURE']

    opponent_strategies = [ 'TFT'] # 可扩展策略列表

    # 初始化汇总CSV文件
    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "Strategy", "TotalGames",
            "LLM_CooperationRate", "Opponent_CooperationRate",
            "LLM_AverageScore", "Opponent_AverageScore",
            "ConfidenceInterval"
        ])

    # 检查并删除已存在的详细CSV文件（避免追加旧数据）
    if os.path.exists(DETAILED_CSV):
        os.remove(DETAILED_CSV)

    all_results = []

    for strategy in opponent_strategies:
        print(f"\n=== 开始对抗策略: {strategy} ===")
        result = two_stage_match(strategy)
        all_results.append(result)

        # 写入汇总结果
        with open(SUMMARY_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                strategy,
                result["total_games"],
                f"{result['llm_coop']:.3f}",
                f"{result['opp_coop']:.3f}",
                f"{result['llm_score']:.3f}",
                f"{result['opp_score']:.3f}",
                f"[{result['ci_lower']:.3f}, {result['ci_upper']:.3f}]"
            ])

    print(f"\n✅ 所有策略处理完成！结果保存至：")
    print(f"- 详细轮次结果: {DETAILED_CSV}（同时记录原始决策和实际执行）")
    print(f"- 汇总结果: {SUMMARY_CSV}（LLM合作率基于原始决策）")