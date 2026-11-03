# -*- coding: utf-8 -*-
"""
airkeys_pc.py — 空气钢琴 · 电脑端发声脚本 v10（numpy 合成音版，替代旧 MIDI 脚本）

任务 1（双音色频率合成）：不再依赖 Windows MIDI 合成器，直接用 numpy 按"正弦波叠加
  配方"实时合成声音。提供两种音色：
    --timbre piano   钢琴音：基音 + 低次谐波，尾巴长（像琴弦慢慢共振收掉）
    --timbre box     八音盒音：基音衰减快 + 高次谐波亮，叮的一声短促清脆
  运行中按键盘 1 / 2 可随时切换，不用重启。

任务 2（音量轻重·力度感应）：板子发来的消息升级为 "列号,dy"（如 "8,5.5"），dy 是
  手指下压幅度（像素）。这里把 dy 在 [PRESS_DY, FULL_DY] 一段线性映射成 0.1~1.0 的
  音量系数，直接乘在合成波形上：按得深声音大、按得浅声音小。

任务 3（多线程和弦）：每个音在独立线程里完成波形合成后加进"声部列表"，声卡回调
  逐块把所有声部叠加输出，互不阻塞；全程不使用 sd.play()/sd.wait()。
  双手同时按下 = 列表里同时多个音 = 真和弦。

运行：
    .venv\\Scripts\\python src\\airkeys_pc.py            (默认钢琴音)
    .venv\\Scripts\\python src\\airkeys_pc.py --timbre box
    .venv\\Scripts\\python src\\airkeys_pc.py --port 5005
依赖：Python 3.8+；pip install numpy sounddevice（本项目已备好 .venv 虚拟环境）
注意：首次运行时放行防火墙 UDP 5005；板子端对应脚本 src/airkeys_board_v10.py（发 "列号,dy"）。
旧协议兼容：如果收到纯列号（老板子），按中等力度 0.55 发声。
"""

import argparse
import socket
import threading
import time

import numpy as np
import sounddevice as sd

# ---------- 音频常量 ----------
SR = 44100                          # 采样率（Hz）

# 正在发声的"声部"列表：每个元素 = [波形数组, 已播放位置]
# 多线程和弦 = 每按一键就加一个声部进来；声卡回调按块把所有声部逐帧叠加输出，
# 播完的声部自动丢弃。绝不使用"按时间戳定位"的混音环——那会跟声卡时钟对不上。
voices = []
voices_lock = threading.Lock()


# ---------- 16 音 B2~C5（与旧 MIDI 47~72 一致，中央 C4=60 在第 8 列） ----------
NOTES = [47, 48, 50, 52, 53, 55, 57, 59, 60, 62, 64, 65, 67, 69, 71, 72]


def freq_of(midi):
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


FREQS = [freq_of(n) for n in NOTES]

# ---------- 音色配方：每个分波 = (倍频数, 相对振幅, 衰减时间常数 tau/秒) ----------
# 钢琴：基音 + 低次谐波，尾巴长（像琴弦慢慢共振收掉）
PIANO = dict(
    name="钢琴",
    partials=[(1, 1.00, 0.90), (2, 0.50, 0.55), (3, 0.30, 0.40), (4, 0.15, 0.30)],
    dur=1.2,
)
# 八音盒：基音衰减快 + 高次谐波亮，叮的一声短促清脆
BOX = dict(
    name="八音盒",
    partials=[(1, 1.00, 0.35), (2, 0.10, 0.20), (3, 0.55, 0.22), (4, 0.35, 0.12), (5, 0.15, 0.08)],
    dur=0.9,
)
TIMBRES = {"piano": PIANO, "box": BOX}
# 当前音色（1/2 键随时切，只影响接下来按的音）
CUR_TIMBRE = [PIANO]


# ---------- 波形合成：按配方把多个正弦波叠加，再套指数衰减包络 ----------
def synth(freq, vel, recipe):
    n = int(recipe["dur"] * SR)
    t = np.arange(n) / SR
    attack = 1.0 - np.exp(-t * 30.0)      # 几毫秒的渐入，避免起音"咔哒"爆音
    buf = np.zeros(n, dtype=np.float32)
    for mult, amp, tau in recipe["partials"]:
        env = np.exp(-t / tau) * attack
        buf += amp * env * np.sin(2 * np.pi * freq * mult * t)
    peak = np.max(np.abs(buf))
    if peak > 0:
        buf /= peak                       # 归一化到峰值 1
    rel = min(int(0.02 * SR), n)          # 结尾 20ms 线性渐隐，避免掐断时的"咔哒"声
    buf[-rel:] *= np.linspace(1.0, 0.0, rel)
    return buf * vel                      # 乘音量系数（0.1~1.0 轻响）


# ---------- 和弦：每个音一个线程，合成后加进"声部列表" ----------
def play_note(col, vel):
    sig = synth(FREQS[col], vel, CUR_TIMBRE[0])  # 重活（整段合成）都在这个音自己的线程里干
    with voices_lock:
        voices.append([sig, 0])                  # 加进声部列表，回调会把它叠出来


# ---------- 音频回调：把列表里所有声部的同一段都叠加进声道 ----------
def on_audio(outdata, frames, time_info, status):
    out = np.zeros(frames, dtype=np.float32)
    with voices_lock:
        alive = []
        for vo in voices:
            buf, pos = vo
            take = min(frames, len(buf) - pos)   # 这一块要取多少样本
            if take > 0:
                out[:take] += buf[pos:pos + take]
                vo[1] += take                    # 推进该声部播放位置
            if vo[1] < len(buf):
                alive.append(vo)                 # 没播完，下个 block 继续
        voices[:] = alive
    outdata[:, 0] = np.tanh(out.astype(np.float64)) * 0.9   # 软限幅：多音叠加防破音


# ---------- 力度映射：dy → 0.1~1.0 ----------
PRESS_DY_REF = 0.8     # 板子的触发门槛（低于它不会发这条消息）
FULL_DY = 3.0          # dy 达到这个值 = 满力度（可现场按手感调）


def vel_from_dy(dy):
    ratio = (dy - PRESS_DY_REF) / (FULL_DY - PRESS_DY_REF)
    return float(np.clip(0.1 + 0.9 * ratio, 0.1, 1.0))


# ---------- UDP 接收循环：解析 "列号,dy" → 开音线程 ----------
def udp_loop(port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("", port))
    sock.settimeout(1.0)
    print(f"等待板子消息 (UDP {port})... Ctrl+C 退出。当前音色：{CUR_TIMBRE[0]['name']}（按 1=钢琴 2=八音盒）")
    while True:
        try:
            data, _ = sock.recvfrom(64)
        except socket.timeout:
            continue
        try:
            txt = data.decode("utf-8").strip()
            if "," in txt:                      # 新协议：列号,dy
                col_s, dy_s = txt.split(",")
                col = int(col_s)
                dy = float(dy_s)
            else:                               # 老协议：纯列号 → 按中等力度
                col = int(txt)
                dy = 1.8
            if 0 <= col < 16:
                vel = vel_from_dy(dy)
                # 每个音独立线程 → 互不等待 → 同时按就是和弦
                threading.Thread(target=play_note, args=(col, vel), daemon=True).start()
                print(f"列 {col:2d}  dy={dy:5.2f}  力度={vel:.2f}  {FREQS[col]:7.2f} Hz  [{CUR_TIMBRE[0]['name']}]")
        except (ValueError, TypeError):
            pass


# ---------- 键盘 1/2 切换音色（Windows msvcrt，无回车） ----------
def key_thread():
    import msvcrt
    while True:
        try:
            k = msvcrt.getwch().lower()
        except Exception:                       # 非交互终端里读不到键就安静跳过
            time.sleep(0.2)
            continue
        if k == "1":
            CUR_TIMBRE[0] = TIMBRES["piano"]
            print("[音色] 切换为 钢琴")
        elif k == "2":
            CUR_TIMBRE[0] = TIMBRES["box"]
            print("[音色] 切换为 八音盒")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="空气钢琴 PC 端发声（numpy 合成音）")
    ap.add_argument("--timbre", default="piano", choices=["piano", "box"],
                    help="初始音色：piano=钢琴（默认）/ box=八音盒")
    ap.add_argument("--port", type=int, default=5005, help="UDP 监听端口")
    args = ap.parse_args()

    CUR_TIMBRE[0] = TIMBRES[args.timbre]
    threading.Thread(target=key_thread, daemon=True).start()
    with sd.OutputStream(samplerate=SR, channels=1, dtype="float32",
                         blocksize=512, callback=on_audio):
        udp_loop(args.port)
