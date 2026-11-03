# -*- coding: utf-8 -*-
"""
airkeys_board_v10.py — 空气钢琴 · 板子端主程序（HDMI 显示版 · 视觉升级版）

运行环境：PYNQ-Z2 + PYNQ 3.1.1 镜像（Jupyter Notebook / 终端）
依赖：opencv (cv2), numpy, pynq（官方 base overlay）
功能：USB 摄像头 → ROI 掩膜只认纸键盘区(蓝框) → 透视校正对齐格子(可选) → 16 列帧差
      运动检测 → "手指下压"触发判定 → UDP 发送 "列号,dy" → 实时画面 + 分界线 + 高亮。

相对 8.9 定稿版（airkeys_board.py）的升级：
  任务2 协议升级：UDP 发 "列号,dy"（如 "8,5.5"），dy=下压幅度，供 PC 端做力度（轻重）映射；
  任务5 ROI 掩膜（默认开）：USE_ROI=True，蓝色框 = 唯一发声区，四周涂黑，框外运动不响；
  任务4 透视对齐：四角标定后设 PERSPECTIVE=True，cv2.warpPerspective 把纸键盘拉直贴合格子；
  和弦支持（左右分区）：把 16 列切成 0-7（左手区）与 8-15（右手区）两个完全独立的检测器，
         各自照原本的单冠军列逻辑跑一遍，各自有自己的质心历史；同帧最多发两个音 = 和弦；
  任务6 滑奏豁免：DEBOUNCE 0.15→0.05（快速连音）；横向移动 dx 极大时强制跳过 MAX_PIX
         上限，防止"滑奏"被当成影子拦截。

检测逻辑（8.9 定稿 · 单帧 EDGE 判定 + 左右分区和弦）：
  左右挥不响，只有手指往下按才响；dy > PRESS_DY 且 dy*X_TOL > dx；
  每列独立锁：哪一列抬手(LIFT_DY)或本区静帧(QUIET_FRAMES)才解锁哪一列；
  左右两区互不等待 → 左手按第3列的同时右手按第11列，两音同时发 = 和弦；
  同列仍受 DEBOUNCE 去抖，连弹同一列要抬一下才允许再响。

使用前必须：显示屏先开机再接板子再上电（HDMI 读 EDID）；上一内核占用摄像头时先
Kernel → Restart。电脑端对应脚本：src/airkeys_pc.py（numpy 合成音，支持力度/和弦）。
回退：想用 8.9 老版，跑原来 airkeys_board.py + airkeys_pc_midi.ps1 即可。
"""

import cv2
import numpy as np
import time
import socket
from pynq.overlays.base import BaseOverlay
from pynq.lib.video import VideoMode

# ---------- 0. 加载官方 overlay ----------
# 重要！显示屏必须先通电开机，再运行本代码
base = BaseOverlay("base.bit")
hdmi = base.video.hdmi_out

mode = VideoMode(640, 480, 24)   # 之前实测：很多显示器拒绝 1280x720，640x480 最稳
hdmi.configure(mode)
hdmi.start()
W, H = mode.width, mode.height
print("HDMI 输出分辨率:", W, "x", H)

# ---------- 1. 打开摄像头 ----------
cap = None
for attempt in range(10):
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap.isOpened():
        break
    time.sleep(1)
if not cap or not cap.isOpened():
    raise SystemExit("摄像头打不开")
# 用 MJPEG 压缩 + 限 30 帧：帧间隔 33ms，按下到出声的延迟比 15 帧少一半（顶不住 30 帧就改回 15，并把参数节里 dy/像素数类门槛翻倍调回）（不支持就自动忽略）
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_FPS, 30)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
# 自动曝光打开（0.75=自动）：画面暗、手识别不到时必须开；自动白平衡关（防整幅色偏误触发）
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
cap.set(cv2.CAP_PROP_AUTO_WB, 0)
# 只留最新一帧（1）：摄像头旧帧积压会让"按下到出声"越来越慢，这个设置是延迟最大的克星
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ---------- 2. UDP（发给 PC） ----------
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
PC_IP = "192.168.2.2"   # PC 的静态 IP
PC_PORT = 5005

# ---------- 2.5 ROI 掩膜（任务 5）+ 透视对齐（任务 4） ----------
# ★ 现场标定方法（只做一次）：
#   1) 纸上画 16 格铺在桌上，摄像头固定俯拍；
#   2) 看 HDMI 画面，把纸键盘"四个角"在画面里的像素坐标大致量出来
#      （左上→右上→右下→左下），填进下面 CALIB_POINTS 对应的 4 组数；
#   3) 蓝框默认就会画出来——先不动 PERSPECTIVE，把蓝框的四个角对准纸键盘
#      四边，蓝框内=唯一识别区（框外挥手不会响）；
#   4) 想要画面拉直贴合格子，再把 PERSPECTIVE 改成 True。
USE_ROI = True                            # 任务5 默认开：只识别蓝框内，四周涂黑（框外运动不发声）
PERSPECTIVE = False                       # 任务4：想拉直格子时改成 True
DRAW_QUAD = True                          # True：HDMI 上画蓝色识别框，便于现场调坐标
CALIB_POINTS = [                          # 纸键盘四角（左上→右上→右下→左下）
    (13, 60), (311, 60), (318, 186), (4, 186)
]

_warp_src = np.float32(CALIB_POINTS)
_warp_dst = np.float32([[0, 0], [319, 0], [319, 239], [0, 239]])
_WARP_M = cv2.getPerspectiveTransform(_warp_src, _warp_dst) if PERSPECTIVE else None
# ROI 掩膜：黑色背景下只保留纸键盘四边形（任务5：四周涂黑，屏蔽桌子/背景运动）
# 掩膜固定 320x240：主循环里已把每帧统一 resize 到 320x240，尺寸永远匹配
_roi_mask = np.zeros((240, 320), np.uint8)
if USE_ROI:
    cv2.fillPoly(_roi_mask,
                 [np.array(CALIB_POINTS, dtype=np.int32).reshape(-1, 1, 2)], 255)
_quad_pts = np.array(CALIB_POINTS, dtype=np.int32).reshape(-1, 1, 2)

# ---------- 3. 参数（沿用 8.9 定稿，仅 DEBOUNCE 0.15→0.05 支持滑奏快速连音） ----------
COLS = 16                 # 16 列 = 16 个音（B2~C5，中央C 在第8列）
DIFF_THRESHOLD = 20       # 画面变化多少算"动了"（静止乱响 → 25；手指不响 → 20）
MIN_MOTION = 100          # 全画面动起来的总量下限，低于它 = 噪声/没动，不响（静止乱响 → 200；手指不响 → 60）
MIN_PIX = 15              # 动得最多的那一列里，动的像素至少这么多才算真动作，太少=噪声（大拇指不响 → 10）
MAX_PIX = 800             # 一列里动的像素超过这个 = 影子/桌子晃/整条手臂，不算（影子还触发 → 400）
PRESS_DY = 0.8            # 手指往下移超过几像素算一次"按下"（慢按/大拇指不响 → 0.6；轻轻一碰就响 → 1.5）
X_TOL = 1.5               # 按下时允许带多少左右漂：条件 dy*X_TOL > dx（左右斜挥也响 → 0.8；斜着按总被拦 → 2.0）
LIFT_DY = -0.7           # 手指往上移超过几像素算"抬手"，抬完才允许按。负号=向上（连弹第二下不响 → -0.5）
QUIET_FRAMES = 3          # 连续几帧完全没动作 = 手已离开，同样允许按下一音
DEBOUNCE = 0.05           # 两次发音的最短间隔（秒）：任务6 由 0.15 改小，快速连音/滑奏不吞音（乱响 → 0.2）
GLIDE_DX = 60             # 任务6 滑奏豁免：横移超过这么多像素 = "正在滑"，强制跳过 MAX_PIX 上限
FLIP = 1                  # 音阶方向：反了就改 0

# 画面最边上几列，手会被画面边切掉一半动作，自动放宽标准
EDGE = [0, 1, 2, 3, 12, 13, 14, 15]

gamma = 0.45
lookut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype('uint8')

prev_gray = None
last_fire = [0.0] * COLS   # 每列上次发音时间（配合 DEBOUNCE 做去抖锁）
lift_ok = [True] * COLS    # 每列一把锁：本列抬过手才允许再响；不同列互不等待 → 和弦
# 左右两区各自一套"质心跟踪器"（下标 0=左区 0-7，1=右区 8-15），互不污染
H_px = [None, None]        # 上一帧本区冠军列内的质心 x
H_py = [None, None]        # 上一帧本区冠军列内的质心 y
H_best = [-1, -1]          # 上一帧本区的冠军列（全局列号）
H_quiet = [0, 0]           # 本区连续无有效动作帧数
quiet = 0                  # 全局连续无动作帧数（整幅安静 → 全部解锁）

ret, frame = cap.read()
while not ret:          # 第一帧可能读不到（摄像头刚启动），自动重试，不会崩
    ret, frame = cap.read()
print("摄像头实际输出:", frame.shape, "（程序内部统一按 320x240 处理，蓝框坐标也按 320x240 填）")

print("16 列视觉升级版 + HDMI：蓝框=唯一识别区(框外不响)，黄框=当前动最多列，红框=刚发声列。左右分区=可和弦")

while True:
    ret, frame = cap.read()
    if not ret:
        continue
    # 兜底：有些摄像头不听话，不按 320x240 出帧 → 统一压回 320x240：
    # 1) 掩膜(bitwise_and)要求和帧同尺寸，不压就会报错；
    # 2) 按列像素门槛(MIN_PIX/MAX_PIX/MIN_MOTION)都是按 320x240 调的，不压会跑偏。
    if frame.shape[1] != 320 or frame.shape[0] != 240:
        frame = cv2.resize(frame, (320, 240))

    # ---------- 任务5：ROI 掩膜（先屏蔽背景，再去拉直） ----------
    if USE_ROI:
        # 双保险：掩膜按当前帧实际尺寸自适应 resize，任何分辨率都不会再崩
        _mask = (cv2.resize(_roi_mask, (frame.shape[1], frame.shape[0]))
                 if _roi_mask.shape != frame.shape[:2] else _roi_mask)
        frame = cv2.bitwise_and(frame, frame, mask=_mask)

    # ---------- 任务4：透视对齐（把纸键盘拉直成"横平竖直"，16 列贴合格子） ----------
    if PERSPECTIVE and _WARP_M is not None:
        frame = cv2.warpPerspective(frame, _WARP_M, (320, 240))

    hit_cols = []            # 本帧发声的列（最多两个：左区一个、右区一个）
    best = -1                # 全局冠军列，仅用于黄框显示
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.LUT(gray, lookut)              # 提亮后再做帧差，动作信号才明显
    gray = cv2.GaussianBlur(gray, (5, 5), 0)  # 糊一下：传感器"沙沙"噪声被抹掉，手指动作不受影响
    if prev_gray is not None:
        diff = cv2.absdiff(gray, prev_gray)
        _, motion = cv2.threshold(diff, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
        motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))  # 抹掉孤零零的噪声点
        col_w = motion.shape[1] // COLS        # 每列宽度（透视拉直后 = 纸键盘 1/16）
        # 每列的运动量；best = 整幅动得最多的列（只用于黄框显示）
        energies = [int(np.count_nonzero(motion[:, i * col_w:(i + 1) * col_w])) for i in range(COLS)]
        best = int(np.argmax(energies))
        total = sum(energies)
        now = time.time()

        # 总开关：整幅真有动作才进去判（和原版一致，保证灵敏度和原来一样）
        if total > MIN_MOTION:
            quiet = 0
            # ========== 左右两区各跑一遍原版逻辑 ==========
            # 左区列 0-7，右区列 8-15；每区只在自己这 8 列里挑冠军列、算质心、判下压。
            # 每区有自己的质心历史 H_px/H_py/H_best，左手动不会干扰右手的位置判断。
            for h in range(2):
                lo = h * 8
                hi = lo + 8
                half_e = energies[lo:hi]
                hb = int(np.argmax(half_e)) + lo      # 本区冠军列（全局列号）

                # 只在本区冠军列里取动作质心（整幅质心会被另一只手稀释，方向判断会乱跳）
                ys, xs = np.nonzero(motion[:, hb * col_w:(hb + 1) * col_w])
                col_cnt = ys.size
                # 手跨在列分界线上、本区冠军列来回跳时：把上一帧那列也一起算，
                # 否则列归属一变，左右位移会假跳一大截，把下压误拦下来
                if (H_px[h] is not None and lo <= H_best[h] < hi
                        and abs(H_best[h] - hb) == 1):
                    y2, x2 = np.nonzero(motion[:, H_best[h] * col_w:(H_best[h] + 1) * col_w])
                    if y2.size:
                        ys = np.concatenate([ys, y2])
                        xs = np.concatenate([xs, x2 + (H_best[h] - hb) * col_w])

                # 边缘列放宽一些（手被画面边挡掉一部分动作，信号天生弱一点）
                edge = hb in EDGE
                min_pix = MIN_PIX // 2 if edge else MIN_PIX
                max_pix = MAX_PIX * 2 if edge else MAX_PIX
                press_dy = PRESS_DY / 1.5 if edge else PRESS_DY
                x_tol = X_TOL * 2 if edge else X_TOL

                if col_cnt > min_pix:      # 本区真有动作（噪声量不够的区整段跳过，不更新位置）
                    H_quiet[h] = 0
                    cx = xs.mean() + hb * col_w   # 换成全画面坐标，两帧之间才好比较
                    cy = ys.mean()
                    if H_px[h] is not None:
                        dy = cy - H_py[h]         # 往下为正、往上为负
                        dx = abs(cx - H_px[h])
                        # 任务6 滑奏豁免：dx 极大 = 手指在快速横向滑，跳过"一列像素超上限=影子"的拦截
                        not_shadow = (col_cnt < max_pix) or (dx > GLIDE_DX)
                        # 下压判定：往下移得够多，且下移量明显盖过左右漂（x_tol 就是左右容忍度）
                        if (dy > press_dy and dy * x_tol > dx and lift_ok[hb]
                                and now - last_fire[hb] > DEBOUNCE and not_shadow):
                            last_fire[hb] = now
                            lift_ok[hb] = False
                            hit_cols.append(hb)
                            out = (15 - hb) if FLIP else hb
                            # 任务2：连 dy 一起发，PC 端据此做"音量轻重"
                            sock.sendto(("%d,%.2f" % (out, dy)).encode(), (PC_IP, PC_PORT))
                            print(">>> [%s区] 第" % ("左" if h == 0 else "右"), hb,
                                  "列按下 (dy=%.1f dx=%.1f px=%d) → 发:%d,%.2f" % (dy, dx, col_cnt, out, dy))
                        elif dy < LIFT_DY:        # 手指上抬 = 这一列解锁，本列允许再按
                            lift_ok[hb] = True
                    # 每帧都更新本区位置（"跟踪器"模式：位置不能冻结，否则下次 dy 会假跳变）
                    H_px[h], H_py[h] = cx, cy
                    H_best[h] = hb
                else:
                    # 本区这几帧没像样的动作：连续够久 = 手已离开本区 → 清位置 + 只解锁本区
                    H_quiet[h] += 1
                    if H_quiet[h] >= QUIET_FRAMES:
                        H_px[h] = H_py[h] = None
                        H_best[h] = -1
                        for c in range(lo, hi):
                            lift_ok[c] = True
        else:
            quiet += 1
            if quiet >= QUIET_FRAMES:   # 整幅连续几帧没动作 = 手已离开，所有列解锁
                lift_ok = [True] * COLS
                H_px = [None, None]
                H_py = [None, None]
                H_best = [-1, -1]
                H_quiet = [0, 0]
    prev_gray = gray

    # ---------- 4. 画线并输出到 HDMI ----------
    disp = frame.copy()
    if USE_ROI and DRAW_QUAD:              # 蓝色框 = 唯一发声区（任务5）
        if PERSPECTIVE:                    # 拉直后整幅就是纸键盘 → 蓝框画在四边
            cv2.rectangle(disp, (0, 0), (disp.shape[1] - 1, disp.shape[0] - 1), (255, 0, 0), 1)
        else:                              # 标定阶段：蓝框画在原始画面上 = 真正只响这个范围
            cv2.polylines(disp, [_quad_pts], True, (255, 0, 0), 1)
    # 左右分区中线（画粗一点，方便看左右手该放哪）——可选视觉提示
    cv2.line(disp, (disp.shape[1] // 2, 0), (disp.shape[1] // 2, disp.shape[0] - 1), (0, 128, 255), 2)
    for i in range(1, COLS):                        # 绿色列分界线
        x = i * disp.shape[1] // COLS
        cv2.line(disp, (x, 0), (x, disp.shape[0] - 1), (0, 255, 0), 1)
    if best >= 0:                                   # 动得最多的列画黄框
        cv2.rectangle(disp, (best * disp.shape[1] // COLS, 0),
                      ((best + 1) * disp.shape[1] // COLS, disp.shape[0] - 1), (0, 255, 255), 1)
    for hc in hit_cols:                             # 刚发声的列画红框（可能左右各一个）
        cv2.rectangle(disp, (hc * disp.shape[1] // COLS, 0),
                      ((hc + 1) * disp.shape[1] // COLS, disp.shape[0] - 1), (0, 0, 255), 3)
    disp = cv2.resize(disp, (W, H))                 # 必须缩放到 HDMI 分辨率
    disp = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)    # 摄像头出的是 BGR，转成 RGB 颜色才正常（红框才显示红色）
    outframe = hdmi.newframe()                      # PYNQ 3.1.1：先 newframe 拿一帧空画布
    outframe[:] = disp                              # 把处理好的画面填进画布
    hdmi.writeframe(outframe)                       # 再送出去显示（newframe 和 writeframe 必须成对，别多调）

