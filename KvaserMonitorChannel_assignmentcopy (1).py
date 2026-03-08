# -*- coding: utf-8 -*-

import shutil
import argparse
import time
from collections import defaultdict, deque
import numpy as np

from colorama import Fore, Style, init
init()

from canlib import canlib

print("Number of CAN channels:", canlib.getNumberOfChannels())
bitrates = {
    '1M': canlib.canBITRATE_1M, '500K': canlib.canBITRATE_500K, '250K': canlib.canBITRATE_250K,
    '125K': canlib.canBITRATE_125K, '100K': canlib.canBITRATE_100K, '62K': canlib.canBITRATE_62K,
    '50K': canlib.canBITRATE_50K, '83K': canlib.canBITRATE_83K, '10K': canlib.canBITRATE_10K,
}

def profile_log_file(file_path):
    id_counts = defaultdict(int)
    id_timestamps = defaultdict(list)

    print("=" * 80)
    print(f"STEP 1: Profiling log file -> {file_path}")
    print("=" * 80)
    start_time = time.time()

    try:
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 1 and parts[1].isdigit():
                    try:
                        can_id = int(parts[1])
                        id_counts[can_id] += 1
                        timestamp = float(parts[-2])
                        id_timestamps[can_id].append(timestamp)
                    except (ValueError, IndexError):
                        pass
    except FileNotFoundError:
        print(f"\n[ERROR] Log file not found at path: {file_path}")
        return None, None

    print(f"-> Profiling finished in {time.time() - start_time:.2f} seconds.")
    profiled_ids = set(id_counts.keys())
    print(f"-> Found {len(profiled_ids)} unique CAN IDs in the log file.")

    profile_stats = {}
    for can_id in profiled_ids:
        ts = np.array(id_timestamps[can_id], dtype=float)
        mean_interval, std_interval = 0.0, 0.0
        if len(ts) > 1:
            intervals = np.diff(ts)
            mean_interval = float(intervals.mean())
            std_interval = float(intervals.std())
        profile_stats[can_id] = {'mean_interval': mean_interval, 'std_interval': std_interval, 'count': int(id_counts[can_id])}

    return profiled_ids, profile_stats

def printframe(frame, width, color_map):
    color = color_map.get(frame.id, Fore.WHITE)
    print(color, end="")
    form = '═^' + str(width - 1)
    print(format(f" Frame received (ID: {hex(frame.id)}) ", form))
    print("id:", frame.id)
    print("data:", frame.data.hex(' '))
    print("dlc:", frame.dlc)
    print("timestamp:", frame.timestamp)
    print("packetcount:", packetcount)
    print(Style.RESET_ALL, end="")

def pause_on_unknown_id():
    print(Fore.YELLOW + Style.BRIGHT + "PAUSED (unknown ID). Press ENTER to continue, or type 'q' then ENTER to quit.")
    try:
        s = input().strip().lower()
    except EOFError:
        return True
    return (s != "q")

def pause_on_burst():
    print(Fore.YELLOW + Style.BRIGHT + "PAUSED (burst/flood detected). Press ENTER to continue, or type 'q' then ENTER to quit.")
    try:
        s = input().strip().lower()
    except EOFError:
        return True
    return (s != "q")

def monitor_channel(channel_number, bitrate, profiled_ids, profile_stats):
    color_palette = [Fore.GREEN, Fore.YELLOW, Fore.BLUE, Fore.MAGENTA, Fore.CYAN, Fore.WHITE]
    id_to_color = {can_id: color_palette[i % len(color_palette)]
                   for i, can_id in enumerate(sorted(list(profiled_ids)))}

    last_timestamps = {}
    last_alert_wall = defaultdict(lambda: -1e9)

    unknown_seen = set()

    # Timing anomaly tuning (still based on device timestamps)
    Z_THRESHOLD = 6.0
    STD_FLOOR = 0.0005
    EARLY_FACTOR = 0.60
    LATE_FACTOR = 1.80
    ALERT_COOLDOWN_S = 0.25

    # Flood detection tuning (NOW uses perf_counter timebase)
    FLOOD_WINDOW_S = 0.050
    FLOOD_COUNT = 40
    FLOOD_CLEAR_COUNT = 10
    FLOOD_COOLDOWN_S = 0.50

    per_id_times = defaultdict(deque)   # id -> deque[arrival timestamps]
    flood_active_ids = set()
    last_flood_alert_wall = -1e9

    timeout = 0.5

    ch = canlib.openChannel(channel_number, canlib.canOPEN_ACCEPT_VIRTUAL, bitrate=bitrate)
    ch.setBusOutputControl(canlib.canDRIVER_NORMAL)
    ch.busOn()

    width, _ = shutil.get_terminal_size((80, 20))

    global packetcount
    packetcount = 0

    while True:
        try:
            frame = ch.read(timeout=int(timeout * 1000))
            packetcount += 1

            # High-resolution arrival time for burst/flood detection
            t_arrival = time.perf_counter()

            # ================= Flood detection (arrival time) =================
            dq = per_id_times[frame.id]
            dq.append(t_arrival)
            while dq and (t_arrival - dq[0]) > FLOOD_WINDOW_S:
                dq.popleft()

            in_flood_now = (len(dq) >= FLOOD_COUNT)
            if in_flood_now and (frame.id not in flood_active_ids):
                now_wall = time.time()
                if (now_wall - last_flood_alert_wall) >= FLOOD_COOLDOWN_S:
                    last_flood_alert_wall = now_wall
                    flood_active_ids.add(frame.id)

                    print(Fore.RED + Style.BRIGHT + "\n" + "!" * width)
                    print(Fore.RED + Style.BRIGHT + "CAN FLOOD DETECTED (single-ID burst / arbitration abuse style)")
                    print(Fore.RED + Style.BRIGHT + f"  Flooding ID: {frame.id} ({hex(frame.id)})")
                    print(Fore.RED + Style.BRIGHT + f"  Seen {len(dq)} frames within {FLOOD_WINDOW_S*1000:.0f}ms (arrival-time based)")
                    print(Fore.RED + Style.BRIGHT + "!" * width + "\n")

                    if not pause_on_burst():
                        break

            if frame.id in flood_active_ids and len(dq) <= FLOOD_CLEAR_COUNT:
                flood_active_ids.remove(frame.id)

            # ================= Unknown ID detection (pause) =================
            if frame.id not in profiled_ids:
                if frame.id not in unknown_seen:
                    unknown_seen.add(frame.id)
                    print(Fore.RED + Style.BRIGHT + "\n" + "!" * width)
                    print(Fore.RED + Style.BRIGHT + f"UNKNOWN CAN ID DETECTED (not in profile): {frame.id} ({hex(frame.id)})")
                    print(Fore.RED + Style.BRIGHT + f"  dlc={frame.dlc} data={frame.data.hex(' ')} timestamp={frame.timestamp}")
                    print(Fore.RED + Style.BRIGHT + "!" * width + "\n")
                    if not pause_on_unknown_id():
                        break
                continue

            # ================= Timing anomaly detection (no pause) =================
            current_time_s = frame.timestamp / 1_000_000.0
            if frame.id in last_timestamps:
                live_interval = current_time_s - last_timestamps[frame.id]
                stats = profile_stats.get(frame.id)
                if stats:
                    mean = stats.get('mean_interval', 0.0)
                    std = stats.get('std_interval', 0.0)
                    if mean > 0 and live_interval > 0:
                        std_eff = max(std, STD_FLOOR)
                        z = abs(live_interval - mean) / std_eff
                        early = live_interval < (EARLY_FACTOR * mean)
                        late = live_interval > (LATE_FACTOR * mean)

                        now_wall = time.time()
                        if (z > Z_THRESHOLD or early or late) and (now_wall - last_alert_wall[frame.id] >= ALERT_COOLDOWN_S):
                            last_alert_wall[frame.id] = now_wall
                            print(Fore.RED + Style.BRIGHT + f"\nTIMING ANOMALY for ID {frame.id} ({hex(frame.id)})")
                            print(Fore.RED + Style.BRIGHT + f"  interval={live_interval:.6f}s  mean={mean:.6f}s  std={std:.6f}s (eff={std_eff:.6f}s)")
                            print(Fore.RED + Style.BRIGHT + f"  z={z:.2f}  early={early}  late={late}\n")

            last_timestamps[frame.id] = current_time_s

            printframe(frame, width, id_to_color)

        except canlib.CanNoMsg:
            continue
        except KeyboardInterrupt:
            print(Style.RESET_ALL + "\nStop.")
            break
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CAN monitor with flood/burst pause + unknown ID pause.")
    default_log_file = r"C:\Users\holt1_22\Downloads\jazz2017_12_12_shop_home.txt"
    parser.add_argument('--profile-file', default=default_log_file)
    parser.add_argument('channel', type=int, default=0, nargs='?')
    parser.add_argument('--bitrate', '-b', default='500k')
    args = parser.parse_args()

    profiled_ids, profile_stats = profile_log_file(args.profile_file)
    if profiled_ids is not None:
        monitor_channel(args.channel, bitrates[args.bitrate.upper()], profiled_ids, profile_stats)

    print("\nScript finished.")