"""Find the m_isDead offset in PlayerObject by comparing alive vs dead snapshots.

Usage:
    1. Launch Geometry Dash, enter a level
    2. Run: python scripts/calibrate_gd_offsets.py
    3. While ALIVE, press Enter to take snapshot 1
    4. Die in the game (don't restart yet!)
    5. While DEAD, press Enter to take snapshot 2
    6. Script shows all bytes that went 0->1 (candidate m_isDead offsets)
    7. Repeat to narrow down
"""

import ctypes
import ctypes.wintypes as wt
import struct
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from deepdash.gd_mem import (
    GDReader, _read_u64, _read_u8, OFF_PLAY_LAYER, OFF_PLAYER1,
)


def read_player_bytes(reader, start=0x000, end=0x1200):
    """Read a range of bytes from PlayerObject."""
    gm = _read_u64(reader.handle, reader.gm_ptr_addr)
    if gm == 0:
        return None, 0
    play_layer = _read_u64(reader.handle, gm + OFF_PLAY_LAYER)
    if play_layer == 0:
        return None, 0
    player1 = _read_u64(reader.handle, play_layer + OFF_PLAYER1)
    if player1 == 0:
        return None, 0
    size = end - start
    buf = ctypes.create_string_buffer(size)
    n = ctypes.c_size_t(0)
    ok = ctypes.windll.kernel32.ReadProcessMemory(
        reader.handle, ctypes.c_uint64(player1 + start), buf, size, ctypes.byref(n))
    if not ok:
        return None, player1
    return buf.raw, player1


def main():
    print("Connecting to Geometry Dash...")
    try:
        reader = GDReader()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print("Make sure Geometry Dash is running.")
        return

    print(f"Connected! PID={reader.pid}")
    start, end = 0x000, 0x1200
    candidates = None
    round_num = 0

    while True:
        round_num += 1
        print(f"\n=== Round {round_num} ===")

        # Snapshot ALIVE
        input("Stay ALIVE in the level, then press Enter...")
        alive_data, player_addr = read_player_bytes(reader, start, end)
        if alive_data is None:
            print("ERROR: Could not read player. Are you in a level?")
            continue
        print(f"  Alive snapshot: {len(alive_data)} bytes from PlayerObject @ 0x{player_addr:X}")

        # Snapshot DEAD
        input("Now DIE, and while dead press Enter...")
        dead_data, _ = read_player_bytes(reader, start, end)
        if dead_data is None:
            print("ERROR: Could not read player.")
            continue
        print(f"  Dead snapshot: {len(dead_data)} bytes")

        # Find bytes that went 0 -> 1
        new_candidates = set()
        for i in range(len(alive_data)):
            off = start + i
            if alive_data[i] == 0 and dead_data[i] == 1:
                new_candidates.add(off)

        if candidates is None:
            candidates = new_candidates
        else:
            candidates &= new_candidates

        print(f"\n  Bytes that went 0->1: {len(new_candidates)}")
        for off in sorted(new_candidates):
            marker = " <-- SURVIVED" if off in candidates else ""
            print(f"    0x{off:03X}{marker}")

        if candidates:
            print(f"\n  Consistent across all rounds: {len(candidates)}")
            for off in sorted(candidates):
                print(f"    0x{off:03X}")

        if len(candidates) <= 3:
            print(f"\n  >>> Likely m_isDead offset(s): {', '.join(f'0x{o:03X}' for o in sorted(candidates))}")
            print(f"  Update OFF_IS_DEAD in deepdash/gd_mem.py")
            resp = input("\n  Run another round to narrow down? (y/n) ")
            if resp.lower() != 'y':
                break
        else:
            print(f"\n  {len(candidates)} candidates remaining. Run more rounds to narrow down.")

    reader.close()
    print("Done.")


if __name__ == "__main__":
    main()
