import numpy as np
import numba as nb
from marmachine import *
# Use nb.njit with cache to save compiled result between runs
@nb.njit(cache=True, fastmath=True)
def generate_bdata(changelist, initial_bufs, MARGA_BUFS, COUNTER_MAX):
    """
    JITted version: preallocate buffers sized by changelist length.
    Returns (times_arr, offsets_arr, indices_arr, values_arr).
    """
    # === part 1: same as before: build times/offsets/indices/values ===
    n = changelist.shape[0]

    # quick empty case
    if n == 0:
        return (np.empty(0, dtype=np.int64),
                np.empty(1, dtype=np.int64),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.uint16),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.uint32))
    # preallocated storage sized by n (upper bound)
    times_tmp = np.empty(n, dtype=np.int64)
    offsets_tmp = np.empty(n + 1, dtype=np.int64)  # offsets per unique timestep
    indices_tmp = np.empty(n, dtype=np.int64)
    values_tmp = np.empty(n, dtype=np.uint16)

    # state
    current_bufs = initial_bufs.copy()  # uint16 array
    change_masks = np.zeros(MARGA_BUFS, dtype=np.uint32)  # use wider for bit ops
    changed = np.zeros(MARGA_BUFS, dtype=np.uint8)  # 0/1 flags

    current_time = changelist[0, 0]

    time_count = 0
    idx_pos = 0
    offsets_tmp[0] = 0  # initial offset

    for i in range(n):
        time = changelist[i, 0]
        buf = int(changelist[i, 1])
        val = int(changelist[i, 2])
        mask = int(changelist[i, 3])

        if time != current_time:
            # close timestep: write time and emit changed buffers
            times_tmp[time_count] = current_time
            # iterate buffers and output the ones marked changed
            for b in range(MARGA_BUFS):
                if changed[b] != 0:
                    indices_tmp[idx_pos] = b
                    values_tmp[idx_pos] = current_bufs[b]
                    idx_pos += 1
            time_count += 1
            offsets_tmp[time_count] = idx_pos
            # reset masks & changed flags
            for b in range(MARGA_BUFS):
                change_masks[b] = 0
                changed[b] = 0
            current_time = time

        # compute buffer diff and apply
        buf_diff = (int(current_bufs[buf]) ^ val) & mask
        # keep assert for debug; in nopython this throws AssertionError if violated
        assert (buf_diff & int(change_masks[buf])) == 0 # "Tried to set a buffer to two values at once"
        if buf_diff == 0:
            # skip
            continue

        val_masked = val & mask
        old_val_unmasked = int(current_bufs[buf]) & (~mask)
        new_val = old_val_unmasked | val_masked
        change_masks[buf] = change_masks[buf] | mask
        current_bufs[buf] = np.uint16(new_val)
        changed[buf] = 1

    # close final timestep
    times_tmp[time_count] = current_time
    for b in range(MARGA_BUFS):
        if changed[b] != 0:
            indices_tmp[idx_pos] = b
            values_tmp[idx_pos] = current_bufs[b]
            idx_pos += 1
    time_count += 1
    offsets_tmp[time_count] = idx_pos

    # slice to actual sizes
    times_out = times_tmp[:time_count].copy()
    offsets_out = offsets_tmp[:time_count+1].copy()
    indices_out = indices_tmp[:idx_pos].copy()
    values_out = values_tmp[:idx_pos].copy()

    # === part 2: process time offsets (future_counts) ===
    T = times_out.shape[0]
    future_counts = np.zeros(T, dtype=np.int64)

    # iterate reversed pairs: i from T-1 down to 1 compare with i-1
    for i in range(T - 1, 0, -1):
        # current change is index i, previous is i-1
        ch_time = times_out[i]
        prev_time = times_out[i - 1]

        # size of indices array for 'current' change
        cur_count = offsets_out[i + 1] - offsets_out[i]  # offsets_out length = T+1

        # timestep between prev and cur in absolute times
        timestep = np.int32(ch_time - prev_time)  # match your original int32 behavior
        timediff = np.int32(cur_count - timestep)

        if timediff > 0:
            # move prev event into the past by timediff
            times_out[i - 1] = prev_time - int(timediff)
            # record how many of prev's buffers should be output in future
            future_counts[i - 1] = int(timediff)

    # convert to differential timesteps: replace times_out by deltas
    last_time = 0
    for i in range(T):
        ch0 = times_out[i]
        # delta = ch0 - last_time
        delta = ch0 - last_time
        times_out[i] = delta
        last_time = ch0
    
     # === part 3: statistics (counting) ===
    num_events = T
    total_buf_writes = 0
    est_waits = 0
    est_nops = 0
    for i in range(T):
        b_instrs = offsets_out[i+1] - offsets_out[i]
        total_buf_writes += b_instrs
        dtime = int(times_out[i])
        excess = dtime - b_instrs
        ex_tmp = excess
        # count waits
        while ex_tmp > 2:
            # conservative bound: wait_time can be up to COUNTER_MAX+3
            wait_time = ex_tmp
            if wait_time > (COUNTER_MAX + 3):
                wait_time = COUNTER_MAX + 3
            est_waits += 1
            ex_tmp -= wait_time
        if ex_tmp:
            # remaining short segments -> nops (use leftover ex_tmp)
            est_nops += ex_tmp

    est_total_inst = total_buf_writes + est_waits + est_nops + 1

    # === part 4: generate bdata instructions ===
    bdata_size = est_total_inst + MARGA_BUFS + 100  # small margin
    # ensure at least 1
    if bdata_size < 16:
        bdata_size = 16
    bdata = np.zeros(bdata_size, dtype=np.uint32)
    instr_idx = 0

    # helper: inline insta / instb encoding (as in your reference)
    # insta(instr, data) -> (instr << 24) | (data & 0xFFFFFF)
    # instb(tgt, delay, data) -> (IDATA<<24) | ((tgt & 0x7f) << 24) | (delay << 16) | (data & 0xFFFF)
    # (note: this mirrors your provided expression; top byte contains IDATA|tgt)
    for k in range(MARGA_BUFS):
        # reversed order: k=0 => last element
        ib = initial_bufs[MARGA_BUFS - 1 - k]
        # instb(MARGA_BUFS-1-k, k, ib)
        top = (IDATA << 24) | (( (MARGA_BUFS - 1 - k) & 0x7f) << 24)
        word = np.uint32(top | ((k & 0xFF) << 16) | (np.uint32(ib) & 0xFFFF))
        bdata[instr_idx] = word
        instr_idx += 1

    # timing helpers
    last_buf_time_left = np.zeros(MARGA_BUFS, dtype=np.int32)  # kept for parity (not used)
    buf_time_left = np.zeros(MARGA_BUFS, dtype=np.int32)

    # iterate events in order (times_out already deltas)
    for t in range(T):
        dtime = int(times_out[t])   # delta time since previous event
        s = offsets_out[t]
        e = offsets_out[t+1]
        b_instrs = e - s

        # soak up extra time
        excess_dtime = dtime - b_instrs
        ex_tmp = excess_dtime
        # first: consume large waits (IWAIT)
        while ex_tmp > 2:
            wait_time = ex_tmp
            if wait_time > (COUNTER_MAX + 3):
                wait_time = COUNTER_MAX + 3
            # encode insta(IWAIT, wait_time - 3)
            data = wait_time - 3
            word = np.uint32((IWAIT << 24) | (np.uint32(data) & 0xFFFFFF))
            bdata[instr_idx] = word
            instr_idx += 1
            ex_tmp -= wait_time

        # remaining small delays -> INOPs (use leftover ex_tmp)
        if ex_tmp > 0:
            for kk in range(ex_tmp):
                word = np.uint32((INOP << 24) | (0 & 0xFFFFFF))
                bdata[instr_idx] = word
                instr_idx += 1

        # apply reduction of previous buf_time_left by excess_dtime (clamp to 0)
        if excess_dtime != 0:
            for bb in range(MARGA_BUFS):
                bt = buf_time_left[bb] - excess_dtime
                if bt < 0:
                    bt = 0
                buf_time_left[bb] = bt

        # this_time_offset: how many buffers of previous event will be emitted in its future
        this_time_offset = int(future_counts[t])

        # loop through buffer writes for this event
        # Note: event indices are indices_out[s : e], values are values_out[s : e]
        for m in range(b_instrs):
            ind = int(indices_out[s + m])
            dat = int(values_out[s + m])  # 16-bit
            execution_delay = b_instrs - m - 1
            btli = int(buf_time_left[ind])
            # check if buffer empty at this cycle
            if btli <= m:
                extra_delay = execution_delay + this_time_offset
                # set buf_time_left[ind] = this_time_offset + b_instrs
                buf_time_left[ind] = this_time_offset + b_instrs
            else:
                extra_delay = this_time_offset - btli + b_instrs - 1
                # buf_time_left[ind] += extra_delay + 1
                buf_time_left[ind] = buf_time_left[ind] + extra_delay + 1

            # clamp extra_delay to fit in one byte (0..255). If larger you might want to emit IWAITs instead.
            if extra_delay < 0:
                extra_delay = 0
            if extra_delay > 255:
                extra_delay = 255

            # instb(ind, extra_delay, dat) encoding:
            top = (IDATA << 24) | ((ind & 0x7f) << 24)
            word = np.uint32(top | ((int(extra_delay) & 0xFF) << 16) | (np.uint32(dat) & 0xFFFF))
            bdata[instr_idx] = word
            instr_idx += 1

        # after writing this timestep's buffer instructions, decrement buf_time_left by b_instrs
        if b_instrs != 0:
            for bb in range(MARGA_BUFS):
                bt = buf_time_left[bb] - b_instrs
                if bt < 0:
                    bt = 0
                buf_time_left[bb] = bt

    # finish sequence
    word = np.uint32((IFINISH << 24) | (0 & 0xFFFFFF))
    bdata[instr_idx] = word
    instr_idx += 1

    # trim output bdata
    bdata_out = bdata[:instr_idx].copy()

    # Return everything (times/offsets/indices/values/future_counts/bdata)
    return times_out, offsets_out, indices_out, values_out, future_counts, bdata_out


# def cl2ol_fast(changelist):
#     initial_bufs = np.array([845, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint16)
#     npchangelist = np.array(changelist, dtype=np.int64)
#     MARGA_BUFS = 17
#     COUNTER_MAX = 0xFFFFFF
#     times, offsets, indices, values, _ = generate_bdata(npchangelist, initial_bufs, MARGA_BUFS, COUNTER_MAX)

#     unique_list = []
#     for i in range(times.shape[0]):
#         s = offsets[i]
#         e = offsets[i+1]
#         idxs = indices[s:e].astype(np.int64)
#         vals = values[s:e].astype(np.uint16)
#         unique_list.append((int(times[i]), idxs.copy(), vals.copy(), 0))
#     return unique_list


def bdata_fast(changelist, initial_bufs, MARGA_BUFS, COUNTER_MAX):
    initial_bufs = np.array(initial_bufs, dtype=np.uint16)
    npchangelist = np.array(changelist, dtype=np.int64)
    MARGA_BUFS = int(MARGA_BUFS)
    COUNTER_MAX = int(COUNTER_MAX)
    times, offsets, indices, values, future_cnts, bdata_fast_out = generate_bdata(npchangelist, initial_bufs, MARGA_BUFS, COUNTER_MAX)
    return bdata_fast_out


# ---- Example / test with the provided data ----
if __name__ == "__main__":
    changelist = np.array([(50, 9, 33205, 65535)])#, (50, 10, 2808, 32767), (50, 11, 33205, 65535), (50, 12, 2808, 32767), (50, 10, 32768, 32768), (50, 12, 32768, 32768), (50, 16, 64, 64), (50, 16, 128, 128), (50, 13, 33205, 65535), (50, 14, 2808, 32767), (50, 14, 32768, 32768), (50, 15, 256, 65280), (51, 10, 0, 32768), (51, 12, 0, 32768), (51, 14, 0, 32768), (100, 3, 1408, 65535), (100, 4, 1408, 65535), (100, 16, 16, 16), (100, 16, 32, 32), (101, 16, 0, 16), (101, 16, 0, 32), (466, 2, 1040, 65535), (466, 1, 1048, 65535), (468, 2, 528, 65535), (468, 1, 788, 65535), (469, 2, 272, 65535), (469, 1, 788, 65535), (737, 16, 0, 256), (737, 16, 0, 512), (737, 5, 0, 65535), (737, 6, 0, 65535), (737, 7, 0, 65535), (737, 8, 0, 65535), (737, 15, 0, 1), (737, 15, 0, 2), (737, 9, 33205, 65535), (737, 10, 2808, 32767), (737, 11, 33205, 65535), (737, 12, 2808, 32767), (737, 10, 0, 32768), (737, 12, 0, 32768), (62177, 10, 32768, 32768), (62178, 10, 0, 32768), (123617, 15, 1, 1), (178874, 15, 512, 65280), (237628, 2, 272, 65535), (237628, 1, 2760, 65535), (238856, 2, 272, 65535), (238856, 1, 4736, 65535), (240085, 2, 272, 65535), (240085, 1, 6712, 65535), (241314, 2, 272, 65535), (241314, 1, 8684, 65535), (242543, 2, 272, 65535), (242543, 1, 10660, 65535), (243772, 2, 272, 65535), (243772, 1, 12636, 65535), (245000, 2, 272, 65535), (245000, 1, 14608, 65535), ], dtype=np.int64)

    initial_bufs = np.array([845, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.uint16)
    MARGA_BUFS = 17
    COUNTER_MAX = 0xFFFFFF

    times, offsets, indices, values, future_counts, bdata = generate_bdata(changelist, initial_bufs, MARGA_BUFS, COUNTER_MAX)

    # Provide a friendly "unique" list similar to original representation for printing
    # We'll convert offsets/indices/values back into list-of-tuples: [time, indices_array, values_array, 0]
    # unique_list = []
    # for i in range(times.shape[0]):
    #     s = offsets[i]
    #     e = offsets[i+1]
    #     idxs = indices[s:e].astype(np.int64)
    #     vals = values[s:e].astype(np.uint16)
    #     future_cnts = future_counts[i]
    #     unique_list.append((int(times[i]), idxs.copy(), vals.copy(), future_cnts.copy()))
    #     # print(f"Time: {times[i]}, Indices: {idxs}, Values: {vals}, Future Count: {future_cnts}") 
    # print(times, offsets, indices, values )
    # print(unique_list)  # return for display

    print(bdata)

 