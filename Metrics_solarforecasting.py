from typing import Dict, List, Tuple
import numpy as np
import math
from matplotlib import pyplot as plt
from tslearn.metrics import dtw, dtw_path
class SwingingDoor:

    def init_snap(self, archived_pnt: Dict[str, float], value: float, trade_date: int,
                  time: int, positive_dev: float, negative_dev: float) -> Dict[str, float]:

        prev_val = float(archived_pnt['value'])
        prev_time = int(archived_pnt['time_value'])
        time = int(time)
        value = float(value)
        Smax = (value + positive_dev * value - prev_val) / (time - prev_time)
        Smin = (value - negative_dev * value - prev_val) / (time - prev_time)
        slope = (value - prev_val) / (time - prev_time)

        return {
            'value': value,
            'trade_date': trade_date,
            'time': time,
            'Smax': Smax,
            'Smin': Smin,
            'Slope': slope
        }

    def snap_archive(self, snapshot: Dict[str, float], is_snap: bool) -> Dict[str, float]:

        return {
            'value': snapshot['value'],
            'trade_date': snapshot['trade_date'],
            'time_value': snapshot['time'],
            'is_snap': is_snap,
        }

    def compress(self, time_series: List[float], kwh_sens: float, average_minutes: int) -> Tuple[List[float], List[int]]:

        archive: List[Dict[str, float]] = []
        res: List[float] = []
        times: List[int] = []

        POSITIVE_DEV: float = kwh_sens / 100
        NEGATIVE_DEV: float = POSITIVE_DEV

        counter: int = 0
        archive_count: int = 0

        for idx, val in enumerate(time_series):
            value: float = val
            trade_date: int = idx

            if counter == 0:
                # This is the header so we skip this iteration
                pass

            elif counter == 1:
                # This is the first data point, always added into archive
                archive.append({
                    'value': value,
                    'trade_date': trade_date,
                    'time_value': counter,
                    'is_snap': False,
                })
                archive_count += 1

            elif counter == 2:
                # This is the first snapshot that we will receive
                SNAPSHOT: Dict[str, float] = self.init_snap(
                    archive[archive_count - 1],
                    value,
                    trade_date,
                    counter,
                    POSITIVE_DEV,
                    NEGATIVE_DEV,
                )

                tmp_arch: Dict[str, float] = self.snap_archive(SNAPSHOT, False)
                archive.append(tmp_arch)
                res.append(tmp_arch['value'])
                times.append(tmp_arch['trade_date'])

            else:
                # Set up incoming value
                INCOMING: Dict[str, float] = self.init_snap(
                    archive[archive_count - 1],
                    value,
                    trade_date,
                    counter,
                    POSITIVE_DEV,
                    NEGATIVE_DEV,
                )
                if SNAPSHOT['Smin'] <= INCOMING['Slope'] <= SNAPSHOT['Smax']:
                    # It is within the filtration bounds, edit the INCOMING and
                    # set the SNAPSHOT. When editing INCOMING, make sure that the incoming
                    # slopes are not bigger than the current SNAPSHOT's slopes
                    INCOMING['Smax'] = min(SNAPSHOT['Smax'], INCOMING['Smax'])
                    INCOMING['Smin'] = max(SNAPSHOT['Smin'], INCOMING['Smin'])
                    SNAPSHOT = INCOMING
                else:
                    # It is outside the bounds so we must archive the current SNAPSHOT
                    # and init a new snap using this new archived point and INCOMING
                    tmp_arch: Dict[str, float] = self.snap_archive(SNAPSHOT, False)
                    archive.append(tmp_arch)
                    res.append(tmp_arch['value'])
                    times.append(tmp_arch['trade_date'])

                    archive_count += 1
                    SNAPSHOT = self.init_snap(
                        archive[archive_count - 1],
                        value,
                        trade_date,
                        counter,
                        POSITIVE_DEV,
                        NEGATIVE_DEV,
                    )

            counter += 1

        # Always add the latest point into the archive
        tmp_arch: Dict[str, float] = self.snap_archive(SNAPSHOT, True)
        archive.append(tmp_arch)
        res.append(tmp_arch['value'])
        times.append(tmp_arch['trade_date'])

        return self.average_per_hour(res, times, average_minutes)

    def average_per_hour(self, series: List[float], times: List[int], minutes: int) -> Tuple[List[float], List[int]]:
        res_times: List[int] = []
        res_series: List[float] = []

        end: int = times[-1]
        end = math.ceil(end / minutes) * minutes

        for i in range(0, end, minutes):
            min_time: int = i
            max_time: int = i + minutes
            tmp_observations: List[float] = []

            for idx, val in enumerate(times):
                if min_time <= val <= max_time:
                    tmp_observations.append(series[idx])

            res_times.extend([x for x in range(i, i + minutes)])

            if len(tmp_observations) < 1:
                tmp_observations = [res_series[-1]]

            avg: float = np.average(tmp_observations)
            res_series.extend([avg for _ in range(0, minutes)])

        return res_series, res_times



def calc_ramp_score(reference_x: List[float], reference_y: List[float], competing_x: List[float],
                    competing_y: List[float], avg_mins: int) -> float:

    t_min = reference_x[0]
    t_max = reference_x[-1]

    result = []
    for i in range(t_min, t_max, avg_mins):
        result.append(abs(np.trapz(y=competing_y[i:i + avg_mins], x=competing_x[i:i + avg_mins]) - np.trapz(
            y=reference_y[i:i + avg_mins], x=reference_x[i:i + avg_mins])))
    return (1 / (t_max - t_min)) * sum(result)


def get_ramp_score(title_y: str, ref_ls: List[float], model_ls: List[float], avg_mins: int = 60, sens: int = 80,
                   name: str = 'Compete', plot: bool = True) -> float:
    kwh_sens = sens
    kwh_sens = kwh_sens / 100

    swinging_door = SwingingDoor()
    y_reference, x_reference = swinging_door.compress(ref_ls, kwh_sens, avg_mins)
    y_compete, x_compete = swinging_door.compress(model_ls, kwh_sens, avg_mins)

    if plot:
        plt.plot(ref_ls, linestyle='-', color='gray', label='Actual')
        plt.plot(x_reference, y_reference, color='blue', linestyle=':', label='Ramp Observed')
        plt.plot(x_compete, y_compete, color='red', linestyle=':', label='Ramp Predicted')

        fz = 20
        plt.title('SwingDoor compression ' + str(name), fontsize=fz)
        plt.xlabel('Time in minutes', fontsize=fz)
        plt.ylabel(title_y, fontsize=fz)

        plt.legend()
        plt.show()
        plt.close()

    rs = calc_ramp_score(x_reference, y_reference, x_compete, y_compete, avg_mins)
    return rs


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))
def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)
def MAE(pred, true):
    return np.mean(np.abs(pred - true))
def MSE(pred, true):
    return np.mean((pred - true) ** 2)
def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))
def nRMSE(pred, true):
    return RMSE(pred, true) / np.mean(true)
def MBE(pred, true):
    return (pred - true).mean()
def nMBE(pred, true):
    return (pred - true).sum() / true.sum()
def DTW(pred, true):
    
    sim = np.sqrt(dtw(pred, true))
    return sim
def ramp_score(pred, true):
    ramp = get_ramp_score(
    title_y='GHI',  
    ref_ls=true,
    model_ls=pred,
    avg_mins=2,  
    sens=18,      
    name='CAB',  
    plot=True    
    )
    return ramp
def calculate_area(alignment_path, ideal_path):
    N = max(max(point) for point in ideal_path) + 1  
    area = 0
    for i in range(1, len(alignment_path)):
        prev_x, prev_y = alignment_path[i-1]
        curr_x, curr_y = alignment_path[i]
        ideal_prev_y = prev_x
        ideal_curr_y = curr_x
        height = curr_x - prev_x
        base1 = abs(prev_y - ideal_prev_y)
        base2 = abs(curr_y - ideal_curr_y)
        segment_area = (base1 + base2) * height / 2
        area += segment_area
    return area

def calculate_tdi(predict, target):

    N_output = len(predict)
    ideal_path = [(i, i) for i in range(N_output)]
    path, R = dtw_path(predict, target)
    area = calculate_area(path, ideal_path)
    tdi = area * 2 / ((N_output-1) * (N_output-1))
    return tdi, R

def metric(pred, true):
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    nrmse = nRMSE(pred, true)
    mbe = MBE(pred, true)
    nmbe = nMBE(pred, true)
    ramp = ramp_score(pred, true)
    cost = DTW(pred, true)
    tdi, R = calculate_tdi(pred, true)
    return {
        'RMSE': rmse,
        'MBE': mbe,
        'Ramp Score': ramp,
        'DTW': cost,
        'TDI': tdi,
    }
