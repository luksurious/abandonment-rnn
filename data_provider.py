import math
import numpy as np
import glob
import pandas as pd
from xml.dom import minidom
import io
import re
import json
from urllib.parse import unquote
import pickle
from scipy.spatial import distance

from keras_preprocessing.sequence import pad_sequences

PATH = 'kge/'
VERBOSE = False


def load_data():
    was_user_info_loaded = True
    try:
        user_info = pickle.load(open("data/user_info_filtered.pkl", "rb"))
    except Exception:
        user_info = read_kge_log()
        was_user_info_loaded = False

    was_data_loaded = True
    try:
        data_frames = pickle.load(open("data/mouse_logs_filtered.pkl", "rb"))
    except Exception:
        data_frames = read_csv_logs(user_info)
        was_data_loaded = False

    if not was_data_loaded:
        data_frames = filter_data_frames(data_frames, user_info, True)
        # user_info = filter_user_info(user_info, data_frames)
    if not was_user_info_loaded:
        user_info = filter_user_info(user_info, data_frames)

    return user_info, data_frames


def read_kge_log():
    try:
        user_info = pickle.load(open("data/user_info.pkl", "rb"))
        return user_info
    except Exception:
        pass

    with open(PATH + "kge.log", 'r') as f:
        fo = io.StringIO()
        data = f.readlines()
        fo.writelines(line.replace('undefined', 'N/A')
                      .replace('no user info provided yet', 'no-user-info-provided-yet')
                      .replace(", ", ",") for line in data)
        fo.seek(0)

    user_info = pd.read_csv(fo, sep=' ', header=None, na_values=['None'],
                            names=['timestamp', 'IP', 'sessionId', 'queryId', 'userId', 'moduleVisibility',
                                   'moduleWasNoticed', 'moduleWasUseful', 'moduleWasFaster', 'URL'])

    user_info['moduleWasUsefulBinary'] = 0
    user_info.loc[user_info['moduleWasUseful'] > 3, 'moduleWasUsefulBinary'] = 1
    user_info['moduleWasFasterBinary'] = 0
    user_info.loc[user_info['moduleWasFaster'] > 3, 'moduleWasFasterBinary'] = 1

    pickle.dump(user_info, open("data/user_info.pkl", "wb"))

    return user_info


def read_csv_logs(user_info):
    try:
        data_frames = pickle.load(open("data/mouse_logs.pkl", "rb"))
        return data_frames
    except Exception:
        pass

    # read from scratch
    user_ids = user_info["userId"]

    data_frames = []

    quote_replacement = '|'

    error_count = 0

    file_user_map = {}

    user_ids_set = set()

    pattern = PATH + 'cursorlogs/*.csv'
    for file in glob.glob(pattern):
        print(".", end='')

        xml_file = file.replace('.csv', '.xml')
        try:
            mydoc = minidom.parse(xml_file)
            task_data = mydoc.getElementsByTagName('task')[0].firstChild.data.split('-')
        except Exception as e:
            print(f"\nError in {file}: {e}")
            error_count += 1
            continue
        file_id = re.match(".+/(.+).csv", file).group(1)

        if file_user_map.get(task_data[2], False) is False:
            file_user_map[task_data[2]] = {}

        if file_user_map[task_data[2]].get(task_data[1], False) is False:
            file_user_map[task_data[2]][task_data[1]] = {}

        file_user_map[task_data[2]][task_data[1]][file_id] = '-'.join(task_data)

        user_ids_set.add(task_data[2])

        if not (user_ids == task_data[2]).any():
            # print("User %s not found in ground truth data" % task_data[2])
            continue

        with open(file, 'r') as f:
            fo = io.StringIO()
            data = f.readlines()
            fo.writelines(line.replace(quote_replacement, quote_replacement + quote_replacement)
                          .replace('/', '%s/' % quote_replacement, 1)
                          .replace(' {', "%s %s{" % (quote_replacement, quote_replacement), 1)
                          .replace('} {', "}%s %s{" % (quote_replacement, quote_replacement), 1)
                          .replace('}\n', '}%s\n' % quote_replacement, 1) for line in data)
            fo.seek(0)
        try:
            df = pd.read_csv(fo, sep=' ', quotechar=quote_replacement, quoting=0)

            if len(df) == 0:
                continue

            url = mydoc.getElementsByTagName('url')[0].firstChild.data
            viewport = mydoc.getElementsByTagName('window')[0].firstChild.data
            document = mydoc.getElementsByTagName('document')[0].firstChild.data

            if "search.yahoo.com" not in url:
                continue

            df["session"] = task_data[0]
            df["query"] = task_data[1]
            df["user"] = task_data[2]
            df["module_vis"] = task_data[3]
            df["file"] = file_id
            df["url"] = url
            df["viewport"] = viewport
            df["document"] = document
            data_frames.append(df)
        except Exception as e:
            print(f"\nError in {file}: {e}")
            error_count += 1

    print("\nWith %s got %d errors" % (quote_replacement, error_count))

    with open('data/user-file-map.json', 'w') as fp:
        json.dump(file_user_map, fp)

    pickle.dump(data_frames, open("data/mouse_logs.pkl", "wb"))

    return data_frames


def filter_data_frames(data_frames, user_info, only_final=True):
    data_no_clicks = [df for df in data_frames if len(df[df["event"] == "click"]) == 0
                      and len(df[df["event"] == "mousemove"]) > 0]

    print("Frames without clicks: %d" % len(data_no_clicks))

    data_nc_with_km = [df for df in data_no_clicks if df["module_vis"].values[0] == '1']
    data_nc_without_km = [df for df in data_no_clicks if df["module_vis"].values[0] == '0']

    print("Frames without clicks with KM: %d" % len(data_nc_with_km))
    # print("Frames without clicks without KM: %d" % len(data_nc_without_km))

    data_nc_with_km_sr = [df for df in data_nc_with_km if "search.yahoo.com" in df["url"].values[0]]
    data_nc_without_km_sr = [df for df in data_nc_without_km if "search.yahoo.com" in df["url"].values[0]]

    # print("Frames without clicks with KM on search page: %d" % len(data_nc_with_km_sr))
    # print("Frames without clicks without KM on search page: %d" % len(data_nc_without_km_sr))

    data = (data_nc_with_km_sr, data_nc_without_km_sr)
    if only_final:
        data_nc_with_km_sr = get_data_last_only(data_nc_with_km_sr)
        data_nc_with_km_sr = [df for df in data_nc_with_km_sr if len(df[df["event"] == "mousemove"]) > 1]
        users_tasks, _ = map_user_tasks(data_nc_with_km_sr, user_info)
        data_nc_with_km_sr = filter_by_user_info(user_info, data_nc_with_km_sr, users_tasks)

        data = [df.drop(columns=['xpath', 'attrs', 'cursor', 'url', 'file']) for df in data_nc_with_km_sr]
    pickle.dump(data, open("data/mouse_logs_filtered.pkl", "wb"))

    return data


def filter_user_info(user_info, df1):
    _, filter1 = map_user_tasks(df1, user_info)

    user_info_filtered = user_info[filter1].reset_index()

    for col in ['IP', 'index', 'timestamp', 'moduleWasFaster', 'moduleWasUseful', 'URL']:
        if col in user_info_filtered.columns:
            user_info_filtered.drop(columns=[col], inplace=True)

    pickle.dump(user_info_filtered, open("data/user_info_filtered.pkl", "wb"))

    return user_info_filtered


def get_data_last_only(data_nc_with_km_sr):
    users_time = {"%s%s%s" % (df["user"].values[0], df["query"].values[0], df["session"].values[0]): 0
                  for df in data_nc_with_km_sr}

    # sort reverse by file
    data_nc_with_km_sr = sorted(data_nc_with_km_sr, key=lambda df: df["file"].values[0], reverse=True)

    data_nc_with_km_sr_last = []
    for df in data_nc_with_km_sr:
        task_id = "%s%s%s" % (df["user"].values[0], df["query"].values[0], df["session"].values[0])

        if users_time[task_id] < int(df["file"].values[0]):
            data_nc_with_km_sr_last.append(df)
            users_time[task_id] = int(df["file"].values[0])

    return data_nc_with_km_sr_last


def get_solo_only(data_nc_with_km_sr):
    task_counts_wkm = \
        pd.DataFrame({"user": ["%s%s%s" % (df["user"].values[0], df["query"].values[0], df["session"].values[0])
                               for df in data_nc_with_km_sr]})["user"].value_counts()

    data_nc_with_km_sr_solo = []
    for df in data_nc_with_km_sr:
        task_id = "%s%s%s" % (df["user"].values[0], df["query"].values[0], df["session"].values[0])

        if task_counts_wkm[task_counts_wkm.keys() == task_id][0] == 1:
            data_nc_with_km_sr_solo.append(df)

    return data_nc_with_km_sr_solo


def map_user_tasks(data_frames, user_info):
    # filter user info
    users_tasks = {df["user"].values[0] + df["query"].values[0] + df["session"].values[0]: -1
                   for df in data_frames}

    user_info_filter = np.zeros(len(user_info), dtype=np.bool)
    for index, row in user_info.iterrows():
        task_id = "%s%d%s" % (row['userId'], row['queryId'], row['sessionId'])
        user_info_filter[index] = task_id in users_tasks
        if task_id in users_tasks and row['moduleVisibility'] == 1:
            users_tasks[task_id] = index

    return users_tasks, user_info_filter


def filter_by_user_info(user_info, data_frames, users_tasks):
    data_frames_filtered = []

    user_info_filter = np.zeros(len(user_info), dtype=np.bool)
    for index, row in user_info.iterrows():
        task_id = "%s%d%s" % (row['userId'], row['queryId'], row['sessionId'])
        user_info_filter[index] = task_id in users_tasks

    user_counts = user_info[user_info_filter]["userId"].value_counts()

    user_dupes = user_counts[user_counts > 1].keys().values.tolist()
    user_dupes_seen = []

    for index, row in user_info.iterrows():
        if row['userId'] in user_dupes:
            if row['userId'] in user_dupes_seen:
                user_info_filter[index] = False
            else:
                user_dupes_seen.append(row['userId'])

    for idx, df in enumerate(data_frames):
        task_id = df["user"].values[0] + df["query"].values[0] + df["session"].values[0]
        if users_tasks[task_id] == -1:
            continue

        user_row = user_info.iloc[users_tasks[task_id], :]

        if user_row["URL"] not in unquote(df["url"][0]):
            user_info_filter[users_tasks[task_id]] = False
            if VERBOSE:
                print("Mismatch in URL: %s vs %s (%d)" % (user_row["URL"], unquote(df["url"][0]), idx))
            continue

        data_frames_filtered.append(df)

    # print(len(data_frames_filtered))
    return data_frames_filtered


# extract data for model learning
def extract_data(data_frames, users_tasks, user_info, max_len=50, min_len=5, normalize_viewport=False,
                 normalize_1=False, reset_origin=False, normalize_time=False):
    mouse_moves = []
    mouse_moves_time = []

    attend_useful = []
    attend_faster = []
    attend_useful_faster = []

    for idx, df in enumerate(data_frames):
        pos = df[df["event"] == "mousemove"][["xpos", "ypos"]]
        pos = pos.values

        if len(pos) < min_len:
            # print("Skipping %d..." % idx)
            continue

        times = df[df["event"] == "mousemove"][["timestamp"]].values
        time_diff = [times[i + 1] - times[i] for i in range(len(times) - 1)]

        # for last move, use diff with last? or 0?
        # last_mouse_idx = df[df["event"] == "mousemove"]["timestamp"].index.values[-1]
        # if len(df["timestamp"]) > last_mouse_idx+1:
        #     next_time = df["timestamp"][last_mouse_idx+1]
        # else:
        #     print("Mouse move last event: %d" % idx)
        #     next_time = df["timestamp"][last_mouse_idx]
        # time_diff.append(next_time - times[-1])
        time_diff.append(df["timestamp"].values[-1] - times[-1])
        # time_diff.append(np.array([0]))

        if reset_origin:
            pos[:, 0] = pos[:, 0] - pos[0, 0]
            pos[:, 1] = pos[:, 1] - pos[0, 1]

        viewport = df["viewport"][0].split("x")
        if normalize_viewport:
            pos[:, 0] = pos[:, 0] / int(viewport[0]) * 1280
            # pos[:, 1] = pos[:, 1] / viewport[1] * 1024
        elif normalize_1:
            pos = np.array(pos, dtype=float)
            pos[:, 0] = pos[:, 0] / int(viewport[0])
            pos[:, 1] = pos[:, 1] / int(viewport[1])

        if normalize_time:
            time_diff = np.array(time_diff, dtype=float)
            time_diff /= 150  # normalize per event check

        pos_time = np.concatenate([pos, time_diff], axis=1)

        if len(pos) > max_len:
            # print("Cutting off %d" % idx)
            pos = pos[-max_len:]
            pos_time = pos_time[-max_len:]



        task_id = df["user"].values[0] + df["query"].values[0] + df["session"].values[0]

        if df["module_vis"].values[0] == '1':
            user_row = user_info.iloc[users_tasks[task_id], :]

            if user_row['moduleVisibility'] == 1:
                noticed = user_row["moduleWasNoticed"]
                useful = user_row["moduleWasUsefulBinary"]
                faster = user_row["moduleWasFasterBinary"]
            else:
                print("Error in df %d" % idx)
                continue
        else:
            noticed = 0
            useful = 0
            faster = 0

        if np.isnan(noticed) or np.isnan(useful):
            print("NAN Error in df %d" % idx)
            continue

        attend_useful.append(1 if noticed == 1 and useful == 1 else 0)
        attend_faster.append(1 if noticed == 1 and faster == 1 else 0)
        attend_useful_faster.append(1 if noticed == 1 and useful == 1 and faster == 1 else 0)

        mouse_moves.append(pos)
        mouse_moves_time.append(pos_time)

    return (pad_sequences(np.array(mouse_moves)), pad_sequences(np.array(mouse_moves_time)),
            np.array(attend_useful, dtype=np.int), np.array(attend_faster, dtype=np.int),
            np.array(attend_useful_faster, dtype=np.int))


def extract_data_distance(data_frames, users_tasks, user_info, max_len=50, min_len=5):
    distances = []

    for idx, df in enumerate(data_frames):
        pos = df[df["event"] == "mousemove"][["xpos", "ypos"]]
        pos = pos.values

        if len(pos) < min_len:
            print("Skipping %d..." % idx)
            continue

        extras = df[df["event"] == "mousemove"][["extras"]].values.ravel().tolist()
        extras = [json.loads(extra) for extra in extras]

        item_distances = [item['middle'] for item in extras if 'middle' in item]

        if len(item_distances) > max_len:
            item_distances = item_distances[-max_len:]

        task_id = df["user"].values[0] + df["query"].values[0] + df["session"].values[0]

        if df["module_vis"].values[0] == '1':
            user_row = user_info.iloc[users_tasks[task_id], :]

            if user_row['moduleVisibility'] == 1:
                noticed = user_row["moduleWasNoticed"]
                useful = user_row["moduleWasUsefulBinary"]
            else:
                print("Error in df %d" % idx)
                continue
        else:
            noticed = 0
            useful = 0

        if np.isnan(noticed) or np.isnan(useful):
            print("NAN Error in df %d" % idx)
            continue

        distances.append(item_distances)

    distances = pad_sequences(distances)
    return distances.reshape(-1, len(distances[0]), 1)


def extract_simple_features(args, data_frames, users_tasks, user_info):
    features = []
    labels = []

    for idx, df in enumerate(data_frames):
        # no of hovers
        mousemove_rows = df[df["event"] == "mousemove"]
        extras = mousemove_rows[["extras"]].values.ravel().tolist()
        extras = [json.loads(extra) for extra in extras if extra != '{}']
        hovered = np.count_nonzero([has_interaction(extra) for extra in extras])

        # no of scrolls
        scrolls = len(df[df["event"] == "scroll"])

        pos = mousemove_rows[["xpos", "ypos"]].values
        if len(pos) < args.min_events:
            print("Skipping %d..." % idx)
            continue

        # scroll distance vertical
        vert_range = np.max(pos[:,1]) - np.min(pos[:,1])

        # scroll distance horizontal
        hor_range = np.max(pos[:,0]) - np.min(pos[:,0])

        # dwell time
        timestamps = df["timestamp"].values
        dwell_time = timestamps[-1] - timestamps[0]

        # avg time offset
        times = mousemove_rows[["timestamp"]].values
        time_diff = [times[i + 1] - times[i] for i in range(len(times) - 1)]
        avg_time_diff = np.average(time_diff)

        # no of mouse moves
        mousemove_count = len(mousemove_rows)

        # traj length
        traj_length = 0
        prev_coord = pos[0]

        for coord in pos[1:,]:
            traj_length += distance.euclidean(prev_coord, coord)
            prev_coord = coord

        # vertical reach
        document = df["document"][0].split("x")
        max_reach_vert = np.max(pos[:,1]) / int(document[1])

        # horizontal reach
        max_reach_hor = np.max(pos[:,0]) / int(document[0])


        task_id = df["user"].values[0] + df["query"].values[0] + df["session"].values[0]
        user_row = user_info.iloc[users_tasks[task_id], :]

        if df["module_vis"].values[0] == '1':

            if user_row['moduleVisibility'] == 1:
                noticed = user_row["moduleWasNoticed"]
                useful = user_row["moduleWasUsefulBinary"]
            else:
                print("Error in df %d" % idx)
                continue
        else:
            noticed = 0
            useful = 0

        if np.isnan(noticed) or np.isnan(useful):
            print("NAN Error in df %d" % idx)
            continue

        features.append([hovered, scrolls, dwell_time, vert_range, hor_range, max_reach_hor, max_reach_vert,
                         traj_length, avg_time_diff, mousemove_count])
        labels.append(noticed and useful)

    return np.array(features), np.array(labels)


# Estimates whether user has interacted with the knowledge module.
# This function was ported from rectangles.cpp
# and takes as input 4 distances: the first 3 of which are related to 3 corners, the last one the middle
def has_interaction(graph):
    if not graph:
        return False
    dA, dB, dC, dX = graph['topLeft'], graph['topRight'], graph['bottomLeft'], graph['middle']
    alpha = dA**2                                        # a^2 + b^2
    beta  = dB**2 - alpha                                # x^2 + 2ax
    gamma = dC**2 - alpha                                # y^2 - 2by
    delta = 2 * dX**2 - (2 * alpha + (beta + gamma) / 2) #  ax - by
    eps = 1e-7
    ret = ''
    if abs(beta) < eps and abs(gamma) < eps:
        return 'INSIDE'

    def solve_quadratic(a,b,c):
        if abs(a) < eps:
            return -c/b
        x = b * b - 4 * a * c
        if -eps * (abs(a) + abs(b) + abs(c)) < x and x <= 0:
            x = 0
        if x < 0:
            return []
        if abs(x) < eps:
            return [ -b/(2 * a) ]
        x = math.sqrt(x)
        return [ (-b + x)/(2 * a), (-b - x)/(2 * a) ]

    sol_y2 = solve_quadratic(
        beta + gamma + 2 * delta + 4 * alpha,
        8 * alpha * delta - 2 * beta * gamma - 4 * alpha * beta - 4 * alpha * gamma - 2 * gamma * gamma + 4 * delta * delta,
        gamma**2 * (gamma + beta - 2 * delta)
    )
    sol_y = [ math.sqrt(y2) for y2 in sol_y2 if y2 > eps ]
    if not sol_y:
        return 'NO_INTERACTION' # Actually it's unclear how to handle this case.

    for y in sol_y:
        ax = delta + (y**2 - gamma) / 2
        x2 = beta - 2 * ax
        if x2 > eps:
            x = math.sqrt(x2)
            b = (y**2 - gamma) / (2 * y)
            a = ax/x
            inside = (-x <= a and a <= 0 and 0 <= b and b <= y)
            r = 'INSIDE' if inside else 'OUTSIDE'
            if ret == '':
                ret = r
            elif ret != r:
                ret = 'NO_INTERACTION' # Actually it's unclear how to handle this case.
    return ret if ret != -1 else 'NO_INTERACTION'


def calc_velocity(mouse_moves_time):
    velocities = []

    for item in mouse_moves_time:
        velocity_seq = []
        for idx, (x, y, time_diff) in enumerate(item):
            if idx == len(item)-1:
                velocity_seq.append(0)
            elif x == y == 0:
                velocity_seq.append(0)
                # velocity_seq.append(-1)
            else:
                next_x, next_y, _ = item[idx+1]

                item_dist = distance.euclidean([x, y], [next_x, next_y])
                velocity = item_dist / time_diff
                velocity_seq.append(velocity)

        velocities.append(velocity_seq)

    return np.array(velocities).reshape(-1, len(item), 1)
