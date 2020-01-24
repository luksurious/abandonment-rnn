import numpy as np
import glob
import pandas as pd
from xml.dom import minidom
import io
import re
import json
from urllib.parse import unquote
import pickle

from keras_preprocessing.sequence import pad_sequences

PATH = ''


def read_kge_log():
    with open(PATH + "kge/kge.log", 'r') as f:
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

    return user_info


def read_csv_logs(user_info):
    try:
        data_frames = pickle.load(open("mouse_logs.pkl", "rb"))
        return data_frames
    except Exception:
        pass

    user_ids = user_info["userId"]

    data_frames = []

    quote_replacement = '|'

    error_count = 0

    file_user_map = {}

    user_ids_set = set()

    pattern = PATH + 'kge/cursorlogs/*.csv'
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

            if "search.yahoo.com" not in url:
                continue

            df["session"] = task_data[0]
            df["query"] = task_data[1]
            df["user"] = task_data[2]
            df["module_vis"] = task_data[3]
            df["file"] = file_id
            df["url"] = url
            data_frames.append(df)
        except Exception as e:
            print(f"\nError in {file}: {e}")
            error_count += 1

    print("\nWith %s got %d errors" % (quote_replacement, error_count))

    with open('user-file-map.json', 'w') as fp:
        json.dump(file_user_map, fp)

    pickle.dump(data_frames, open("mouse_logs.pkl", "wb"))

    return data_frames


def filter_data_frames(data_frames):
    data_no_clicks = [df for df in data_frames if len(df[df["event"] == "click"]) == 0
                      and len(df[df["event"] == "mousemove"]) > 0]

    print("Frames without clicks: %d" % len(data_no_clicks))

    data_nc_with_km = [df for df in data_no_clicks if df["module_vis"].values[0] == '1']
    data_nc_without_km = [df for df in data_no_clicks if df["module_vis"].values[0] == '0']

    print("Frames without clicks with KM: %d" % len(data_nc_with_km))
    print("Frames without clicks without KM: %d" % len(data_nc_without_km))

    data_nc_with_km_sr = [df for df in data_nc_with_km if "search.yahoo.com" in df["url"].values[0]]
    data_nc_without_km_sr = [df for df in data_nc_without_km if "search.yahoo.com" in df["url"].values[0]]

    print("Frames without clicks with KM on search page: %d" % len(data_nc_with_km_sr))
    print("Frames without clicks without KM on search page: %d" % len(data_nc_without_km_sr))

    return data_nc_with_km_sr, data_nc_without_km_sr


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

    return users_tasks


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

    len(user_info[user_info_filter])

    for idx, df in enumerate(data_frames):
        task_id = df["user"].values[0] + df["query"].values[0] + df["session"].values[0]
        if users_tasks[task_id] == -1:
            continue

        user_row = user_info.iloc[users_tasks[task_id], :]

        if user_row["URL"] not in unquote(df["url"][0]):
            user_info_filter[users_tasks[task_id]] = False
            print("Mismatch in URL: %s vs %s (%d)" % (user_row["URL"], unquote(df["url"][0]), idx))
            continue

        data_frames_filtered.append(df)

    # print(len(data_frames_filtered))
    return data_frames_filtered


# extract data for model learning
def extract_data(data_frames, users_tasks, user_info, max_len=50, min_len=5):
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
        time_diff.append(df["timestamp"].values[-1] - times[-1])

        pos_time = np.concatenate([pos, time_diff], axis=1)

        # test cutting off mouse moves
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

