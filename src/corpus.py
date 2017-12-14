# -*- coding: UTF-8 -*-

import sys
import numpy as np

from src.common import io


class Corpus:
    def __init__(self):
        self.n_users = 0
        self.n_items = 0
        self.n_clicks = 0
        self.user2id = dict()
        self.item2id = dict()
        self.id2user = dict()
        self.id2item = dict()
        self.pos_per_user = list()

    def load_data(self, data_path, user_min, item_min):
        print("  Loading clicks from {}, userMin = {}  itemMin = {} ".format(data_path, user_min, item_min), end='')
        self._load_clicks(data_path, user_min, item_min)
        self._show_user_seq(10)
        print("\n  \"n_users\": {}, \"n_items\": {}, \"n_clicks\": {}".format(self.n_users, self.n_items, self.n_clicks))

    def _load_clicks(self, data_path, user_min, item_min):
        # First pass, count interactions for each user/item
        f = io.read_gz_file(data_path)
        user_cnt, item_cnt = dict(), dict()
        total_clicks = 0
        if getattr(f, '__iter__', None):
            for n_read, line in enumerate(f):
                if n_read % 100000 == 0:
                    print('.', end='')
                    sys.stdout.flush()
                if len(line.split()) != 4:
                    continue
                [user_name, item_name, value, vote_time] = line.split()
                try:
                    user_name, item_name, value, vote_time = \
                        user_name.strip(), item_name.strip(), float(value), int(vote_time)
                except ValueError:
                    continue
                if user_name not in user_cnt:
                    user_cnt[user_name] = 0
                if item_name not in item_cnt:
                    item_cnt[item_name] = 0
                user_cnt[user_name] += 1
                item_cnt[item_name] += 1
                total_clicks = n_read
        print("\n  First pass: #users = {}, #items = {}, #clicks = {} ".format(
            len(user_cnt), len(item_cnt), total_clicks), end='')

        # Second pass, allocate id for user/item which satisfy the limit of user_min/item_min
        f2 = io.read_gz_file(data_path)
        if getattr(f2, '__iter__', None):
            for n_read, line in enumerate(f2):
                if n_read % 100000 == 0:
                    print('.', end='')
                    sys.stdout.flush()
                if len(line.split()) != 4:
                    continue
                [user_name, item_name, value, vote_time] = line.split()
                try:
                    user_name, item_name, value, vote_time = \
                        user_name.strip(), item_name.strip(), float(value), int(vote_time)
                except ValueError:
                    continue
                if user_cnt[user_name] < user_min or item_cnt[item_name] < item_min:
                    continue
                self.n_clicks += 1
                if user_name not in self.user2id:
                    self.user2id[user_name] = self.n_users
                    self.id2user[self.n_users] = user_name
                    self.pos_per_user.append([])
                    self.n_users += 1
                if item_name not in self.item2id:
                    self.item2id[item_name] = self.n_items
                    self.id2item[self.n_items] = item_name
                    self.n_items += 1
                self.pos_per_user[self.user2id[user_name]].append({'item': self.item2id[item_name],
                                                                   'time': vote_time})

        # TODO: whether to romove users whose valid clicks are less than 5

        # Rand clicks for each user according to timestamp
        # TODO: multi-thread sorting
        print("\n  Sorting clicks for each user ", end='')
        for u in range(self.n_users):
            self.pos_per_user[u].sort(key=lambda x: x['time'])
            if u % 10000 == 0:
                print('.', end='')
                sys.stdout.flush()

        print()

    def _show_user_seq(self, num):
        # Observe whether user click sequence is healthy
        for i in range(num):
            u = np.random.randint(0, self.n_users)
            print("    user {:<7}: ".format(u), end='')
            for record in self.pos_per_user[u]:
                print("{:<6} ".format(record['time']), end='')
            print()
