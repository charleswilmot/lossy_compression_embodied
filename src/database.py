import sqlite3 as sql
import numpy as np
import pandas as pd
import io


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sql.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


# Converts np.array to TEXT when inserting
sql.register_adapter(np.ndarray, adapt_array)
# Converts TEXT to np.array when selecting
sql.register_converter("array", convert_array)


class ResultDatabase:
    def __init__(self, path):
        print('opening database at {}'.format(path))
        self.path = path
        self.conn = sql.connect(path, detect_types=sql.PARSE_DECLTYPES)
        self.cursor = self.conn.cursor()
        command = '''SELECT count(name) FROM sqlite_master WHERE type='table' AND name='results';'''
        self.cursor.execute(command)
        command = '''CREATE TABLE IF NOT EXISTS results (
                     date_time DATETIME NOT NULL,
                     collection TEXT NOT NULL,
                     experiment_type TEXT NOT NULL,
                     bottleneck_size INTEGER NOT NULL,
                     pre_encoding_size INTEGER NOT NULL,
                     arm0_end_eff_readout FLOAT NOT NULL,
                     arm0_positions_readout FLOAT NOT NULL,
                     arm0_velocities_readout FLOAT NOT NULL,
                     arm1_end_eff_readout FLOAT NOT NULL,
                     arm1_positions_readout FLOAT NOT NULL,
                     arm1_velocities_readout FLOAT NOT NULL,
                     frame_error_map array NOT NULL,
                     full_conf VARBINARY(2048) NOT NULL
                  );'''
        self.cursor.execute(command)

    def insert(self,
            date_time,
            collection,
            experiment_type,
            bottleneck_size,
            pre_encoding_size,
            arm0_end_eff_readout,
            arm0_positions_readout,
            arm0_velocities_readout,
            arm1_end_eff_readout,
            arm1_positions_readout,
            arm1_velocities_readout,
            frame_error_map,
            full_conf):
        results = (
            date_time,
            collection,
            experiment_type,
            bottleneck_size,
            pre_encoding_size,
            arm0_end_eff_readout,
            arm0_positions_readout,
            arm0_velocities_readout,
            arm1_end_eff_readout,
            arm1_positions_readout,
            arm1_velocities_readout,
            frame_error_map,
            full_conf,
        )
        command = '''INSERT INTO results(
                        date_time,
                        collection,
                        experiment_type,
                        bottleneck_size,
                        pre_encoding_size,
                        arm0_end_eff_readout,
                        arm0_positions_readout,
                        arm0_velocities_readout,
                        arm1_end_eff_readout,
                        arm1_positions_readout,
                        arm1_velocities_readout,
                        frame_error_map,
                        full_conf)
                     VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)'''
        self.cursor.execute(command, results)

    def get_dataframe(self, command):
        return pd.read_sql(command, self.conn)

    def commit(self):
        self.conn.commit()



if __name__ == '__main__':
    from datetime import datetime

    db = ResultDatabase('/tmp/test3.db')



    db.insert(
        date_time = datetime(2021, 11, 15, 4, 30, 0),
        collection = 'my_collection',
        experiment_type = 'my_epx_type',
        bottleneck_size = 123,
        pre_encoding_size = 456,
        arm0_end_eff_readout = 1.2,
        arm0_positions_readout = 2.3,
        arm0_velocities_readout = 3.4,
        arm1_end_eff_readout = 4.5,
        arm1_positions_readout = 5.6,
        arm1_velocities_readout = 6.7,
        frame_error_map = np.random.randint(0, 10, size=(4, 5)),
        full_conf = b'aze',
    )
    db.conn.commit()

    results = db.get_dataframe('SELECT * FROM results')
    print(results)
