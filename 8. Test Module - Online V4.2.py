import os

# ---- HARD SAFETY LIMITS (before ANY heavy imports) ----
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # OK to keep on

import threading
import queue
import time
from datetime import datetime, timedelta

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")   # or "Agg" if no GUI
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from pyModbusTCP.client import ModbusClient

import tensorflow as tf


class TensorFlowWorker(threading.Thread):
    def __init__(self, model_path, scaler_path, config_path, in_q, out_q):
        super().__init__(daemon=True)
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.config_path = config_path
        self.in_q = in_q
        self.out_q = out_q

    def run(self):
        print("🧠 TF thread starting...")

        #import tensorflow as tf  # Safe: import inside thread

        # Load configuration
        config = joblib.load(self.config_path)
        self.SEQ_LEN = config["seq_len"]
        self.FEATURES = config["features"]
        self.TARGET = config["target"]
        self.ALL_COLUMNS = self.FEATURES + [self.TARGET]

        # Load scaler and model
        self.scaler = joblib.load(self.scaler_path)
        self.model = tf.keras.models.load_model(self.model_path, compile=False)

        # Warm-up prediction
        dummy_input = np.zeros((1, self.SEQ_LEN, len(self.FEATURES)))
        self.model.predict(dummy_input, verbose=0)
        print("✅ TF model loaded & warmed")

        while True:
            buffer = self.in_q.get()  # blocks until data available
            if buffer is None:  # signal to stop thread
                break

            # Convert buffer to DataFrame and select last SEQ_LEN rows
            df_input = pd.DataFrame(buffer).tail(self.SEQ_LEN)[self.ALL_COLUMNS]

            # Scale features + target
            scaled = self.scaler.transform(df_input)

            # Prepare LSTM input: only features, add batch dimension
            X = scaled[:, :len(self.FEATURES)]
            X = X[np.newaxis, :, :]  # shape = (1, SEQ_LEN, n_features)

            # Predict scaled target
            pred_scaled = self.model.predict(X, verbose=0)[0][0]

            # Inverse transform to actual temperature
            dummy = np.zeros((1, len(self.ALL_COLUMNS)))
            dummy[0, -1] = pred_scaled
            pred_actual = self.scaler.inverse_transform(dummy)[0, -1]

            # Send prediction back
            self.out_q.put(pred_actual)


def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    MODEL_PATH  = os.path.join(BASE_DIR, "heater_forecast_model.h5")
    SCALER_PATH = os.path.join(BASE_DIR, "heater_forecast_model_scaler.pkl")
    CONFIG_PATH = os.path.join(BASE_DIR, "heater_forecast_model_config.pkl")

    # Load config
    config = joblib.load(CONFIG_PATH)
    SEQ_LEN = config["seq_len"]
    FORECAST_GAP = config["forecast_gap"]

    in_q = queue.Queue(maxsize=1)
    out_q = queue.Queue(maxsize=1)

    tf_worker = TensorFlowWorker(MODEL_PATH, SCALER_PATH, CONFIG_PATH, in_q, out_q)
    tf_worker.start()

    c = ModbusClient(host="10.128.25.11", port=502, auto_open=True)

    data_buffer = []
    predictions = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    line_actual, = ax.plot([], [], label="Actual")
    line_pred, = ax.plot([], [], "--", label="Forecast")
    ax.legend()

    print("🚀 Monitoring started")

    while True:
        ts = datetime.now()

        # Read Modbus registers
        regs = c.read_holding_registers(100, 20)
        if not regs:
            time.sleep(5)
            continue

        row = {
            "fDateTime": ts,
            "InletT_PV": regs[1] / 10,
            "InletT_SV": regs[2] / 10,
            "OutletT_SV": regs[4] / 10,
            "FlueGasT_PV": regs[5] / 10,
            "FlueGasT_SV": regs[6] / 10,
            "Flow_PV": regs[7] / 10,
            "RealPower": regs[16] * 3 / 50,
            "OutletT_PV": regs[3] / 10,
        }

        data_buffer.append(row)
        if len(data_buffer) > 120:  # keep buffer limited
            data_buffer.pop(0)

        # ---- Send buffer to TF thread ----
        if len(data_buffer) >= SEQ_LEN and in_q.empty():
            in_q.put(list(data_buffer))

        # ---- Receive prediction ----
        if not out_q.empty():
            pred = out_q.get()
            future_ts = ts + timedelta(minutes=FORECAST_GAP)
            predictions.append((future_ts, pred))
            print(f"[{ts:%H:%M:%S}] Forecast (t+{FORECAST_GAP}min): {pred:.1f}°C")

        # ---- Plot ----
        df_plot = pd.DataFrame(data_buffer)
        df_plot["fDateTime"] = pd.to_datetime(df_plot["fDateTime"])

        # Update actual line
        line_actual.set_data(df_plot["fDateTime"], df_plot["OutletT_PV"])

        # Update prediction line
        if predictions:
            t_pred, v_pred = zip(*predictions)
            line_pred.set_data(list(t_pred), list(v_pred))

        # Update axes
        if len(df_plot) >= 2:
            x_min = df_plot["fDateTime"].iloc[0]
            x_max = max(df_plot["fDateTime"].iloc[-1], predictions[-1][0] if predictions else df_plot["fDateTime"].iloc[-1])
            ax.set_xlim(x_min, x_max)
        ax.set_ylim(150, 250)

        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        fig.autofmt_xdate()
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(60)


if __name__ == "__main__":
    main()

    
