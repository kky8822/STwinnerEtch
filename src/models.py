import pandas as pd
import torch

from src.stwinner_model.models import STwinnerEtchModel


class Models:
    def __init__(self, task_info, mode):
        self.mode = mode
        self.cols_x = task_info.get("cols_x", {})
        self.cols_y = task_info.get("cols_y", {})
        self.h_param = task_info.get("h_param", {})
        self.result_path = task_info.get("result_path", "")
        self.model_path = task_info.get("model_path", "")
        self.data_path = task_info.get("data_path", "")

        self.model = None
        self.step_sequence = task_info.get("step_sequence", [])
        self.step_plasma_label = task_info.get("step_plasma_label", [])
        self.step_etch_label = task_info.get("step_etch_label", [])

        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    def get_data(self):
        self.df = pd.read_csv(self.data_path)

        n_grid = self.h_param["etch_Etch"]["n_grid"]
        prof_in_cols = [
            f"{col}_{self.step_sequence[0]}"
            for col in self.cols_x["etch_Etch"][:n_grid]
        ]
        rcp_cols = [
            f"{col}_{step}"
            for step in self.step_sequence
            for col in self.cols_x["plasma_Plasma"]
        ]
        time_cols = [f"etch_time_{step}" for step in self.step_sequence]
        input_cols = prof_in_cols + rcp_cols + time_cols
        input_data = self.df[input_cols].values

        return input_data

    def load(self):
        self.build()
        torch.save(self.model.state_dict(), self.model_path)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True, map_location=self.device))

    def build(self):
        self.model = STwinnerEtchModel(self.h_param)
        self.model.build(self.cols_x, self.cols_y)

    def predict(self, input_data):
        self.model = self.model.to(self.device)
        x = torch.tensor(input_data).float().to(self.device)
        y = self.model.forward(x).detach().cpu().numpy()

        mts_cols = [f"mts{i}_{step}" for step in self.step_sequence for i in range(len(self.h_param["etch_Etch"]["mts_idx"]))]
        prof_cols = [f"{col}_{step}" for step in self.step_sequence for col in self.cols_y["etch_Etch"]]
        flux_cols = [f"{col}_{step}" for step in self.step_sequence for col in self.cols_y["plasma_Plasma"]]
        columns = mts_cols + prof_cols + flux_cols

        df_out = pd.DataFrame(columns=columns, data=y)
        df_out.to_csv(self.result_path, index=False)

