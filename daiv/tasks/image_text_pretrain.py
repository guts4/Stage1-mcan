"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from daiv.common.registry import registry
from daiv.tasks.base_task import BaseTask


@registry.register_task("image_text_pretrain")
class ImageTextPretrainTask(BaseTask):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def valid_step(self, model, data_loader):
        #task_cfg = self.config.task_cfg  # Access the task_cfg from the config
        #k_test = task_cfg.get("k_test", 5)  # Default value for k_test if not provided
        k_test=10
        
        sim_matrix = model.compute_sim_matrix(data_loader, k_test)
        print("Sim Matrix:", sim_matrix)

        return {"sim_matrix": sim_matrix}

    def after_evaluation(self, val_result, split_name, **kwargs):
        result_file = self.save_result(
            val_result,
            result_dir=registry.get_path("result_dir"),
            filename=f"{split_name}_vqa_result",
            remove_duplicate="question_id",
        )

        metrics = self._report_metrics(result_file=result_file, split=split_name)
        print('metrics:', metrics)

        return metrics

    @classmethod
    def setup_task(cls, cfg):
        return cls(cfg)
