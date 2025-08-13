# Copyright 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: Apache-2.0

from .interleave_datasets import UnifiedEditIterableDataset
from .t2i_dataset import T2IIterableDataset
from .vlm_dataset import SftJSONLIterableDataset
from .interleave_datasets.think_trace_dataset import ThinkTraceJSONLIterableDataset


DATASET_REGISTRY = {
    't2i_pretrain': T2IIterableDataset,
    'vlm_sft': SftJSONLIterableDataset,
    'unified_edit': UnifiedEditIterableDataset,
    'think_trace': ThinkTraceJSONLIterableDataset,
    # new enteries for direct sft, textual_cot, visual_cot
    'sft': SftJSONLIterableDataset,
    'textual_cot': SftJSONLIterableDataset,
    'visual_cot': ThinkTraceJSONLIterableDataset,
}

DATASET_INFO = {
    'sft': {
        'frozenlake_sft': {
            'data_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
            'jsonl_path': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/frozenlake/train/frozenlake_thinktrace.jsonl',
        },
        'maze_small_sft': {
            'data_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
            'jsonl_path': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/maze_small/train/maze_small_thinktrace.jsonl',
        },
        'tetris_small_sft': {
            'data_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
            'jsonl_path': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/tetris_small/train/tetris_small_thinktrace.jsonl',
        },
        # Can add chess_sft, other_task_sft here later
    },
    'textual_cot': {
        'frozenlake_textual_cot': {
            'data_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
            'jsonl_path': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/frozenlake/train/frozenlake_thinktrace.jsonl',
        },
        'maze_small_textual_cot': {
            'data_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
            'jsonl_path': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/maze_small/train/maze_small_thinktrace.jsonl',
        },
        'tetris_small_textual_cot': {
            'data_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
            'jsonl_path': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/tetris_small/train/tetris_small_thinktrace.jsonl',
        },
        # Can add chess_textual_cot, other_task_textual_cot here later
    },
    'visual_cot': {
        'frozenlake_visual_cot': {
            'data_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
            'jsonl_path': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/frozenlake/train/frozenlake_thinktrace.jsonl',
            'image_prefix_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
        },
        'maze_small_visual_cot': {
            'data_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
            'jsonl_path': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/maze_small/train/maze_small_thinktrace.jsonl',
            'image_prefix_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
        },
        'tetris_small_visual_cot': {
            'data_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
            'jsonl_path': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/tetris_small/train/tetris_small_thinktrace.jsonl',
            'image_prefix_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
        },
        # Can add chess_visual_cot, other_task_visual_cot here later
    },
    #old misc enteries
    'think_trace': {
        'think_trace_dataset': {
            'data_dir': '/dev/shm/data/Zebra-CoT/zebra-cot-images',
            'jsonl_path': '/dev/shm/data/Zebra-CoT/zebra_cot.jsonl',
            'image_prefix_dir': '/dev/shm/data/Zebra-CoT',  # Base path for relative image paths
            'num_total_samples': 1000,
        },
        'chess_thinktrace': {
            'data_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/datasets/',
            'jsonl_path': '/home/colligo/project/vlm/Bagel-Zebra-CoT/datasets/chess_thinktrace.jsonl',
            'image_prefix_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/datasets/chess_thinktrace_images',
            'num_total_samples': 1000,  # 1000 samples for training
        },
        'frozenlake_thinktrace': {
            'data_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
            'jsonl_path': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data/data/frozenlake/train/frozenlake_thinktrace.jsonl',
            'image_prefix_dir': '/home/colligo/project/vlm/Bagel-Zebra-CoT/visual_data',
            'num_total_samples': None,  # Use all samples
        },
    },
}