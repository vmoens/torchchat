# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import pathlib

import pandas as pd


def flatten(data):
    result = []
    for device in data['devices']:
        for model_type in device['model_types']:
            for execution_mode in model_type['execution_modes']:
                quant_schemes = execution_mode['quantization_options']['quant_schemes']
                embedding_quant_schemes = execution_mode['quantization_options']['embedding_quant_schemes']
                for quant_scheme in quant_schemes:
                    for weight_dtype in quant_scheme.get('weight_dtypes', []):
                        result.append({
                            'device': device['name'],
                            'model_type': model_type['type'],
                            'execution_mode': execution_mode['mode'],
                            'quant_scheme': quant_scheme['scheme'],
                            'weight_dtype': weight_dtype,
                            'activation_dtype': None,
                            'weight_group_size': execution_mode['quantization_options']['weight_group_sizes'][0],
                            'embedding_group_size': execution_mode['quantization_options']['embedding_group_sizes'][0]
                        })
                    for activation_dtype in quant_scheme.get('activation_dtypes', []):
                        result.append({
                            'device': device['name'],
                            'model_type': model_type['type'],
                            'execution_mode': execution_mode['mode'],
                            'quant_scheme': quant_scheme['scheme'],
                            'weight_dtype': None,
                            'activation_dtype': activation_dtype,
                            'weight_group_size': execution_mode['quantization_options']['weight_group_sizes'][0],
                            'embedding_group_size': execution_mode['quantization_options']['embedding_group_sizes'][0]
                        })
                for embedding_quant_scheme in embedding_quant_schemes:
                    for weight_dtype in embedding_quant_scheme.get('weight_dtipes', []):
                        result.append({
                            'device': device['name'],
                            'model_type': model_type['type'],
                            'execution_mode': execution_mode['mode'],
                            'quant_scheme': embedding_quant_scheme['scheme'],
                            'weight_dtype': weight_dtype,
                            'activation_dtype': None,
                            'weight_group_size': execution_mode['quantization_options']['weight_group_sizes'][0],
                            'embedding_group_size': execution_mode['quantization_options']['embedding_group_sizes'][0]
                        })
    return result


def get_pandas_df(file=None):
    if file is None:
        file = pathlib.Path(__file__).parent / "quant.json"
    with open(file) as f:
        data = json.load(f)
    result = flatten(data)
    return pd.DataFrame(result)


def select_in_df(df, **kwargs):
    id = True
    for field, value in kwargs.items():
        id = (df[field] == value) & id
    return df.loc[id]
